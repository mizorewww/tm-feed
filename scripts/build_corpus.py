#!/usr/bin/env python3
"""
Build static HTML corpus & hybrid recommender assets for tm-feed from:
- L-M-Sherlock/tm-index (CC0 1.0)
- L-M-Sherlock/ZhiHuArchive (CC0 1.0)

Every post is rendered as a standalone static HTML page (curl-friendly, zero JS required to read):
- public/index.html             (static homepage with pre-rendered initial feed + search input)
- public/posts/<id>.html        (3,580 complete static HTML articles & answers + 6 static bottom recs)
- public/data/feed-index.json   (compact metadata for client-side history recommender & instant search)
- public/data/embeddings.bin    (3,580 x 128 Int8 L2-normalized hybrid semantic+graph vectors)
- public/data/neighbors.json    (Top-12 cross-item graph + co-citation neighbors per item index)
"""

from __future__ import annotations

import collections
import datetime as dt
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse, urlsplit

import numpy as np
import scipy.sparse as sp
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TM_INDEX_DIR = Path("/tmp/research/tm-index-rec/repos/tm-index/data")
ARCHIVE_DIR = Path("/tmp/research/tm-index-rec/repos/ZhiHuArchive")
OUT_PUBLIC = Path(__file__).resolve().parent.parent / "public"
POSTS_DIR = OUT_PUBLIC / "posts"
DATA_DIR = OUT_PUBLIC / "data"

SHANGHAI_TZ = dt.timezone(dt.timedelta(hours=8))
BOTTOM_REC_LIMIT = 6

AUTHOR_PROFILE_URLS = {
    "4c592f496dc33822b560b382907ff1d0": "https://www.zhihu.com/people/L.M.Sherlock",
    "3c9990a12cdbcd92e20b1387b160f0a3": "https://www.zhihu.com/people/JarrettYe",
}

ZHIHU_ITEM_RE = re.compile(
    r"^https?://(?:zhuanlan\.zhihu\.com/p/(\d+)|www\.zhihu\.com/question/\d+/answer/(\d+)|l-m-sherlock\.github\.io/ZhiHuArchive/(\d+)\.html)(?:[/?#].*)?$",
    re.I,
)


def clean_ws(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = BeautifulSoup(str(value), "lxml").get_text(" ", strip=True)
    return clean_ws(text)


def truncate_hook(value: str, limit: int = 165) -> str:
    text = clean_ws(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" ,.;，。；、：:") + "…"


def ts_to_date(ts: int | float | None, fallback: str = "") -> str:
    if not ts:
        return fallback
    try:
        return dt.datetime.fromtimestamp(int(ts), SHANGHAI_TZ).strftime("%Y-%m-%d")
    except Exception:
        return fallback


def normalize_href(raw_url: str, valid_ids: set[str]) -> tuple[str, str | None]:
    """Return (cleaned_href_for_posts_dir, matched_internal_id_or_None)."""
    if not raw_url:
        return "", None
    url = raw_url.strip()
    if url.startswith("//"):
        url = "https:" + url
    for bad in ("ttps://", "htps://", "ihttps://"):
        if url.startswith(bad):
            url = "https://" + url[len(bad) :]
            break

    if "link.zhihu.com" in url:
        try:
            parsed = urlparse(url)
            target = parse_qs(parsed.query).get("target", [None])[0]
            if target:
                url = unquote(target)
        except Exception:
            pass

    for bad in ("ttps://", "htps://", "ihttps://"):
        if url.startswith(bad):
            url = "https://" + url[len(bad) :]
            break

    m = ZHIHU_ITEM_RE.match(url)
    if m:
        matched_id = m.group(1) or m.group(2) or m.group(3)
        if matched_id in valid_ids:
            return f"{matched_id}.html", matched_id

    m_rel = re.match(r"^\./(\d+)\.html$", url)
    if m_rel and m_rel.group(1) in valid_ids:
        mid = m_rel.group(1)
        return f"{mid}.html", mid

    parsed_split = urlsplit(url)
    if parsed_split.netloc in ("", "api.zhihu.com", "www.zhihu.com"):
        pm = re.fullmatch(r"/(?:api/v4/)?people/([^/]+)/?", parsed_split.path)
        if pm and pm.group(1) in AUTHOR_PROFILE_URLS:
            return AUTHOR_PROFILE_URLS[pm.group(1)], None

    return url, None


def apply_restoration(raw_html: str, restoration: dict | None) -> tuple[str, dict | None]:
    if not restoration or not isinstance(restoration, dict):
        return raw_html, None
    try:
        soup = BeautifulSoup(raw_html, "lxml")
        for op in restoration.get("operations", []):
            after_text = clean_ws(op.get("after_text", ""))
            tag_name = op.get("after_tag", "p")
            anchor = next(
                (
                    el
                    for el in soup.find_all(tag_name)
                    if clean_ws(el.get_text(" ", strip=True)) == after_text
                ),
                None,
            )
            if anchor is None:
                continue
            if op.get("remove_following_empty_paragraphs"):
                sib = anchor.find_next_sibling()
                while sib and sib.name == "p" and not clean_ws(sib.get_text(" ", strip=True)):
                    nxt = sib.find_next_sibling()
                    sib.decompose()
                    sib = nxt
            frag = BeautifulSoup(op.get("html", ""), "lxml")
            body_nodes = list(frag.body.contents) if frag.body else list(frag.contents)
            ins = anchor
            for node in body_nodes:
                ins.insert_after(node)
                ins = node
            if op.get("replace_anchor"):
                anchor.decompose()
        out = "".join(str(x) for x in soup.body.contents) if soup.body else str(soup)
        notice = {
            "label": restoration.get("label", "存档补全"),
            "notice": restoration.get("notice", ""),
        }
        return out, notice
    except Exception:
        return raw_html, None


def transform_html(raw_html: str, valid_ids: set[str]) -> tuple[str, list[dict], int]:
    if not raw_html:
        return "", [], 0

    soup = BeautifulSoup(raw_html, "lxml")
    root = soup.body if soup.body else soup

    for u in root.find_all("u"):
        u.unwrap()

    for img in root.find_all("img"):
        actual = img.get("data-actualsrc") or img.get("data-original")
        if actual:
            img["src"] = actual

        for attr in ("data-actualsrc", "data-original", "data-rawwidth", "data-rawheight", "data-size", "data-caption"):
            img.attrs.pop(attr, None)

        final_src = img.get("src", "")
        if not final_src or final_src.startswith("data:image/svg+xml"):
            if not img.get("eeimg") and "equation" not in final_src:
                img.decompose()
                continue

        if img.get("eeimg") == "1" or "zhihu.com/equation" in final_src:
            img["class"] = list(set(img.get("class", []) + ["math-eq"]))
            img["loading"] = "lazy"
            img["referrerpolicy"] = "no-referrer"
        else:
            img["loading"] = "lazy"
            img["decoding"] = "async"
            img["referrerpolicy"] = "no-referrer"

    footnotes: dict[int, dict] = {}
    for sup in root.find_all("sup"):
        text = sup.get("data-text")
        url = sup.get("data-url")
        numero = sup.get("data-numero")
        if numero and (text or url):
            try:
                n = int(numero)
            except ValueError:
                continue
            clean_u, matched_id = normalize_href(url or "", valid_ids)
            footnotes[n] = {
                "text": clean_ws(text or url or f"参考 {n}"),
                "url": clean_u,
                "tm_id": matched_id,
            }
            sup.attrs.clear()
            sup["class"] = ["cite"]
            sup["id"] = f"cite-{n}"
            sup.clear()
            a_tag = soup.new_tag("a", href=f"#ref-{n}")
            a_tag.string = f"[{n}]"
            sup.append(a_tag)

    for a in root.find_all("a"):
        href = a.get("href", "")
        if href.startswith("#ref-") or href.startswith("#cite-"):
            continue
        clean_u, matched_id = normalize_href(href, valid_ids)
        if not clean_u:
            continue
        a["href"] = clean_u
        for attr in ("data-draft-node", "data-draft-type", "data-card-oid", "data-za-not-track-link"):
            a.attrs.pop(attr, None)
        if matched_id:
            a["data-tm-id"] = matched_id
            cls = a.get("class", [])
            if isinstance(cls, str):
                cls = cls.split()
            if "tm-internal" not in cls:
                cls.append("tm-internal")
            a["class"] = cls
            a.attrs.pop("target", None)
            a.attrs.pop("rel", None)
        elif clean_u.startswith("http"):
            a["target"] = "_blank"
            a["rel"] = "noopener noreferrer"

    for h1 in root.find_all("h1"):
        h1.name = "h2"

    toc: list[dict] = []
    h_idx = 0
    for h in root.find_all(["h2", "h3"]):
        title_txt = clean_ws(h.get_text(" ", strip=True))
        if not title_txt:
            continue
        h_idx += 1
        hid = f"sec-{h_idx}"
        h["id"] = hid
        toc.append({
            "id": hid,
            "level": 2 if h.name == "h2" else 3,
            "title": title_txt[:80],
        })

    if footnotes:
        hr = soup.new_tag("hr")
        root.append(hr)
        ol = soup.new_tag("ol", attrs={"class": "references"})
        for n in sorted(footnotes.keys()):
            ref = footnotes[n]
            li = soup.new_tag("li", id=f"ref-{n}")
            back = soup.new_tag("a", href=f"#cite-{n}", attrs={"class": "ref-back", "aria-label": "返回正文"})
            back.string = "↩"
            li.append(back)
            li.append(" " + ref["text"] + " ")
            if ref["url"]:
                ext_a = soup.new_tag("a", href=ref["url"])
                if ref["tm_id"]:
                    ext_a["data-tm-id"] = ref["tm_id"]
                    ext_a["class"] = ["tm-internal"]
                    ext_a.string = "站内阅读 ↗"
                else:
                    ext_a["target"] = "_blank"
                    ext_a["rel"] = "noopener noreferrer"
                    ext_a.string = ref["url"][:68] + ("…" if len(ref["url"]) > 68 else "")
                li.append(ext_a)
            ol.append(li)
        root.append(ol)

    visible_text = clean_ws(root.get_text(" ", strip=True))
    out_html = "".join(str(child) for child in root.contents)
    return out_html, toc, len(visible_text)


def render_card_html(meta: dict, root_rel: str) -> str:
    """Render static .post-row HTML card usable in both index.html and posts/<id>.html."""
    author_disp = f"{meta['a']} · 回答" if meta["k"] == "ans" else meta["a"]
    post_href = f"{root_rel}/posts/{meta['id']}.html" if root_rel == "." else f"{meta['id']}.html"
    if post_href.startswith("./"):
        post_href = post_href[2:]
    q_base = "./" if root_rel == "." else "../"
    tags_html = ""
    if meta["kw"]:
        tags_links = "".join(
            f'<a href="{q_base}?q={quote(k)}">#{esc(k)}</a>' for k in meta["kw"][:4]
        )
        tags_html = f'<p class="tags">{tags_links}</p>'

    return (
        f'<article class="post-row" data-item-id="{esc(meta["id"])}">'
        f'<div class="post-meta">'
        f'<time datetime="{esc(meta["d"])}">{esc(meta["d"])}</time>'
        f'<a class="meta-author" href="{q_base}?q={quote(meta["a"])}">{esc(author_disp)}</a>'
        f'<span>▲ {meta["v"]} · {meta["m"]} 分钟</span>'
        f'</div>'
        f'<div class="post-summary">'
        f'<h2><a href="{esc(post_href)}"><span>{esc(meta["t"])}</span><span class="arr" aria-hidden="true">↗</span></a></h2>'
        f'<p>{esc(meta["s"])}</p>'
        f'{tags_html}'
        f'</div>'
        f'<i class="rule" aria-hidden="true"></i>'
        f'</article>'
    )


def render_post_page_html(payload: dict, bottom_metas: list[dict], seed_ids: list[str]) -> str:
    title_e = esc(payload["title"])
    desc_e = esc(truncate_hook(payload["summary"], 180))
    author_e = esc(payload["author"])
    date_e = esc(payload["date"])

    toc_list_html = "".join(
        f'<li class="level-{h["level"]}"><a href="#{esc(h["id"])}" data-toc-target="{esc(h["id"])}">{esc(h["title"])}</a></li>'
        for h in payload["toc"]
    )
    toc_mobile = (
        f'<details class="toc-mobile"><summary>目录</summary><nav aria-label="文章目录"><ol>{toc_list_html}</ol></nav></details>'
        if toc_list_html
        else ""
    )
    toc_desktop = (
        f'<nav class="toc" aria-label="目录"><p>目录</p><ol id="desktop-toc-list">{toc_list_html}</ol></nav>'
        if toc_list_html
        else ""
    )

    q_fold = ""
    if payload["kind"] == "answer" and payload.get("question_detail_html"):
        q_fold = (
            f'<details class="question-fold">'
            f'<summary>查看问题描述</summary>'
            f'<div class="question-fold-body prose">{payload["question_detail_html"]}</div>'
            f'</details>'
        )

    tags_html = ""
    if payload["keywords"]:
        t_links = "".join(
            f'<a href="../?q={quote(k)}">#{esc(k)}</a>' for k in payload["keywords"]
        )
        tags_html = f'<p class="tags">{t_links}</p>'

    restoration_html = ""
    if payload.get("restoration"):
        r = payload["restoration"]
        restoration_html = f'<aside class="restoration-banner"><strong>{esc(r["label"])}</strong>：{esc(r["notice"])}</aside>'

    orig_authors_links = " · ".join(
        f'<a href="../?q={quote(a)}">{esc(a)}</a>'
        for a in (payload["original_authors"] or [payload["author"]])
    )
    orig_sources_row = ""
    if payload["original_source_urls"]:
        s_links = "<br>".join(
            f'<a href="{esc(u)}" target="_blank" rel="noopener noreferrer">{esc(u)} ↗</a>'
            for u in payload["original_source_urls"]
        )
        orig_sources_row = f"<div><dt>原文</dt><dd>{s_links}</dd></div>"

    bottom_cards_html = "".join(render_card_html(m, "..") for m in bottom_metas)
    seeds_attr = esc(",".join(seed_ids))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_e}</title>
<meta name="description" content="{desc_e}">
<meta name="author" content="{author_e}">
<meta property="og:title" content="{title_e}">
<meta property="og:description" content="{desc_e}">
<meta property="og:type" content="article">
<meta property="article:published_time" content="{date_e}">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#f9fafc" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#13161b" media="(prefers-color-scheme: dark)">
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500&family=Noto+Serif+SC:wght@400;500;600&display=swap" rel="stylesheet">
<script>try{{const t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t}}catch{{}}</script>
<link rel="stylesheet" href="../style.css">
</head>
<body id="top" data-page="post" data-root=".." data-post-id="{esc(payload['id'])}" data-rec-seeds="{seeds_attr}">
<div class="progress" id="scroll-progress" aria-hidden="true"></div>
<div class="shell">
  <main id="main">
    <article class="post-article">
      <header class="article-header">
        <p class="article-meta">
          <a class="back-link" href="../">← 返回</a>
          <span class="sep" aria-hidden="true">/</span>
          <time datetime="{date_e}">{date_e}</time>
          <a class="tag" href="../?q={quote(payload['column'])}">{esc(payload['column'])}</a>
          <a class="tag" href="../?q={quote(payload['author'])}">{author_e}</a>
          <span>约 {payload['minutes']} 分钟</span>
          <span>▲ {payload['votes']}</span>
        </p>
        <h1>{title_e}</h1>
        {q_fold}
        {tags_html}
        <i class="rule" aria-hidden="true"></i>
      </header>
      {toc_mobile}
      <div class="article-body">
        <div class="prose" id="article-prose">
          {restoration_html}
          {payload['html']}
        </div>
        {toc_desktop}
        <footer class="colophon">
          <dl>
            <div><dt>作者</dt><dd>{orig_authors_links}</dd></div>
            {orig_sources_row}
            <div><dt>来源</dt><dd>{esc(payload['publisher'])} · <a href="{esc(payload['zhihu_url'])}" target="_blank" rel="noopener noreferrer">知乎原帖 ↗</a></dd></div>
          </dl>
        </footer>
      </div>
    </article>

    <section class="bottom-recs" aria-label="延伸阅读">
      <div class="bottom-recs-head">
        <i class="rule top" aria-hidden="true"></i>
        <h3 class="rec-heading">延伸阅读</h3>
        <span>结合本文内容与你的阅读偏好推荐</span>
        <i class="rule" aria-hidden="true"></i>
      </div>
      <div id="bottom-rec-list">
        {bottom_cards_html}
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <span>语料源自 <a href="https://github.com/L-M-Sherlock/tm-index" target="_blank" rel="noopener noreferrer">tm-index</a> &amp; <a href="https://github.com/L-M-Sherlock/ZhiHuArchive" target="_blank" rel="noopener noreferrer">ZhiHuArchive</a>（CC0 1.0 · 原著权利归原作者）</span>
    <nav aria-label="设置与链接">
      <button class="theme" type="button" id="theme-toggle" aria-label="切换深浅色主题" title="切换深浅色">
        <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
          <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <path d="M8 1.75a6.25 6.25 0 0 1 0 12.5z" fill="currentColor"/>
        </svg>
        <span>主题</span>
      </button>
      <a href="https://github.com/mizorewww/tm-feed" target="_blank" rel="noopener">GitHub</a>
    </nav>
  </footer>
</div>
<button class="to-top" id="to-top-btn" type="button" aria-label="回到顶部">↑</button>
<script src="../app.js" defer></script>
</body>
</html>
"""


def render_home_page_html(initial_metas: list[dict]) -> str:
    cards_html = "".join(render_card_html(m, ".") for m in initial_metas)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>译文漫游</title>
<meta name="description" content="Thoughts Memo 译文与问答个性化阅读流（收录 2,187 篇译文与 1,393 篇知乎回答）">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#f9fafc" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#13161b" media="(prefers-color-scheme: dark)">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500&family=Noto+Serif+SC:wght@400;500;600&display=swap" rel="stylesheet">
<script>try{{const t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t}}catch{{}}</script>
<link rel="stylesheet" href="style.css">
</head>
<body id="top" data-page="home" data-root=".">
<div class="progress" id="scroll-progress" aria-hidden="true"></div>
<div class="shell">
  <main id="main">
    <section class="writing">
      <div class="search-bar">
        <input id="search-input" type="search" placeholder="搜索标题、内容、作者或标签…" autocomplete="off" spellcheck="false">
        <span class="search-count" id="search-count" hidden></span>
        <i class="rule" aria-hidden="true"></i>
      </div>
      <div id="feed-list">
        {cards_html}
      </div>
      <div class="feed-more" id="feed-sentinel">向下滚动加载更多</div>
    </section>
  </main>

  <footer class="site-footer">
    <span>语料源自 <a href="https://github.com/L-M-Sherlock/tm-index" target="_blank" rel="noopener noreferrer">tm-index</a> &amp; <a href="https://github.com/L-M-Sherlock/ZhiHuArchive" target="_blank" rel="noopener noreferrer">ZhiHuArchive</a>（CC0 1.0 · 原著权利归原作者）</span>
    <nav aria-label="设置与链接">
      <button class="theme" type="button" id="theme-toggle" aria-label="切换深浅色主题" title="切换深浅色">
        <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
          <circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <path d="M8 1.75a6.25 6.25 0 0 1 0 12.5z" fill="currentColor"/>
        </svg>
        <span>主题</span>
      </button>
      <a href="https://github.com/mizorewww/tm-feed" target="_blank" rel="noopener">GitHub</a>
    </nav>
  </footer>
</div>
<button class="to-top" id="to-top-btn" type="button" aria-label="回到顶部">↑</button>
<script src="app.js" defer></script>
</body>
</html>
"""


def mmr_pick(scored_cands: list[tuple[float, int]], sim_mat: np.ndarray, items_meta: list[dict], k: int, lam: float = 0.72) -> list[int]:
    pool = scored_cands[: max(k * 8, 80)]
    chosen: list[int] = []
    used = [False] * len(pool)
    for _ in range(min(k, len(pool))):
        best_p = -1
        best_v = -1e9
        for p, (sc, c_idx) in enumerate(pool):
            if used[p]:
                continue
            max_pen = 0.0
            same_auth_cnt = 0
            for prev_idx in chosen[-4:]:
                v_sim = max(0.0, float(sim_mat[c_idx, prev_idx]))
                same_a = 0.55 if items_meta[c_idx]["a"] == items_meta[prev_idx]["a"] else 0.0
                if same_a > 0:
                    same_auth_cnt += 1
                pen = 0.5 * v_sim + same_a
                if pen > max_pen:
                    max_pen = pen
            if same_auth_cnt >= 2:
                max_pen += 0.8
            val = lam * sc - (1.0 - lam) * max_pen
            if val > best_v:
                best_v = val
                best_p = p
        if best_p == -1:
            break
        used[best_p] = True
        chosen.append(pool[best_p][1])
    return chosen


def main() -> None:
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove legacy .json files in public/posts/ if present
    for old_json in POSTS_DIR.glob("*.json"):
        old_json.unlink()

    print("[1/6] Loading tm-index metadata & ZhiHuArchive restorations...")
    cat_raw = json.loads((TM_INDEX_DIR / "catalog.json").read_text(encoding="utf-8"))
    ans_raw = json.loads((TM_INDEX_DIR / "answer-references.json").read_text(encoding="utf-8"))

    restorations_map: dict[str, dict] = {}
    rest_path = ARCHIVE_DIR / "restorations.json"
    if rest_path.exists():
        rdata = json.loads(rest_path.read_text(encoding="utf-8"))
        if isinstance(rdata, dict):
            if "items" in rdata and isinstance(rdata["items"], dict):
                restorations_map = rdata["items"]
            else:
                for k, v in rdata.items():
                    if isinstance(v, dict):
                        restorations_map[str(k)] = v

    included_articles = [a for a in cat_raw["articles"] if a.get("decision") != "exclude"]
    ans_dict_raw = ans_raw.get("answers", {})
    ans_dict = {str(x["id"]): x for x in ans_dict_raw} if isinstance(ans_dict_raw, list) else {str(k): v for k, v in ans_dict_raw.items()}

    ans_to_articles: dict[str, list[dict]] = collections.defaultdict(list)
    for art in included_articles:
        aid_self = str(art["id"])
        levels_map = art.get("answer_reference_levels") or {}
        for ans_id in art.get("cited_by_answer_ids") or []:
            ans_id_s = str(ans_id)
            lv_info = levels_map.get(ans_id_s) or {}
            lv = lv_info.get("level") if lv_info.get("level") in (1, 2, 3) else 2
            ans_to_articles[ans_id_s].append({
                "id": aid_self,
                "level": lv,
                "votes": art.get("voteup_count", 0),
            })

    valid_answer_ids = [aid for aid in ans_dict if (ARCHIVE_DIR / "answer" / f"{aid}.json").exists()]
    valid_article_ids = {str(a["id"]) for a in included_articles if (ARCHIVE_DIR / "article" / f"{a['id']}.json").exists()}
    all_valid_ids = valid_article_ids | set(valid_answer_ids)
    print(f"  -> Matched {len(valid_article_ids)} articles + {len(valid_answer_ids)} answers = {len(all_valid_ids)} items")

    included_articles = [a for a in included_articles if str(a["id"]) in valid_article_ids]
    included_articles.sort(key=lambda a: (a.get("voteup_count", 0), a.get("date", "")), reverse=True)
    valid_answer_ids.sort(key=lambda aid: (ans_dict[aid].get("voteup_count", 0), ans_dict[aid].get("date", "")), reverse=True)

    items_meta: list[dict] = []
    id_to_idx: dict[str, int] = {}
    full_payloads: dict[str, dict] = {}
    corpus_texts: list[str] = []

    print("[2/6] Parsing & sanitizing 3,580 full-text HTML documents...")
    for a in included_articles:
        aid = str(a["id"])
        arch_json = json.loads((ARCHIVE_DIR / "article" / f"{aid}.json").read_text(encoding="utf-8"))
        raw_html = arch_json.get("content", "")
        restored_html, rest_notice = apply_restoration(raw_html, restorations_map.get(aid))
        cleaned_html, toc, vis_chars = transform_html(restored_html, all_valid_ids)

        chars = int(a.get("read_chars") or vis_chars or 1000)
        minutes = max(1, round(chars / 480))
        orig_authors = a.get("original_authors") or []
        primary_author = orig_authors[0] if orig_authors else a.get("publisher", "Thoughts Memo")
        column = clean_ws(a.get("column") or "未分类")
        keywords = a.get("keywords") or []
        summary = clean_ws(a.get("summary") or "")
        votes = int(a.get("voteup_count") or arch_json.get("voteup_count") or 0)
        date_str = a.get("date") or ts_to_date(arch_json.get("created"))

        idx = len(items_meta)
        id_to_idx[aid] = idx
        items_meta.append({
            "id": aid,
            "k": "art",
            "t": clean_ws(a["title"]),
            "a": primary_author,
            "au": orig_authors,
            "pub": a.get("publisher", "Thoughts Memo"),
            "c": column,
            "d": date_str,
            "v": votes,
            "m": minutes,
            "chars": chars,
            "kw": keywords[:5],
            "s": truncate_hook(summary, 165),
            "in_deg": len(a.get("referenced_by") or []),
            "ans_deg": len(a.get("cited_by_answer_ids") or []),
        })

        full_payloads[aid] = {
            "id": aid,
            "kind": "article",
            "title": clean_ws(a["title"]),
            "author": primary_author,
            "original_authors": orig_authors,
            "original_source_urls": a.get("original_source_urls") or [],
            "publisher": a.get("publisher", "Thoughts Memo"),
            "column": column,
            "date": date_str,
            "votes": votes,
            "minutes": minutes,
            "keywords": keywords,
            "summary": summary,
            "zhihu_url": a.get("url") or f"https://zhuanlan.zhihu.com/p/{aid}",
            "restoration": rest_notice,
            "toc": toc,
            "html": cleaned_html,
            "_raw_refs": a.get("references") or [],
            "_raw_ref_by": a.get("referenced_by") or [],
        }

        corpus_texts.append(" ".join([
            (clean_ws(a["title"]) + " ") * 3,
            (column + " ") * 2,
            " ".join(orig_authors) * 2,
            (" ".join(keywords) + " ") * 3,
            summary,
        ]))

    for ans_id in valid_answer_ids:
        ans_info = ans_dict[ans_id]
        arch_json = json.loads((ARCHIVE_DIR / "answer" / f"{ans_id}.json").read_text(encoding="utf-8"))
        raw_html = arch_json.get("content", "")
        restored_html, rest_notice = apply_restoration(raw_html, restorations_map.get(ans_id))
        cleaned_html, toc, vis_chars = transform_html(restored_html, all_valid_ids)

        q_obj = arch_json.get("question") or {}
        q_title = clean_ws(ans_info.get("title") or q_obj.get("title") or f"知乎回答 {ans_id}")
        q_detail_raw = q_obj.get("detail") or ""
        q_detail_html, _, _ = transform_html(q_detail_raw, all_valid_ids) if q_detail_raw else ("", [], 0)

        plain_body = strip_html(raw_html)
        chars = max(len(plain_body), vis_chars, 120)
        minutes = max(1, round(chars / 480))
        publisher = clean_ws(ans_info.get("author") or (arch_json.get("author") or {}).get("name") or "Thoughts Memo")
        votes = int(ans_info.get("voteup_count") or arch_json.get("voteup_count") or 0)
        date_str = ans_info.get("date") or ts_to_date(arch_json.get("created_time"))

        cited_arts = sorted(
            [x for x in ans_to_articles.get(ans_id, []) if x["id"] in id_to_idx],
            key=lambda x: (x["level"], -x["votes"]),
        )
        kw_counter: collections.Counter[str] = collections.Counter()
        col_counter: collections.Counter[str] = collections.Counter()
        for ca in cited_arts:
            art_meta = items_meta[id_to_idx[ca["id"]]]
            w = 3 if ca["level"] == 1 else (2 if ca["level"] == 2 else 1)
            for kw in art_meta["kw"]:
                kw_counter[kw] += w
            if art_meta["c"] not in ("杂项", "未分类"):
                col_counter[art_meta["c"]] += w

        inferred_kws = [k for k, _ in kw_counter.most_common(5)]
        inferred_col = col_counter.most_common(1)[0][0] if col_counter else "问答实战"
        hook = truncate_hook(plain_body, 165)

        idx = len(items_meta)
        id_to_idx[ans_id] = idx
        items_meta.append({
            "id": ans_id,
            "k": "ans",
            "t": q_title,
            "a": publisher,
            "au": [publisher],
            "pub": publisher,
            "c": inferred_col,
            "d": date_str,
            "v": votes,
            "m": minutes,
            "chars": chars,
            "kw": inferred_kws,
            "s": hook,
            "in_deg": 0,
            "ans_deg": len(cited_arts),
        })

        qid = ans_info.get("question_id") or q_obj.get("id") or ""
        zhihu_url = ans_info.get("url") or (f"https://www.zhihu.com/question/{qid}/answer/{ans_id}" if qid else f"https://www.zhihu.com/answer/{ans_id}")

        full_payloads[ans_id] = {
            "id": ans_id,
            "kind": "answer",
            "title": q_title,
            "author": publisher,
            "original_authors": [publisher],
            "original_source_urls": [],
            "publisher": publisher,
            "column": inferred_col,
            "date": date_str,
            "votes": votes,
            "minutes": minutes,
            "keywords": inferred_kws,
            "summary": hook,
            "question_detail_html": q_detail_html,
            "zhihu_url": zhihu_url,
            "restoration": rest_notice,
            "toc": toc,
            "html": cleaned_html,
            "_cited_arts": cited_arts,
        }

        corpus_texts.append(" ".join([
            (q_title + " ") * 3,
            (inferred_col + " ") * 2,
            (" ".join(inferred_kws) + " ") * 3,
            plain_body[:1600],
        ]))

    n_items = len(items_meta)
    print(f"[3/6] Computing quality priors & co-citation graph for {n_items} items...")
    max_log_v = max(math.log1p(m["v"]) for m in items_meta) or 1.0
    max_log_in = max(math.log1p(m["in_deg"]) for m in items_meta) or 1.0
    max_log_ans = max(math.log1p(m["ans_deg"]) for m in items_meta) or 1.0
    for m in items_meta:
        qv = math.log1p(m["v"]) / max_log_v
        qi = math.log1p(m["in_deg"]) / max_log_in
        qa = math.log1p(m["ans_deg"]) / max_log_ans
        m["q"] = round(0.50 * qv + 0.30 * qi + 0.20 * qa if m["k"] == "art" else 0.72 * qv + 0.28 * qa, 4)

    edge_weights: dict[tuple[int, int], float] = collections.defaultdict(float)
    for aid in valid_article_ids:
        i = id_to_idx[aid]
        for ref in full_payloads[aid]["_raw_refs"]:
            tid = str(ref.get("id") if isinstance(ref, dict) else ref)
            if tid in id_to_idx and tid != aid:
                j = id_to_idx[tid]
                edge_weights[(i, j)] += 1.0
                edge_weights[(j, i)] += 0.85

    for ans_id in valid_answer_ids:
        a_idx = id_to_idx[ans_id]
        cited = full_payloads[ans_id]["_cited_arts"]
        deg = len(cited)
        if deg == 0:
            continue
        aa_denom = max(0.7, math.log(1.0 + deg))
        for ca in cited:
            i = id_to_idx[ca["id"]]
            lv_w = 1.0 if ca["level"] == 1 else (0.65 if ca["level"] == 2 else 0.3)
            edge_weights[(a_idx, i)] += 1.2 * lv_w
            edge_weights[(i, a_idx)] += 1.0 * lv_w
        capped = cited[:25]
        for u_pos in range(len(capped)):
            u_idx = id_to_idx[capped[u_pos]["id"]]
            wu = 1.0 if capped[u_pos]["level"] == 1 else (0.65 if capped[u_pos]["level"] == 2 else 0.3)
            for v_pos in range(u_pos + 1, len(capped)):
                v_idx = id_to_idx[capped[v_pos]["id"]]
                wv = 1.0 if capped[v_pos]["level"] == 1 else (0.65 if capped[v_pos]["level"] == 2 else 0.3)
                aa_score = (wu * wv) / aa_denom
                edge_weights[(u_idx, v_idx)] += 1.4 * aa_score
                edge_weights[(v_idx, u_idx)] += 1.4 * aa_score

    rows, cols, vals = [], [], []
    for (u, v), w in edge_weights.items():
        rows.append(u)
        cols.append(v)
        vals.append(w)
    for s_idx in range(n_items):
        rows.append(s_idx)
        cols.append(s_idx)
        vals.append(2.0)

    adj_norm = normalize(sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32), norm="l2", axis=1)

    print("[4/6] Computing 128-d Hybrid Semantic + Graph SVD Embeddings...")
    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_df=0.85, max_features=42000, sublinear_tf=True, dtype=np.float32).fit_transform(corpus_texts)
    tfidf_word = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2, max_df=0.85, max_features=16000, sublinear_tf=True, dtype=np.float32).fit_transform(corpus_texts)
    hybrid_features = sp.hstack([tfidf_char * 0.65, tfidf_word * 0.45, adj_norm * 0.40], format="csr")

    emb_f32 = TruncatedSVD(n_components=128, random_state=42).fit_transform(hybrid_features).astype(np.float32)
    emb_f32 = normalize(emb_f32, norm="l2", axis=1)
    emb_i8 = np.clip(np.round(emb_f32 * 127.0), -127, 127).astype(np.int8)
    (DATA_DIR / "embeddings.bin").write_bytes(emb_i8.tobytes())

    print("[5/6] Building neighbor graph & static bottom-6 recommendations...")
    sim_matrix = emb_f32 @ emb_f32.T
    per_item_adj: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    for (u, v), w in edge_weights.items():
        if u != v:
            per_item_adj[u].append((v, w))

    neighbors_compact: list[list[list[int]]] = []
    for i in range(n_items):
        cand_scores: dict[int, float] = {}
        for nb, w in per_item_adj.get(i, []):
            cand_scores[nb] = min(1.0, math.log1p(w) * 0.60) + float(sim_matrix[i, nb]) * 0.40
        top_sem = np.argpartition(-sim_matrix[i], 18)[:18]
        for nb in top_sem:
            nb_i = int(nb)
            if nb_i != i and nb_i not in cand_scores:
                cand_scores[nb_i] = float(sim_matrix[i, nb_i]) * 0.78
        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:14]
        neighbors_compact.append([[nb, int(round(sc * 100))] for nb, sc in ranked])

    (DATA_DIR / "neighbors.json").write_text(json.dumps(neighbors_compact, separators=(",", ":")), encoding="utf-8")

    print("[6/6] Rendering 3,580 standalone static HTML post pages + static index.html...")
    for idx, meta in enumerate(items_meta):
        iid = meta["id"]
        payload = full_payloads[iid]
        payload.pop("_raw_refs", None)
        payload.pop("_raw_ref_by", None)
        payload.pop("_cited_arts", None)

        # Static fallback bottom 6 cards (content + graph + prior quality + MMR diversity)
        direct_lookup = {nb: w_int / 100.0 for nb, w_int in neighbors_compact[idx]}
        top_vec_idx = np.argpartition(-sim_matrix[idx], 80)[:80]
        cand_pool_idx = set(int(x) for x in top_vec_idx) | set(direct_lookup.keys())
        cand_pool_idx.discard(idx)

        scored_cands: list[tuple[float, int]] = []
        for c_idx in cand_pool_idx:
            c_score = 0.55 * float(sim_matrix[idx, c_idx]) + 0.30 * direct_lookup.get(c_idx, 0.0) + 0.15 * items_meta[c_idx]["q"]
            scored_cands.append((c_score, c_idx))
        scored_cands.sort(key=lambda x: x[0], reverse=True)

        chosen_indices = mmr_pick(scored_cands, sim_matrix, items_meta, BOTTOM_REC_LIMIT, lam=0.70)
        bottom_metas = [items_meta[c_i] for c_i in chosen_indices]
        seed_ids = [items_meta[nb]["id"] for nb, _ in neighbors_compact[idx][:8]]

        page_html = render_post_page_html(payload, bottom_metas, seed_ids)
        (POSTS_DIR / f"{iid}.html").write_text(page_html, encoding="utf-8")

    # Render static index.html with Top 20 MMR-diversified quality items for curl / no-JS
    home_cands = sorted(
        [(0.80 * m["q"] + 0.20 * min(1.0, (m["in_deg"] + m["ans_deg"]) / 30.0), idx) for idx, m in enumerate(items_meta)],
        key=lambda x: x[0],
        reverse=True,
    )
    home_initial_indices = mmr_pick(home_cands, sim_matrix, items_meta, 20, lam=0.74)
    (OUT_PUBLIC / "index.html").write_text(
        render_home_page_html([items_meta[i] for i in home_initial_indices]),
        encoding="utf-8",
    )

    compact_items = [
        {
            "id": m["id"],
            "k": m["k"],
            "t": m["t"],
            "a": m["a"],
            "c": m["c"],
            "d": m["d"],
            "v": m["v"],
            "m": m["m"],
            "q": m["q"],
            "kw": m["kw"][:4],
            "s": m["s"],
            "deg": m["in_deg"] + m["ans_deg"],
        }
        for m in items_meta
    ]
    (DATA_DIR / "feed-index.json").write_text(
        json.dumps({"items": compact_items}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print("Done! All 3,580 static HTML pages + recommender assets generated.")


if __name__ == "__main__":
    main()
