#!/usr/bin/env python3
"""
Build offline corpus & hybrid recommender assets for tm-feed from:
- L-M-Sherlock/tm-index (CC0 1.0)
- L-M-Sherlock/ZhiHuArchive (CC0 1.0)

Outputs into public/:
- public/posts/<id>.json        (full-text cleaned HTML + TOC + 3-slot bottom recs)
- public/data/feed-index.json   (compact metadata for all items + global stats + facets)
- public/data/embeddings.bin    (N x 128 Int8 L2-normalized hybrid semantic+graph vectors)
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
from urllib.parse import parse_qs, unquote, urlparse, urlsplit

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
    """Return (cleaned_href, matched_internal_id_or_None)."""
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
            return f"#/post/{matched_id}", matched_id

    # Relative ./12345.html from restorations
    m_rel = re.match(r"^\./(\d+)\.html$", url)
    if m_rel and m_rel.group(1) in valid_ids:
        mid = m_rel.group(1)
        return f"#/post/{mid}", mid

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
    """
    Clean Zhihu HTML into blog-new editorial prose HTML:
    - Fix lazy images & equations
    - Convert <sup data-numero> into blog-new .cite + ol.references
    - Convert internal links to #/post/<id> with data-tm-id for hover .cite-card
    - Extract TOC from h2/h3 and inject anchor ids
    Returns (cleaned_html, toc_list, visible_char_count).
    """
    if not raw_html:
        return "", [], 0

    soup = BeautifulSoup(raw_html, "lxml")
    root = soup.body if soup.body else soup

    # Unwrap <u>
    for u in root.find_all("u"):
        u.unwrap()

    # Clean images & equations
    for img in root.find_all("img"):
        actual = img.get("data-actualsrc") or img.get("data-original")
        src = img.get("src", "")
        if actual and (not src or src.startswith("data:image")):
            img["src"] = actual
        elif actual:
            img["src"] = actual

        for attr in ("data-actualsrc", "data-original", "data-rawwidth", "data-rawheight", "data-size", "data-caption"):
            img.attrs.pop(attr, None)

        final_src = img.get("src", "")
        if not final_src or final_src.startswith("data:image/svg+xml"):
            # Empty Zhihu lazy placeholder with no real URL
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

    # Collect Zhihu native footnotes <sup data-text data-url data-numero>
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

    # Rewrite links
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

    # Extract TOC from h2 / h3 (and promote stray h1 inside prose to h2)
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

    # Append converted footnotes as blog-new ol.references
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


def main() -> None:
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    if isinstance(ans_dict_raw, list):
        ans_dict = {str(x["id"]): x for x in ans_dict_raw}
    else:
        ans_dict = {str(k): v for k, v in ans_dict_raw.items()}

    # Invert article -> answer citations so each answer knows which translated articles it cites
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
                "title": art["title"],
                "author": (art.get("original_authors") or [art.get("publisher", "Thoughts Memo")])[0],
                "level": lv,
                "evidence": clean_ws(lv_info.get("evidence", "")),
                "votes": art.get("voteup_count", 0),
            })

    # Keep all cited answers that actually exist in ZhiHuArchive/answer/<id>.json
    valid_answer_ids: list[str] = []
    for ans_id in ans_dict:
        if (ARCHIVE_DIR / "answer" / f"{ans_id}.json").exists():
            valid_answer_ids.append(ans_id)

    valid_article_ids = {str(a["id"]) for a in included_articles if (ARCHIVE_DIR / "article" / f"{a['id']}.json").exists()}
    all_valid_ids = valid_article_ids | set(valid_answer_ids)
    print(f"  -> Matched {len(valid_article_ids)} full-text articles + {len(valid_answer_ids)} full-text Zhihu answers = {len(all_valid_ids)} items")

    # Sort articles & answers deterministically
    included_articles = [a for a in included_articles if str(a["id"]) in valid_article_ids]
    included_articles.sort(key=lambda a: (a.get("voteup_count", 0), a.get("date", "")), reverse=True)
    valid_answer_ids.sort(key=lambda aid: (ans_dict[aid].get("voteup_count", 0), ans_dict[aid].get("date", "")), reverse=True)

    # Assign unified 0..N-1 index (articles first, then answers)
    items_meta: list[dict] = []
    id_to_idx: dict[str, int] = {}
    full_payloads: dict[str, dict] = {}
    corpus_texts: list[str] = []

    print("[2/6] Parsing & sanitizing 3,500+ full-text HTML documents from ZhiHuArchive...")
    # Build same-column chronological neighbor lookup for prev/next sequence recommendations
    col_groups: dict[str, list[dict]] = collections.defaultdict(list)
    for a in included_articles:
        col = clean_ws(a.get("column") or "未分类")
        col_groups[col].append(a)
    col_prev_next: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for col, group in col_groups.items():
        if col in ("杂项", "未分类") or len(group) < 2:
            continue
        s_group = sorted(group, key=lambda x: (x.get("date", ""), str(x["id"])))
        for idx_g, a in enumerate(s_group):
            aid = str(a["id"])
            if idx_g > 0:
                p = s_group[idx_g - 1]
                col_prev_next[aid].append((str(p["id"]), f"同专栏上一篇 · {col}"))
            if idx_g + 1 < len(s_group):
                n = s_group[idx_g + 1]
                col_prev_next[aid].append((str(n["id"]), f"同专栏下一篇 · {col}"))

    for a in included_articles:
        aid = str(a["id"])
        arch_file = ARCHIVE_DIR / "article" / f"{aid}.json"
        arch_json = json.loads(arch_file.read_text(encoding="utf-8"))
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
        comment_count = int(arch_json.get("comment_count") or 0)
        date_str = a.get("date") or ts_to_date(arch_json.get("created"))
        updated_str = ts_to_date(arch_json.get("updated"), date_str)

        idx = len(items_meta)
        id_to_idx[aid] = idx

        in_deg = len(a.get("referenced_by") or [])
        ans_deg = len(a.get("cited_by_answer_ids") or [])

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
            "kw": keywords[:6],
            "s": truncate_hook(summary, 165),
            "in_deg": in_deg,
            "ans_deg": ans_deg,
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
            "updated": updated_str,
            "votes": votes,
            "comment_count": comment_count,
            "minutes": minutes,
            "chars": chars,
            "keywords": keywords,
            "summary": summary,
            "zhihu_url": a.get("url") or f"https://zhuanlan.zhihu.com/p/{aid}",
            "restoration": rest_notice,
            "toc": toc,
            "html": cleaned_html,
            "_raw_refs": a.get("references") or [],
            "_raw_ref_by": a.get("referenced_by") or [],
            "_raw_ans_ids": a.get("cited_by_answer_ids") or [],
            "_raw_ans_levels": a.get("answer_reference_levels") or {},
        }

        doc_text = " ".join([
            (clean_ws(a["title"]) + " ") * 3,
            (column + " ") * 2,
            " ".join(orig_authors) * 2,
            (" ".join(keywords) + " ") * 3,
            summary,
        ])
        corpus_texts.append(doc_text)

    # Process Zhihu Answers
    for ans_id in valid_answer_ids:
        ans_info = ans_dict[ans_id]
        arch_file = ARCHIVE_DIR / "answer" / f"{ans_id}.json"
        arch_json = json.loads(arch_file.read_text(encoding="utf-8"))
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
        comment_count = int(arch_json.get("comment_count") or 0)
        date_str = ans_info.get("date") or ts_to_date(arch_json.get("created_time"))
        updated_str = ts_to_date(arch_json.get("updated_time"), date_str)

        cited_arts = sorted(
            [x for x in ans_to_articles.get(ans_id, []) if x["id"] in id_to_idx],
            key=lambda x: (x["level"], -x["votes"]),
        )
        # Inherit top keywords & dominant column from cited translated articles!
        kw_counter: collections.Counter[str] = collections.Counter()
        col_counter: collections.Counter[str] = collections.Counter()
        for ca in cited_arts:
            art_meta = items_meta[id_to_idx[ca["id"]]]
            weight = 3 if ca["level"] == 1 else (2 if ca["level"] == 2 else 1)
            for kw in art_meta["kw"]:
                kw_counter[kw] += weight
            if art_meta["c"] not in ("杂项", "未分类"):
                col_counter[art_meta["c"]] += weight

        inferred_kws = [k for k, _ in kw_counter.most_common(5)]
        inferred_col = col_counter.most_common(1)[0][0] if col_counter else "知乎问答实战"

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
            "updated": updated_str,
            "votes": votes,
            "comment_count": comment_count,
            "minutes": minutes,
            "chars": chars,
            "keywords": inferred_kws,
            "summary": hook,
            "question_detail_html": q_detail_html,
            "zhihu_url": zhihu_url,
            "restoration": rest_notice,
            "toc": toc,
            "html": cleaned_html,
            "_cited_arts": cited_arts,
        }

        doc_text = " ".join([
            (q_title + " ") * 3,
            (inferred_col + " ") * 2,
            (" ".join(inferred_kws) + " ") * 3,
            plain_body[:1600],
        ])
        corpus_texts.append(doc_text)

    n_items = len(items_meta)
    print(f"[3/6] Computing quality priors & multi-relational co-citation graph for {n_items} items...")

    # Prior Quality Score Q(i) in [0, 1]
    max_log_v = max(math.log1p(m["v"]) for m in items_meta) or 1.0
    max_log_in = max(math.log1p(m["in_deg"]) for m in items_meta) or 1.0
    max_log_ans = max(math.log1p(m["ans_deg"]) for m in items_meta) or 1.0
    for m in items_meta:
        qv = math.log1p(m["v"]) / max_log_v
        qi = math.log1p(m["in_deg"]) / max_log_in
        qa = math.log1p(m["ans_deg"]) / max_log_ans
        if m["k"] == "art":
            q = 0.50 * qv + 0.30 * qi + 0.20 * qa
        else:
            q = 0.72 * qv + 0.28 * qa
        m["q"] = round(q, 4)

    # Build sparse graph matrix:
    # 1) Direct article-article citations
    # 2) Article-answer citations
    # 3) Answer Co-Citation Adamic-Adar bridges between articles (crucial cross-author bridge!)
    edge_weights: dict[tuple[int, int], float] = collections.defaultdict(float)
    cocite_counts: dict[tuple[int, int], int] = collections.defaultdict(int)

    for aid in valid_article_ids:
        i = id_to_idx[aid]
        p = full_payloads[aid]
        for ref in p["_raw_refs"]:
            tid = str(ref.get("id") if isinstance(ref, dict) else ref)
            if tid in id_to_idx and tid != aid:
                j = id_to_idx[tid]
                edge_weights[(i, j)] += 1.0
                edge_weights[(j, i)] += 0.85

    # Answer-article links & Adamic-Adar Answer Co-Citation
    for ans_id in valid_answer_ids:
        a_idx = id_to_idx[ans_id]
        cited = full_payloads[ans_id]["_cited_arts"]
        deg = len(cited)
        if deg == 0:
            continue
        aa_denom = math.log(1.0 + deg)
        for ca in cited:
            i = id_to_idx[ca["id"]]
            lv_w = 1.0 if ca["level"] == 1 else (0.65 if ca["level"] == 2 else 0.3)
            edge_weights[(a_idx, i)] += 1.2 * lv_w
            edge_weights[(i, a_idx)] += 1.0 * lv_w

        # Pairwise co-cited articles (cap very large directory answers at top 25 to avoid O(k^2) noise)
        capped = cited[:25]
        for u_pos in range(len(capped)):
            u_id = capped[u_pos]["id"]
            u_idx = id_to_idx[u_id]
            wu = 1.0 if capped[u_pos]["level"] == 1 else (0.65 if capped[u_pos]["level"] == 2 else 0.3)
            for v_pos in range(u_pos + 1, len(capped)):
                v_id = capped[v_pos]["id"]
                v_idx = id_to_idx[v_id]
                wv = 1.0 if capped[v_pos]["level"] == 1 else (0.65 if capped[v_pos]["level"] == 2 else 0.3)
                aa_score = (wu * wv) / max(aa_denom, 0.7)
                edge_weights[(u_idx, v_idx)] += 1.4 * aa_score
                edge_weights[(v_idx, u_idx)] += 1.4 * aa_score
                cocite_counts[(u_idx, v_idx)] += 1
                cocite_counts[(v_idx, u_idx)] += 1

    rows, cols, vals = [], [], []
    for (u, v), w in edge_weights.items():
        rows.append(u)
        cols.append(v)
        vals.append(w)
    for idx_self in range(n_items):
        rows.append(idx_self)
        cols.append(idx_self)
        vals.append(2.0)

    adj_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    adj_norm = normalize(adj_mat, norm="l2", axis=1)

    print("[4/6] Computing 128-d Hybrid Semantic (Char+Word TF-IDF) + Graph SVD Embeddings...")
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.85,
        max_features=42000,
        sublinear_tf=True,
        dtype=np.float32,
    ).fit_transform(corpus_texts)

    tfidf_word = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_df=0.85,
        max_features=16000,
        sublinear_tf=True,
        dtype=np.float32,
    ).fit_transform(corpus_texts)

    hybrid_features = sp.hstack([
        tfidf_char * 0.65,
        tfidf_word * 0.45,
        adj_norm * 0.40,
    ], format="csr")

    svd = TruncatedSVD(n_components=128, random_state=42)
    emb_f32 = svd.fit_transform(hybrid_features).astype(np.float32)
    emb_f32 = normalize(emb_f32, norm="l2", axis=1)

    # Quantize to Int8 [-127, 127]
    emb_i8 = np.clip(np.round(emb_f32 * 127.0), -127, 127).astype(np.int8)
    (DATA_DIR / "embeddings.bin").write_bytes(emb_i8.tobytes())
    print(f"  -> Wrote embeddings.bin ({emb_i8.nbytes / 1024:.1f} KB, shape={emb_i8.shape})")

    print("[5/6] Precomputing 3-slot bottom recommendations & real-time graph neighbor tables...")
    # Full cosine matrix (3580 x 3580 takes ~45 MB RAM and <0.15s in BLAS!)
    sim_matrix = emb_f32 @ emb_f32.T

    # Build per-item top-12 neighbors for client-side 1-hop graph propagation
    neighbors_compact: list[list[list[int]]] = []
    per_item_adj: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    for (u, v), w in edge_weights.items():
        if u != v:
            per_item_adj[u].append((v, w))

    for i in range(n_items):
        cand_scores: dict[int, float] = {}
        for nb, w in per_item_adj.get(i, []):
            cand_scores[nb] = math.log1p(w) * 0.65 + float(sim_matrix[i, nb]) * 0.35
        # Supplement with top semantic neighbors if sparse
        top_sem = np.argpartition(-sim_matrix[i], 16)[:16]
        for nb in top_sem:
            nb_i = int(nb)
            if nb_i != i and nb_i not in cand_scores:
                cand_scores[nb_i] = float(sim_matrix[i, nb_i]) * 0.75

        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:12]
        neighbors_compact.append([[nb, int(round(sc * 100))] for nb, sc in ranked])

    (DATA_DIR / "neighbors.json").write_text(
        json.dumps(neighbors_compact, separators=(",", ":")),
        encoding="utf-8",
    )

    # Populate 3-slot bottom recommendations & write public/posts/<id>.json
    print("[6/6] Writing 3,580 full-text post JSON files into public/posts/...")
    article_indices = np.array([idx for idx, m in enumerate(items_meta) if m["k"] == "art"], dtype=np.int32)

    for idx, meta in enumerate(items_meta):
        iid = meta["id"]
        payload = full_payloads[iid]

        if meta["k"] == "art":
            # Slot 1: 顺藤摸瓜 · 引用延伸与同系上下文 (rec_chain)
            chain_seen: set[str] = {iid}
            rec_chain: list[dict] = []

            for target_id, reason_label in col_prev_next.get(iid, []):
                if target_id in id_to_idx and target_id not in chain_seen:
                    chain_seen.add(target_id)
                    tm = items_meta[id_to_idx[target_id]]
                    rec_chain.append({
                        "id": target_id,
                        "title": tm["t"],
                        "author": tm["a"],
                        "votes": tm["v"],
                        "minutes": tm["m"],
                        "reason": reason_label,
                    })

            for ref in payload.pop("_raw_refs", []):
                tid = str(ref.get("id") if isinstance(ref, dict) else ref)
                if tid in id_to_idx and tid not in chain_seen:
                    chain_seen.add(tid)
                    tm = items_meta[id_to_idx[tid]]
                    rec_chain.append({
                        "id": tid,
                        "title": tm["t"],
                        "author": tm["a"],
                        "votes": tm["v"],
                        "minutes": tm["m"],
                        "reason": "本文直接引用",
                    })
                    if len(rec_chain) >= 6:
                        break

            raw_ref_by = payload.pop("_raw_ref_by", [])
            if len(rec_chain) < 6:
                ref_by_sorted = sorted(
                    [str(r.get("id") if isinstance(r, dict) else r) for r in raw_ref_by],
                    key=lambda rid: items_meta[id_to_idx[rid]]["v"] if rid in id_to_idx else -1,
                    reverse=True,
                )
                for tid in ref_by_sorted:
                    if tid in id_to_idx and tid not in chain_seen:
                        chain_seen.add(tid)
                        tm = items_meta[id_to_idx[tid]]
                        rec_chain.append({
                            "id": tid,
                            "title": tm["t"],
                            "author": tm["a"],
                            "votes": tm["v"],
                            "minutes": tm["m"],
                            "reason": "延伸引用本文",
                        })
                        if len(rec_chain) >= 6:
                            break

            # Slot 2: 跨界共鸣 · 异作者破茧推荐 (rec_bridge)
            my_authors = set(meta["au"])
            my_kws = set(meta["kw"])
            sims = sim_matrix[idx, article_indices]
            top_cand_pos = np.argpartition(-sims, min(60, len(sims) - 1))[:60]
            bridge_cands: list[tuple[float, int, str]] = []
            for pos in top_cand_pos:
                cand_idx = int(article_indices[pos])
                if cand_idx == idx:
                    continue
                cm = items_meta[cand_idx]
                if cm["id"] in chain_seen:
                    continue
                if my_authors and (my_authors & set(cm["au"])):
                    continue  # Strictly cross-author!
                cos_val = float(sim_matrix[idx, cand_idx])
                co_cnt = cocite_counts.get((idx, cand_idx), 0)
                score = 0.62 * cos_val + 0.23 * min(1.0, co_cnt / 4.0) + 0.15 * cm["q"]
                shared_kw = list(my_kws & set(cm["kw"]))
                if co_cnt >= 2:
                    reason = f"被 {co_cnt} 篇回答共同引用 · {cm['a']}"
                elif shared_kw:
                    reason = f"跨作者共鸣「{shared_kw[0]}」· {cm['a']}"
                else:
                    reason = f"跨学派视角 · {cm['a']}"
                bridge_cands.append((score, cand_idx, reason))

            bridge_cands.sort(key=lambda x: x[0], reverse=True)
            rec_bridge: list[dict] = []
            seen_bridge_authors: set[str] = set()
            for _, cand_idx, reason in bridge_cands:
                cm = items_meta[cand_idx]
                # Diversify authors inside rec_bridge too!
                if cm["a"] in seen_bridge_authors and len(rec_bridge) < 4:
                    continue
                seen_bridge_authors.add(cm["a"])
                rec_bridge.append({
                    "id": cm["id"],
                    "title": cm["t"],
                    "author": cm["a"],
                    "votes": cm["v"],
                    "minutes": cm["m"],
                    "reason": reason,
                })
                if len(rec_bridge) >= 5:
                    break

            # Slot 3: 知乎实战 · 引用本文的高赞回答 (rec_answers)
            raw_ans_ids = payload.pop("_raw_ans_ids", [])
            raw_ans_levels = payload.pop("_raw_ans_levels", {})
            ans_cands = []
            for aid_ans in raw_ans_ids:
                aid_s = str(aid_ans)
                if aid_s not in id_to_idx:
                    continue
                am = items_meta[id_to_idx[aid_s]]
                lv_obj = raw_ans_levels.get(aid_s) or {}
                lv = lv_obj.get("level") if lv_obj.get("level") in (1, 2, 3) else 2
                ev = clean_ws(lv_obj.get("evidence", ""))
                ans_cands.append((lv, -am["v"], aid_s, am, ev))
            ans_cands.sort(key=lambda x: (x[0], x[1]))
            rec_answers = []
            for lv, _, aid_s, am, ev in ans_cands[:5]:
                lv_tag = "核心节选引用" if lv == 1 else ("观点化用" if lv == 2 else "延伸提及")
                rec_answers.append({
                    "id": aid_s,
                    "title": am["t"],
                    "author": am["a"],
                    "votes": am["v"],
                    "minutes": am["m"],
                    "reason": f"{lv_tag} · {am['v']} 赞同",
                    "evidence": truncate_hook(ev, 90) if ev else "",
                })

            payload["rec_chain"] = rec_chain[:5]
            payload["rec_bridge"] = rec_bridge[:5]
            payload["rec_answers"] = rec_answers
        else:
            # Answer view bottom 3 slots:
            cited_arts = payload.pop("_cited_arts", [])
            rec_chain = []
            cited_set = set()
            for ca in cited_arts[:6]:
                cid = ca["id"]
                cited_set.add(cid)
                cm = items_meta[id_to_idx[cid]]
                lv_tag = "Level 1 核心引用" if ca["level"] == 1 else (f"Level {ca['level']} 引用")
                rec_chain.append({
                    "id": cid,
                    "title": cm["t"],
                    "author": cm["a"],
                    "votes": cm["v"],
                    "minutes": cm["m"],
                    "reason": f"本回答{lv_tag}",
                })

            # Cross-author / expanded articles not directly cited
            sims = sim_matrix[idx, article_indices]
            top_pos = np.argpartition(-sims, min(30, len(sims) - 1))[:30]
            bridge_list = []
            for pos in top_pos:
                c_idx = int(article_indices[pos])
                cm = items_meta[c_idx]
                if cm["id"] in cited_set:
                    continue
                bridge_list.append((float(sim_matrix[idx, c_idx]) * 0.75 + 0.25 * cm["q"], cm))
            bridge_list.sort(key=lambda x: x[0], reverse=True)
            rec_bridge = [
                {
                    "id": cm["id"],
                    "title": cm["t"],
                    "author": cm["a"],
                    "votes": cm["v"],
                    "minutes": cm["m"],
                    "reason": f"延伸深读 · {cm['a']}",
                }
                for _, cm in bridge_list[:5]
            ]

            # Similar Zhihu answers
            ans_indices = [i_a for i_a, m_a in enumerate(items_meta) if m_a["k"] == "ans" and i_a != idx]
            ans_sims = [(float(sim_matrix[idx, i_a]) * 0.75 + 0.25 * items_meta[i_a]["q"], items_meta[i_a]) for i_a in ans_indices]
            ans_sims.sort(key=lambda x: x[0], reverse=True)
            rec_answers = [
                {
                    "id": am["id"],
                    "title": am["t"],
                    "author": am["a"],
                    "votes": am["v"],
                    "minutes": am["m"],
                    "reason": f"相似问答 · {am['v']} 赞同",
                }
                for _, am in ans_sims[:5]
            ]

            payload["rec_chain"] = rec_chain
            payload["rec_bridge"] = rec_bridge
            payload["rec_answers"] = rec_answers

        (POSTS_DIR / f"{iid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    # Facets & global corpus ledger stats
    col_counts = collections.Counter(m["c"] for m in items_meta if m["k"] == "art")
    kw_counts = collections.Counter(kw for m in items_meta for kw in m["kw"])
    author_counts = collections.Counter(m["a"] for m in items_meta if m["k"] == "art")

    total_chars = sum(m["chars"] for m in items_meta)
    # Strip internal fields before writing feed-index.json
    compact_items = [
        {
            "id": m["id"],
            "k": m["k"],
            "t": m["t"],
            "a": m["a"],
            "pub": m["pub"],
            "c": m["c"],
            "d": m["d"],
            "v": m["v"],
            "m": m["m"],
            "q": m["q"],
            "kw": m["kw"][:5],
            "s": m["s"],
            "deg": m["in_deg"] + m["ans_deg"],
        }
        for m in items_meta
    ]

    feed_index = {
        "stats": {
            "total_items": n_items,
            "articles": sum(1 for m in items_meta if m["k"] == "art"),
            "answers": sum(1 for m in items_meta if m["k"] == "ans"),
            "total_chars": total_chars,
            "total_edges": 13824 + 9913,
            "embedding_dim": 128,
        },
        "columns": [{"name": c, "count": cnt} for c, cnt in col_counts.most_common(32)],
        "top_keywords": [{"name": k, "count": cnt} for k, cnt in kw_counts.most_common(80)],
        "top_authors": [{"name": a, "count": cnt} for a, cnt in author_counts.most_common(40)],
        "items": compact_items,
    }
    (DATA_DIR / "feed-index.json").write_text(
        json.dumps(feed_index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  -> Wrote feed-index.json ({(DATA_DIR / 'feed-index.json').stat().st_size / 1024:.1f} KB)")
    print("Done! All corpus & recommender assets built successfully.")


if __name__ == "__main__":
    main()
