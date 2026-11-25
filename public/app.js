/**
 * TM-Feed (Minimal Edition)
 * - 零显式反馈按钮：点进去阅读即代表感兴趣（结合阅读停留与滚动深度自动加权）
 * - 主页：顶部仅保留一个搜索框；信息流 100% 根据个人阅读历史推荐（无历史时按高质量先验冷启动）
 * - 文末：同款简洁卡片列表，固定有限数量（6 篇），权衡「当前文章内容相关性」与「个人历史偏好」
 */

(() => {
  const EMB_DIM = 128;
  const STORAGE_KEY = "tm_feed_history_v2";
  const BOTTOM_REC_LIMIT = 6;

  let FEED_DATA = null;
  let ITEMS = [];
  let ID_TO_IDX = new Map();
  let EMB_I8 = null;
  let NEIGHBORS = null;
  const POST_CACHE = new Map();

  let currentFeedPool = [];
  let renderedCount = 0;
  let searchQuery = "";
  let feedObserver = null;
  let activeSession = null; // { id, startTime, maxScroll, settled }
  let hoverCard = null;

  // Minimal implicit history state: { history: [{ id, weight, ts }], readMap: { id: { count, dwell, ts } } }
  let state = loadState();

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { history: [], readMap: {} };
      const parsed = JSON.parse(raw);
      return {
        history: Array.isArray(parsed.history) ? parsed.history : [],
        readMap: parsed.readMap && typeof parsed.readMap === "object" ? parsed.readMap : {},
      };
    } catch {
      return { history: [], readMap: {} };
    }
  }

  function saveState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {}
  }

  // Clicking into an article immediately counts as interest; reading longer boosts weight
  function recordClickOpen(id) {
    const sid = String(id);
    if (!ID_TO_IDX.has(sid)) return;
    const now = Date.now();
    const prevRead = state.readMap[sid] || { count: 0, dwell: 0, ts: now };
    state.readMap[sid] = { count: prevRead.count + 1, dwell: prevRead.dwell, ts: now };

    const existing = state.history.find(h => h.id === sid);
    const baseWeight = existing ? Math.min(5.0, existing.weight + 0.8) : 1.6;
    state.history = [
      { id: sid, weight: baseWeight, ts: now },
      ...state.history.filter(h => h.id !== sid),
    ].slice(0, 60);
    saveState();
  }

  function flushActiveReading() {
    if (!activeSession || activeSession.settled) return;
    activeSession.settled = true;
    const sec = (Date.now() - activeSession.startTime) / 1000;
    const sid = activeSession.id;

    if (state.readMap[sid]) {
      state.readMap[sid].dwell = Math.round((state.readMap[sid].dwell || 0) + sec);
    }

    const histEntry = state.history.find(h => h.id === sid);
    if (histEntry) {
      if (sec >= 20 || activeSession.maxScroll >= 0.5) {
        const bonus = activeSession.maxScroll >= 0.8 ? 1.8 : 1.0;
        histEntry.weight = Number(Math.min(6.0, histEntry.weight + bonus).toFixed(2));
      } else if (sec < 3 && activeSession.maxScroll < 0.08 && state.readMap[sid]?.count === 1) {
        // Instant accidental click-and-back: soften weight so it doesn't skew history
        histEntry.weight = 0.35;
      }
    }
    saveState();
    activeSession = null;
  }

  // Vector operations
  function dotItemWithVec(itemIdx, vec) {
    if (!EMB_I8 || !vec) return 0;
    const base = itemIdx * EMB_DIM;
    let s = 0;
    for (let d = 0; d < EMB_DIM; d++) s += (EMB_I8[base + d] / 127.0) * vec[d];
    return s;
  }

  function dotItems(idxA, idxB) {
    if (!EMB_I8) return 0;
    const bA = idxA * EMB_DIM;
    const bB = idxB * EMB_DIM;
    let s = 0;
    for (let d = 0; d < EMB_DIM; d++) s += EMB_I8[bA + d] * EMB_I8[bB + d];
    return s / (127.0 * 127.0);
  }

  function buildHistoryProfile() {
    const recentEntries = [];
    const longEntries = [];
    const tagAff = new Map();
    const authorAff = new Map();
    const graphNeighbors = new Map();

    state.history.forEach((h, pos) => {
      const idx = ID_TO_IDX.get(h.id);
      if (idx === undefined) return;
      const it = ITEMS[idx];
      const w = h.weight || 1.0;

      if (pos < 8) {
        recentEntries.push({ idx, w: w * Math.pow(0.78, pos) });
      }
      longEntries.push({ idx, w: w * Math.pow(0.94, pos) });

      const decay = Math.pow(0.88, pos);
      for (const kw of it.kw || []) {
        tagAff.set(kw, (tagAff.get(kw) || 0) + w * decay * 0.45);
      }
      if (it.a) {
        authorAff.set(it.a, (authorAff.get(it.a) || 0) + w * decay * 0.25);
      }

      if (NEIGHBORS && pos < 6) {
        for (const [nbIdx, wInt] of NEIGHBORS[idx] || []) {
          const gVal = (wInt / 100.0) * Math.min(1.4, w / 2.0) * Math.pow(0.82, pos);
          if (gVal > (graphNeighbors.get(nbIdx) || 0)) {
            graphNeighbors.set(nbIdx, gVal);
          }
        }
      }
    });

    const toNormVec = (arr) => {
      if (!EMB_I8 || !arr.length) return null;
      const v = new Float32Array(EMB_DIM);
      for (const { idx, w } of arr) {
        const base = idx * EMB_DIM;
        for (let d = 0; d < EMB_DIM; d++) v[d] += (EMB_I8[base + d] / 127.0) * w;
      }
      let norm = 0;
      for (let d = 0; d < EMB_DIM; d++) norm += v[d] * v[d];
      norm = Math.sqrt(norm);
      if (norm < 1e-6) return null;
      for (let d = 0; d < EMB_DIM; d++) v[d] /= norm;
      return v;
    };

    return {
      hasHistory: state.history.length > 0,
      uRecent: toNormVec(recentEntries),
      uLong: toNormVec(longEntries),
      tagAff,
      authorAff,
      graphNeighbors,
    };
  }

  function scoreUserHistoryForItem(idx, prof) {
    if (!prof.hasHistory) return 0;
    const it = ITEMS[idx];
    const simRecent = prof.uRecent ? dotItemWithVec(idx, prof.uRecent) : 0;
    const simLong = prof.uLong ? dotItemWithVec(idx, prof.uLong) : 0;
    const gBoost = prof.graphNeighbors.get(idx) || 0;

    let tScore = 0;
    for (const kw of it.kw || []) tScore += prof.tagAff.get(kw) || 0;
    tScore = Math.min(1.0, tScore / 4.0);
    const aScore = Math.min(0.6, (prof.authorAff.get(it.a) || 0) / 5.0);

    return 0.45 * simRecent + 0.25 * simLong + 0.18 * gBoost + 0.12 * (tScore + aScore * 0.5);
  }

  // MMR Diversification over top candidate pool
  function mmrSelect(scoredList, limit, lambda = 0.74) {
    const pool = scoredList.slice(0, Math.min(scoredList.length, Math.max(limit * 6, 120)));
    const rest = scoredList.slice(pool.length);
    const selected = [];
    const used = new Uint8Array(pool.length);
    const pickCount = Math.min(limit, pool.length);

    for (let step = 0; step < pickCount; step++) {
      let bestP = -1;
      let bestVal = -1e9;
      const winStart = Math.max(0, selected.length - 4);

      for (let p = 0; p < pool.length; p++) {
        if (used[p]) continue;
        const candIdx = pool[p].idx;
        const cItem = ITEMS[candIdx];

        let maxSim = 0;
        let recentSameAuthor = 0;
        for (let w = winStart; w < selected.length; w++) {
          const prevIdx = selected[w];
          const prevItem = ITEMS[prevIdx];
          const vSim = EMB_I8 ? Math.max(0, dotItems(candIdx, prevIdx)) : 0;
          const sameAuth = cItem.a && cItem.a === prevItem.a ? 0.55 : 0;
          if (cItem.a && cItem.a === prevItem.a) recentSameAuthor++;
          const pen = 0.5 * vSim + sameAuth;
          if (pen > maxSim) maxSim = pen;
        }
        if (recentSameAuthor >= 2) maxSim += 0.8;

        const mmr = lambda * pool[p].score - (1 - lambda) * maxSim;
        if (mmr > bestVal) {
          bestVal = mmr;
          bestP = p;
        }
      }
      if (bestP === -1) break;
      used[bestP] = 1;
      selected.push(pool[bestP].idx);
    }

    if (limit > pickCount) {
      for (let p = 0; p < pool.length; p++) {
        if (!used[p]) selected.push(pool[p].idx);
      }
      for (const r of rest) selected.push(r.idx);
    }
    return selected;
  }

  /**
   * 1) Home Feed: Purely driven by reading history (cold-start falls back to quality prior)
   */
  function rankHomeFeed() {
    const prof = buildHistoryProfile();
    const scored = [];

    for (let i = 0; i < ITEMS.length; i++) {
      const it = ITEMS[i];
      const isRead = Boolean(state.readMap[it.id]);

      let score;
      if (prof.hasHistory) {
        score = 0.88 * scoreUserHistoryForItem(i, prof) + 0.12 * it.q;
      } else {
        score = 0.80 * it.q + 0.20 * Math.min(1, it.deg / 30) + (Math.random() - 0.5) * 0.06;
      }

      if (isRead) score -= 0.85;
      score += (Math.random() - 0.5) * 0.015;
      scored.push({ idx: i, score });
    }

    scored.sort((a, b) => b.score - a.score);
    const head = mmrSelect(scored.slice(0, 160), 70, 0.74);
    const headSet = new Set(head);
    const tail = scored.filter(x => !headSet.has(x.idx)).map(x => x.idx);
    return head.concat(tail);
  }

  /**
   * 2) Bottom Article Recommendations: Finite (6 cards), balancing Current Article Content + User History
   */
  function rankBottomRecommendations(currentIdx, post) {
    const prof = buildHistoryProfile();

    // Direct structural links from current post (citations, co-citations, citing answers)
    const directMap = new Map();
    for (const r of post.rec_chain || []) {
      const rIdx = ID_TO_IDX.get(String(r.id));
      if (rIdx !== undefined) directMap.set(rIdx, 0.90);
    }
    for (const r of post.rec_bridge || []) {
      const rIdx = ID_TO_IDX.get(String(r.id));
      if (rIdx !== undefined) directMap.set(rIdx, Math.max(directMap.get(rIdx) || 0, 0.82));
    }
    for (const r of post.rec_answers || []) {
      const rIdx = ID_TO_IDX.get(String(r.id));
      if (rIdx !== undefined) directMap.set(rIdx, Math.max(directMap.get(rIdx) || 0, 0.78));
    }
    if (NEIGHBORS && currentIdx !== undefined) {
      for (const [nbIdx, wInt] of NEIGHBORS[currentIdx] || []) {
        directMap.set(nbIdx, Math.max(directMap.get(nbIdx) || 0, wInt / 100.0));
      }
    }

    const curTags = new Set(post.keywords || []);
    const scored = [];

    for (let i = 0; i < ITEMS.length; i++) {
      if (i === currentIdx) continue;
      const it = ITEMS[i];
      const isRead = Boolean(state.readMap[it.id]);

      // Content affinity to current article
      const vecSim = (EMB_I8 && currentIdx !== undefined) ? Math.max(0, dotItems(currentIdx, i)) : 0;
      const graphLink = directMap.get(i) || 0;
      let sharedKw = 0;
      for (const kw of it.kw || []) {
        if (curTags.has(kw)) sharedKw += 0.16;
      }
      const contentScore = 0.52 * vecSim + 0.36 * graphLink + Math.min(0.24, sharedKw);

      // User history preference (excluding current article's instant click dominance by using full profile)
      const histScore = prof.hasHistory ? scoreUserHistoryForItem(i, prof) : it.q;

      // Balanced blend: ~58% Current Article Content + ~30% Personal Reading History + ~12% Prior Quality
      let total = 0.58 * contentScore + 0.30 * histScore + 0.12 * it.q;
      if (isRead) total -= 0.75;

      scored.push({ idx: i, score: total });
    }

    scored.sort((a, b) => b.score - a.score);
    return mmrSelect(scored, BOTTOM_REC_LIMIT, 0.70);
  }

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Unified Minimal Card HTML (used on BOTH Home Feed and Bottom of Article)
  function renderCardHTML(idx) {
    const it = ITEMS[idx];
    const isRead = Boolean(state.readMap[it.id]);
    const authorDisplay = it.k === "ans" ? `${it.a} · 回答` : it.a;

    return `
      <article class="post-row ${isRead ? "is-read" : ""}">
        <div class="post-meta">
          <time datetime="${esc(it.d)}">${esc(it.d)}</time>
          <a class="meta-author" href="#/?q=${encodeURIComponent(it.a)}">${esc(authorDisplay)}</a>
          <span>▲ ${it.v} · ${it.m} 分钟</span>
        </div>
        <div class="post-summary">
          <h2>
            <a href="#/post/${esc(it.id)}">
              <span>${esc(it.t)}</span>
              <span class="arr" aria-hidden="true">↗</span>
            </a>
          </h2>
          <p>${esc(it.s)}</p>
          ${(it.kw && it.kw.length) ? `
            <p class="tags">
              ${it.kw.slice(0, 4).map(k => `<a href="#/?q=${encodeURIComponent(k)}">#${esc(k)}</a>`).join("")}
            </p>
          ` : ""}
        </div>
        <i class="rule" aria-hidden="true"></i>
      </article>
    `;
  }

  function renderHomeView(initialQ = "") {
    document.title = "译文漫游";
    searchQuery = initialQ;

    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <div class="search-bar">
          <input id="search-input" type="search" placeholder="搜索标题、内容、作者或标签…" autocomplete="off" spellcheck="false" value="${esc(searchQuery)}">
          <span class="search-count" id="search-count" hidden></span>
          <i class="rule" aria-hidden="true"></i>
        </div>
        <div id="feed-list"></div>
        <div class="feed-more" id="feed-sentinel">向下滚动加载更多</div>
      </section>
    `;

    const input = document.getElementById("search-input");
    const countEl = document.getElementById("search-count");

    const updateList = (q) => {
      searchQuery = q;
      const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!terms.length) {
        countEl.hidden = true;
        currentFeedPool = rankHomeFeed();
      } else {
        const hits = [];
        for (let i = 0; i < ITEMS.length; i++) {
          const it = ITEMS[i];
          const hay = `${it.t} ${it.a} ${it.c} ${(it.kw || []).join(" ")} ${it.s}`.toLowerCase();
          if (terms.every(t => hay.includes(t))) {
            const titleHit = terms.every(t => it.t.toLowerCase().includes(t)) ? 2.0 : 0;
            hits.push({ idx: i, s: titleHit + it.q });
          }
        }
        hits.sort((a, b) => b.s - a.s);
        currentFeedPool = hits.map(h => h.idx);
        countEl.textContent = `${hits.length} 条结果`;
        countEl.hidden = false;
      }

      document.getElementById("feed-list").innerHTML = "";
      renderedCount = 0;
      appendNextBatch(16);
      setupInfiniteObserver();
    };

    input?.addEventListener("input", () => {
      const val = input.value;
      const targetHash = val.trim() ? `#/?q=${encodeURIComponent(val.trim())}` : "#/";
      history.replaceState(null, "", targetHash);
      updateList(val);
    });

    updateList(initialQ);
  }

  function appendNextBatch(size = 14) {
    const listEl = document.getElementById("feed-list");
    const sentinel = document.getElementById("feed-sentinel");
    if (!listEl) return;

    const slice = currentFeedPool.slice(renderedCount, renderedCount + size);
    if (!slice.length) {
      if (sentinel) sentinel.textContent = renderedCount ? "已到底部" : "没有找到匹配的内容";
      return;
    }

    listEl.insertAdjacentHTML("beforeend", slice.map(renderCardHTML).join(""));
    renderedCount += slice.length;
    if (sentinel) {
      sentinel.textContent = renderedCount >= currentFeedPool.length ? "已到底部" : "向下滚动加载更多";
    }
  }

  function setupInfiniteObserver() {
    if (feedObserver) feedObserver.disconnect();
    const sentinel = document.getElementById("feed-sentinel");
    if (!sentinel) return;
    feedObserver = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting) && renderedCount < currentFeedPool.length) {
        appendNextBatch(14);
      }
    }, { rootMargin: "450px 0px" });
    feedObserver.observe(sentinel);
  }

  async function fetchPost(id) {
    if (POST_CACHE.has(id)) return POST_CACHE.get(id);
    const res = await fetch(`posts/${encodeURIComponent(id)}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    POST_CACHE.set(id, data);
    return data;
  }

  async function renderPostView(id) {
    const main = document.getElementById("main");
    main.innerHTML = `<p class="empty">加载正文中…</p>`;
    scrollTo({ top: 0, behavior: "instant" });

    try {
      const post = await fetchPost(id);
      const idx = ID_TO_IDX.get(String(id));
      document.title = post.title;

      // Clicking in = interested!
      recordClickOpen(id);
      activeSession = { id: String(id), startTime: Date.now(), maxScroll: 0, settled: false };

      const tocItemsHTML = (post.toc || [])
        .map(h => `<li class="level-${h.level}"><a href="#${esc(h.id)}" data-toc-target="${esc(h.id)}">${esc(h.title)}</a></li>`)
        .join("");

      const origSourcesHTML = (post.original_source_urls || [])
        .map(u => `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(u)} ↗</a>`)
        .join("<br>");

      const bottomRecIndices = rankBottomRecommendations(idx, post);

      main.innerHTML = `
        <article class="post-article">
          <header class="article-header">
            <p class="article-meta">
              <button type="button" class="back-link" id="btn-back">← 返回</button>
              <span class="sep" aria-hidden="true">/</span>
              <time datetime="${esc(post.date)}">${esc(post.date)}</time>
              <a class="tag" href="#/?q=${encodeURIComponent(post.column)}">${esc(post.column)}</a>
              <a class="tag" href="#/?q=${encodeURIComponent(post.author)}">${esc(post.author)}</a>
              <span>约 ${post.minutes} 分钟</span>
              <span>▲ ${post.votes}</span>
            </p>
            <h1>${esc(post.title)}</h1>
            ${post.kind === "answer" && post.question_detail_html ? `
              <details class="question-fold">
                <summary>查看问题描述</summary>
                <div class="question-fold-body prose">${post.question_detail_html}</div>
              </details>
            ` : ""}
            ${(post.keywords && post.keywords.length) ? `
              <p class="tags">
                ${post.keywords.map(k => `<a href="#/?q=${encodeURIComponent(k)}">#${esc(k)}</a>`).join("")}
              </p>
            ` : ""}
            <i class="rule" aria-hidden="true"></i>
          </header>

          ${tocItemsHTML ? `
            <details class="toc-mobile">
              <summary>目录</summary>
              <nav aria-label="文章目录"><ol>${tocItemsHTML}</ol></nav>
            </details>
          ` : ""}

          <div class="article-body">
            <div class="prose" id="article-prose">
              ${post.restoration ? `
                <aside class="restoration-banner">
                  <strong>${esc(post.restoration.label)}</strong>：${esc(post.restoration.notice)}
                </aside>
              ` : ""}
              ${post.html}
            </div>

            ${tocItemsHTML ? `
              <nav class="toc" aria-label="目录">
                <p>目录</p>
                <ol id="desktop-toc-list">${tocItemsHTML}</ol>
              </nav>
            ` : ""}

            <footer class="colophon">
              <dl>
                <div>
                  <dt>作者</dt>
                  <dd>${(post.original_authors || [post.author]).map(a => `<a href="#/?q=${encodeURIComponent(a)}">${esc(a)}</a>`).join(" · ")}</dd>
                </div>
                ${origSourcesHTML ? `<div><dt>原文</dt><dd>${origSourcesHTML}</dd></div>` : ""}
                <div>
                  <dt>来源</dt>
                  <dd>${esc(post.publisher)} · <a href="${esc(post.zhihu_url)}" target="_blank" rel="noopener noreferrer">知乎原帖 ↗</a></dd>
                </div>
              </dl>
            </footer>
          </div>
        </article>

        <section class="bottom-recs" aria-label="延伸推荐">
          <div class="bottom-recs-head">
            <i class="rule top" aria-hidden="true"></i>
            <h3 class="rec-heading">延伸阅读</h3>
            <span>结合本文内容与你的阅读偏好推荐</span>
            <i class="rule" aria-hidden="true"></i>
          </div>
          <div>
            ${bottomRecIndices.map(renderCardHTML).join("")}
          </div>
        </section>
      `;

      document.getElementById("btn-back")?.addEventListener("click", () => {
        location.hash = "#/";
      });

      main.querySelectorAll("[data-toc-target]").forEach(a => {
        a.addEventListener("click", e => {
          e.preventDefault();
          document.getElementById(a.dataset.tocTarget)?.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });

      main.querySelectorAll('.prose a[href^="#ref-"], .prose a[href^="#cite-"]').forEach(a => {
        a.addEventListener("click", e => {
          e.preventDefault();
          document.getElementById(a.getAttribute("href").slice(1))?.scrollIntoView({ behavior: "smooth", block: "center" });
        });
      });

      setupTocScrollSpy();
    } catch (err) {
      main.innerHTML = `<p class="empty">无法打开文章：${esc(err.message)} · <a href="#/" style="color:var(--accent)">返回首页</a></p>`;
    }
  }

  function setupTocScrollSpy() {
    const headings = [...document.querySelectorAll("#article-prose h2[id], #article-prose h3[id]")];
    const tocLinks = [...document.querySelectorAll("#desktop-toc-list a")];
    if (!headings.length || !tocLinks.length) return;

    const onScroll = () => {
      let activeId = headings[0].id;
      for (const h of headings) {
        if (h.getBoundingClientRect().top <= 96) activeId = h.id;
        else break;
      }
      tocLinks.forEach(a => {
        if (a.dataset.tocTarget === activeId) a.setAttribute("aria-current", "true");
        else a.removeAttribute("aria-current");
      });
    };
    window._tocSpyHandler = onScroll;
    onScroll();
  }

  // Hover Preview (.cite-card) for footnotes & internal article links
  function clearHoverCard() {
    hoverCard?.remove();
    hoverCard = null;
  }

  function handlePointerOver(e) {
    if (e.pointerType && e.pointerType !== "mouse") return;
    const internalLink = e.target.closest?.("a[data-tm-id]");
    const citeLink = e.target.closest?.(".cite a");

    if (internalLink) {
      const idx = ID_TO_IDX.get(internalLink.dataset.tmId);
      if (idx === undefined || hoverCard?._for === internalLink) return;
      clearHoverCard();
      const it = ITEMS[idx];
      hoverCard = document.createElement("div");
      hoverCard.className = "cite-card";
      hoverCard._for = internalLink;
      hoverCard.innerHTML = `
        <div class="cite-card-head">
          <span>${esc(it.a)}</span>
          <span>▲ ${it.v} · ${it.m} 分钟</span>
        </div>
        <div class="cite-card-title">${esc(it.t)}</div>
        <div>${esc(it.s)}</div>
      `;
      document.body.appendChild(hoverCard);
      positionHoverCard(internalLink);
    } else if (citeLink) {
      if (hoverCard?._for === citeLink) return;
      const refId = citeLink.getAttribute("href")?.slice(1);
      const refEl = refId ? document.getElementById(refId)?.cloneNode(true) : null;
      if (!refEl) return;
      clearHoverCard();
      refEl.querySelector(".ref-back")?.remove();
      hoverCard = document.createElement("div");
      hoverCard.className = "cite-card";
      hoverCard._for = citeLink;
      const bodyDiv = document.createElement("div");
      bodyDiv.append(...refEl.childNodes);
      hoverCard.appendChild(bodyDiv);
      document.body.appendChild(hoverCard);
      positionHoverCard(citeLink);
    }
  }

  function positionHoverCard(anchor) {
    if (!hoverCard) return;
    const r = anchor.getBoundingClientRect();
    const w = hoverCard.offsetWidth;
    hoverCard.style.left = scrollX + Math.max(10, Math.min(r.left - 12, innerWidth - w - 10)) + "px";
    hoverCard.style.top = scrollY + r.bottom + 7 + "px";
  }

  function route() {
    clearHoverCard();
    flushActiveReading();
    window._tocSpyHandler = null;
    if (feedObserver) feedObserver.disconnect();

    const hash = location.hash || "#/";
    if (hash.startsWith("#/post/")) {
      const id = decodeURIComponent(hash.slice("#/post/".length));
      renderPostView(id);
    } else if (hash.startsWith("#/?q=")) {
      const q = decodeURIComponent(hash.slice("#/?q=".length));
      renderHomeView(q);
    } else {
      renderHomeView("");
    }
  }

  function bindGlobalEvents() {
    const root = document.documentElement;
    const isDark = () => (root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";

    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch {}
    });

    document.getElementById("to-top-btn")?.addEventListener("click", () => {
      scrollTo({ top: 0, behavior: "smooth" });
    });

    addEventListener("scroll", () => {
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - innerHeight);
      const ratio = Math.min(1, Math.max(0, scrollY / maxScroll));
      const bar = document.getElementById("scroll-progress");
      if (bar) bar.style.transform = `scaleX(${ratio.toFixed(4)})`;

      const topBtn = document.getElementById("to-top-btn");
      if (topBtn) topBtn.classList.toggle("show", scrollY > 360);

      if (activeSession && ratio > activeSession.maxScroll) {
        activeSession.maxScroll = ratio;
      }
      if (typeof window._tocSpyHandler === "function") {
        window._tocSpyHandler();
      }
    }, { passive: true });

    document.addEventListener("pointerover", handlePointerOver, { passive: true });
    document.addEventListener("pointerout", e => {
      if (hoverCard && !hoverCard.contains(e.relatedTarget) && !hoverCard._for?.contains(e.relatedTarget)) {
        clearHoverCard();
      }
    }, { passive: true });

    addEventListener("hashchange", route);
    addEventListener("beforeunload", flushActiveReading);
  }

  async function boot() {
    bindGlobalEvents();
    try {
      const indexRes = await fetch("data/feed-index.json");
      if (!indexRes.ok) throw new Error(`HTTP ${indexRes.status}`);
      FEED_DATA = await indexRes.json();
      ITEMS = FEED_DATA.items;
      ITEMS.forEach((it, idx) => ID_TO_IDX.set(String(it.id), idx));

      // Load 128-d Int8 vectors & graph neighbors first if fast, so initial home ranking uses full vectors
      const [buf, nbs] = await Promise.all([
        fetch("data/embeddings.bin").then(r => r.ok ? r.arrayBuffer() : null).catch(() => null),
        fetch("data/neighbors.json").then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      if (buf) EMB_I8 = new Int8Array(buf);
      if (nbs) NEIGHBORS = nbs;

      route();
    } catch (err) {
      const bootEl = document.getElementById("boot-loading");
      if (bootEl) bootEl.textContent = `加载失败：${err.message}`;
    }
  }

  boot();
})();
