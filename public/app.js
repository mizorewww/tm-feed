/**
 * TM-Feed Progressive Enhancement Script
 * - Every page (/ and /posts/<id>.html) is already a complete static HTML document for curl / no-JS.
 * - In browsers, this script enhances:
 *   1) Implicit reading history recording (opening a post = interested, weighted by dwell & scroll depth)
 *   2) Home page: re-ranks feed purely from personal reading history + live instant search + infinite scroll
 *   3) Post page: re-ranks the 6 bottom cards by blending Current Article Content + Personal Reading History
 */

(() => {
  const EMB_DIM = 128;
  const STORAGE_KEY = "tm_feed_history_v2";
  const BOTTOM_REC_LIMIT = 6;

  const body = document.body;
  const PAGE_KIND = body.dataset.page || "home"; // "home" | "post"
  const ROOT = body.dataset.root || ".";
  const CURRENT_POST_ID = body.dataset.postId || "";

  let ITEMS = [];
  let ID_TO_IDX = new Map();
  let EMB_I8 = null;
  let NEIGHBORS = null;

  let currentFeedPool = [];
  let renderedCount = 0;
  let feedObserver = null;
  let hoverCard = null;

  let state = loadState();
  const sessionStart = Date.now();
  let maxScrollRatio = 0;
  let dwellSettled = false;

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

  function recordPostOpen(id) {
    const sid = String(id);
    if (!sid) return;
    const now = Date.now();
    const prev = state.readMap[sid] || { count: 0, dwell: 0, ts: now };
    state.readMap[sid] = { count: prev.count + 1, dwell: prev.dwell, ts: now };

    const existing = state.history.find(h => h.id === sid);
    const baseW = existing ? Math.min(5.0, existing.weight + 0.8) : 1.6;
    state.history = [
      { id: sid, weight: baseW, ts: now },
      ...state.history.filter(h => h.id !== sid),
    ].slice(0, 60);
    saveState();
  }

  function settlePostDwell() {
    if (PAGE_KIND !== "post" || !CURRENT_POST_ID || dwellSettled) return;
    dwellSettled = true;
    const sec = (Date.now() - sessionStart) / 1000;
    if (state.readMap[CURRENT_POST_ID]) {
      state.readMap[CURRENT_POST_ID].dwell = Math.round((state.readMap[CURRENT_POST_ID].dwell || 0) + sec);
    }
    const entry = state.history.find(h => h.id === CURRENT_POST_ID);
    if (entry) {
      if (sec >= 20 || maxScrollRatio >= 0.5) {
        const bonus = maxScrollRatio >= 0.8 ? 1.8 : 1.0;
        entry.weight = Number(Math.min(6.0, entry.weight + bonus).toFixed(2));
      } else if (sec < 3 && maxScrollRatio < 0.08 && state.readMap[CURRENT_POST_ID]?.count === 1) {
        entry.weight = 0.35;
      }
    }
    saveState();
  }

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

  function buildHistoryProfile(excludeId = "") {
    const recentEntries = [];
    const longEntries = [];
    const tagAff = new Map();
    const authorAff = new Map();
    const graphNeighbors = new Map();

    const filteredHist = excludeId
      ? state.history.filter(h => h.id !== excludeId)
      : state.history;

    filteredHist.forEach((h, pos) => {
      const idx = ID_TO_IDX.get(h.id);
      if (idx === undefined) return;
      const it = ITEMS[idx];
      const w = h.weight || 1.0;

      if (pos < 8) recentEntries.push({ idx, w: w * Math.pow(0.78, pos) });
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
      hasHistory: filteredHist.length > 0,
      uRecent: toNormVec(recentEntries),
      uLong: toNormVec(longEntries),
      tagAff,
      authorAff,
      graphNeighbors,
    };
  }

  function scoreHistoryForItem(idx, prof) {
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

  function rankHomeFeed() {
    const prof = buildHistoryProfile();
    const scored = [];
    for (let i = 0; i < ITEMS.length; i++) {
      const it = ITEMS[i];
      const isRead = Boolean(state.readMap[it.id]);
      let score = prof.hasHistory
        ? 0.88 * scoreHistoryForItem(i, prof) + 0.12 * it.q
        : 0.80 * it.q + 0.20 * Math.min(1, it.deg / 30) + (Math.random() - 0.5) * 0.05;
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

  function rankBottomRecommendations(currentIdx) {
    const prof = buildHistoryProfile(CURRENT_POST_ID);
    const directMap = new Map();
    if (NEIGHBORS && currentIdx !== undefined) {
      for (const [nbIdx, wInt] of NEIGHBORS[currentIdx] || []) {
        directMap.set(nbIdx, wInt / 100.0);
      }
    }
    const curItem = ITEMS[currentIdx];
    const curTags = new Set(curItem?.kw || []);
    const scored = [];

    for (let i = 0; i < ITEMS.length; i++) {
      if (i === currentIdx) continue;
      const it = ITEMS[i];
      const isRead = Boolean(state.readMap[it.id]);

      const vecSim = (EMB_I8 && currentIdx !== undefined) ? Math.max(0, dotItems(currentIdx, i)) : 0;
      const graphLink = directMap.get(i) || 0;
      let sharedKw = 0;
      for (const kw of it.kw || []) {
        if (curTags.has(kw)) sharedKw += 0.15;
      }
      const contentScore = 0.54 * vecSim + 0.34 * graphLink + Math.min(0.22, sharedKw);
      const histScore = prof.hasHistory ? scoreHistoryForItem(i, prof) : it.q;

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

  function renderCardHTML(idx) {
    const it = ITEMS[idx];
    const isRead = Boolean(state.readMap[it.id]);
    const authorDisplay = it.k === "ans" ? `${it.a} · 回答` : it.a;
    const postHref = ROOT === "." ? `posts/${it.id}.html` : `${it.id}.html`;
    const qBase = ROOT === "." ? "./" : "../";

    return `
      <article class="post-row ${isRead ? "is-read" : ""}" data-item-id="${esc(it.id)}">
        <div class="post-meta">
          <time datetime="${esc(it.d)}">${esc(it.d)}</time>
          <a class="meta-author" href="${qBase}?q=${encodeURIComponent(it.a)}">${esc(authorDisplay)}</a>
          <span>▲ ${it.v} · ${it.m} 分钟</span>
        </div>
        <div class="post-summary">
          <h2>
            <a href="${esc(postHref)}">
              <span>${esc(it.t)}</span>
              <span class="arr" aria-hidden="true">↗</span>
            </a>
          </h2>
          <p>${esc(it.s)}</p>
          ${(it.kw && it.kw.length) ? `
            <p class="tags">
              ${it.kw.slice(0, 4).map(k => `<a href="${qBase}?q=${encodeURIComponent(k)}">#${esc(k)}</a>`).join("")}
            </p>
          ` : ""}
        </div>
        <i class="rule" aria-hidden="true"></i>
      </article>
    `;
  }

  function initHomeEnhancement() {
    const input = document.getElementById("search-input");
    const countEl = document.getElementById("search-count");
    const listEl = document.getElementById("feed-list");
    const sentinel = document.getElementById("feed-sentinel");
    if (!listEl) return;

    const appendBatch = (size = 16) => {
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
    };

    const runFilter = (q) => {
      const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
      if (!terms.length) {
        if (countEl) countEl.hidden = true;
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
        if (countEl) {
          countEl.textContent = `${hits.length} 条结果`;
          countEl.hidden = false;
        }
      }
      listEl.innerHTML = "";
      renderedCount = 0;
      appendBatch(16);

      if (feedObserver) feedObserver.disconnect();
      if (sentinel) {
        feedObserver = new IntersectionObserver(entries => {
          if (entries.some(e => e.isIntersecting) && renderedCount < currentFeedPool.length) {
            appendBatch(14);
          }
        }, { rootMargin: "450px 0px" });
        feedObserver.observe(sentinel);
      }
    };

    const params = new URLSearchParams(location.search);
    const initialQ = params.get("q") || "";
    if (input && initialQ) input.value = initialQ;

    input?.addEventListener("input", () => {
      const val = input.value;
      const nextUrl = val.trim() ? `?q=${encodeURIComponent(val.trim())}` : location.pathname;
      history.replaceState(null, "", nextUrl);
      runFilter(val);
    });

    // Clicking tag/author links on Home intercepts cleanly without full page reload
    listEl.addEventListener("click", e => {
      const qLink = e.target.closest?.('a[href^="./?q="], a[href^="?q="]');
      if (qLink && input) {
        e.preventDefault();
        const u = new URL(qLink.href, location.href);
        const qVal = u.searchParams.get("q") || "";
        input.value = qVal;
        history.replaceState(null, "", `?q=${encodeURIComponent(qVal)}`);
        scrollTo({ top: 0, behavior: "smooth" });
        runFilter(qVal);
      }
    });

    // Only replace initial static cards if user has history OR a search query, else keep static first batch & enable infinite scroll!
    if (initialQ || state.history.length > 0) {
      runFilter(initialQ);
    } else {
      const staticIds = new Set([...listEl.querySelectorAll(".post-row[data-item-id]")].map(el => el.dataset.itemId));
      const ranked = rankHomeFeed();
      const rest = ranked.filter(idx => !staticIds.has(ITEMS[idx].id));
      currentFeedPool = [...staticIds].map(id => ID_TO_IDX.get(id)).filter(x => x !== undefined).concat(rest);
      renderedCount = staticIds.size;
      if (sentinel) {
        feedObserver = new IntersectionObserver(entries => {
          if (entries.some(e => e.isIntersecting) && renderedCount < currentFeedPool.length) {
            appendBatch(14);
          }
        }, { rootMargin: "450px 0px" });
        feedObserver.observe(sentinel);
      }
    }
  }

  function initPostEnhancement() {
    const curIdx = ID_TO_IDX.get(CURRENT_POST_ID);
    const bottomEl = document.getElementById("bottom-rec-list");

    // Personalize bottom 6 cards if user has reading history (or mark read states)
    if (bottomEl && curIdx !== undefined && state.history.length > 1) {
      const recIndices = rankBottomRecommendations(curIdx);
      bottomEl.innerHTML = recIndices.map(renderCardHTML).join("");
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

  function clearHoverCard() {
    hoverCard?.remove();
    hoverCard = null;
  }

  function handlePointerOver(e) {
    if (e.pointerType && e.pointerType !== "mouse") return;
    const internalLink = e.target.closest?.("a[data-tm-id]");
    const citeLink = e.target.closest?.(".cite a");

    if (internalLink && ITEMS.length) {
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

  function bindEvents() {
    const rootEl = document.documentElement;
    const isDark = () => (rootEl.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";

    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      rootEl.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("theme", rootEl.dataset.theme); } catch {}
    });

    document.getElementById("to-top-btn")?.addEventListener("click", () => {
      scrollTo({ top: 0, behavior: "smooth" });
    });

    if (PAGE_KIND === "post") {
      recordPostOpen(CURRENT_POST_ID);
      setupTocScrollSpy();
      addEventListener("pagehide", settlePostDwell);
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden") settlePostDwell();
      });
    }

    addEventListener("scroll", () => {
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - innerHeight);
      const ratio = Math.min(1, Math.max(0, scrollY / maxScroll));
      const bar = document.getElementById("scroll-progress");
      if (bar) bar.style.transform = `scaleX(${ratio.toFixed(4)})`;

      const topBtn = document.getElementById("to-top-btn");
      if (topBtn) topBtn.classList.toggle("show", scrollY > 360);

      if (ratio > maxScrollRatio) maxScrollRatio = ratio;
      if (typeof window._tocSpyHandler === "function") window._tocSpyHandler();
    }, { passive: true });

    document.addEventListener("pointerover", handlePointerOver, { passive: true });
    document.addEventListener("pointerout", e => {
      if (hoverCard && !hoverCard.contains(e.relatedTarget) && !hoverCard._for?.contains(e.relatedTarget)) {
        clearHoverCard();
      }
    }, { passive: true });
  }

  async function boot() {
    bindEvents();
    try {
      const [indexData, buf, nbs] = await Promise.all([
        fetch(`${ROOT}/data/feed-index.json`).then(r => r.ok ? r.json() : null),
        fetch(`${ROOT}/data/embeddings.bin`).then(r => r.ok ? r.arrayBuffer() : null).catch(() => null),
        fetch(`${ROOT}/data/neighbors.json`).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      if (!indexData) return;
      ITEMS = indexData.items || [];
      ITEMS.forEach((it, idx) => ID_TO_IDX.set(String(it.id), idx));
      if (buf) EMB_I8 = new Int8Array(buf);
      if (nbs) NEIGHBORS = nbs;

      if (PAGE_KIND === "home") {
        initHomeEnhancement();
      } else if (PAGE_KIND === "post") {
        initPostEnhancement();
      }
    } catch {}
  }

  boot();
})();
