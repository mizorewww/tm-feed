/**
 * TM-Feed: Local-First Hybrid Recommender & Editorial Reader
 * - 128-d Int8 Semantic + Co-Citation SVD Vectors
 * - Dual-timescale User Profile (short-term session vector + long-term preference + negative pool)
 * - 1-Hop Graph Propagation + Bandit Exploration + MMR Sliding-Window Diversification
 */

(() => {
  const EMB_DIM = 128;
  const STORAGE_KEY = "tm_feed_profile_v1";
  const QUICK_TOPICS = ["间隔重复", "检索练习", "学校教育", "学习动机", "助记媒介", "常青笔记", "自主学习", "二语习得", "刻意练习"];

  // Global Corpus State
  let FEED_DATA = null;      // { stats, columns, top_keywords, top_authors, items }
  let ITEMS = [];            // Compact item objects [0..N-1]
  let ID_TO_IDX = new Map(); // id -> index
  let EMB_I8 = null;         // Int8Array (N * 128)
  let NEIGHBORS = null;      // Array<Array<[nbIdx, weightInt]>>
  const POST_CACHE = new Map();

  // Feed Pagination & Session State
  let currentFeedPool = [];  // Ranked [{ idx, reason, badgeClass }]
  let renderedCount = 0;
  let feedMode = "all";      // "all" | "art" | "ans"
  let activeTopic = "";      // Quick topic filter on home feed
  let feedObserver = null;
  let activeDwell = null;    // { id, idx, startTime, maxScroll, settled }
  let hoverCard = null;

  // Persistent User Profile
  let profile = loadProfile();

  function defaultProfile() {
    return {
      shortHistory: [], // [{ id, weight, ts }] last 8 positive engagements
      liked: {},        // id -> timestamp
      saved: {},        // id -> timestamp
      read: {},         // id -> { count, dwell, ts }
      dismissed: {},    // id -> timestamp
      impressions: {},  // id -> count (unclicked exposures)
      tagWeights: {},   // tag -> float
      authorWeights: {},// author -> float
      colWeights: {},   // column -> float
    };
  }

  function loadProfile() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return defaultProfile();
      return { ...defaultProfile(), ...JSON.parse(raw) };
    } catch {
      return defaultProfile();
    }
  }

  function saveProfile() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
    } catch {}
    updateSavedBadge();
  }

  function updateSavedBadge() {
    const badge = document.getElementById("saved-count");
    if (!badge) return;
    const cnt = Object.keys(profile.saved).length;
    if (cnt > 0) {
      badge.textContent = String(cnt);
      badge.hidden = false;
    } else {
      badge.hidden = true;
    }
  }

  // Record explicit or implicit user feedback
  function recordSignal(id, delta, kind = "read") {
    const idx = ID_TO_IDX.get(String(id));
    if (idx === undefined) return;
    const item = ITEMS[idx];
    const now = Date.now();

    if (delta > 0) {
      profile.shortHistory = [
        { id: item.id, weight: delta, ts: now },
        ...profile.shortHistory.filter(x => x.id !== item.id),
      ].slice(0, 10);

      for (const kw of item.kw || []) {
        profile.tagWeights[kw] = Number(((profile.tagWeights[kw] || 0) + delta * 0.55).toFixed(2));
      }
      if (item.a) {
        profile.authorWeights[item.a] = Number(((profile.authorWeights[item.a] || 0) + delta * 0.35).toFixed(2));
      }
      if (item.c && item.c !== "杂项" && item.c !== "未分类") {
        profile.colWeights[item.c] = Number(((profile.colWeights[item.c] || 0) + delta * 0.3).toFixed(2));
      }
      delete profile.impressions[item.id];
    } else if (delta < 0) {
      for (const kw of item.kw || []) {
        profile.tagWeights[kw] = Number(Math.max(-4, (profile.tagWeights[kw] || 0) + delta * 0.4).toFixed(2));
      }
      if (item.a) {
        profile.authorWeights[item.a] = Number(Math.max(-4, (profile.authorWeights[item.a] || 0) + delta * 0.4).toFixed(2));
      }
    }

    if (kind === "read") {
      const prev = profile.read[item.id] || { count: 0, dwell: 0 };
      profile.read[item.id] = { count: prev.count + 1, dwell: prev.dwell, ts: now };
    }
    saveProfile();
  }

  // Vector Math Helpers (Int8 normalized vectors -> Float32 user vectors)
  function dotItemWithFloatVec(itemIdx, floatVec) {
    if (!EMB_I8 || !floatVec) return 0;
    const base = itemIdx * EMB_DIM;
    let sum = 0;
    for (let d = 0; d < EMB_DIM; d++) {
      sum += (EMB_I8[base + d] / 127.0) * floatVec[d];
    }
    return sum;
  }

  function dotItems(idxA, idxB) {
    if (!EMB_I8) return 0;
    const bA = idxA * EMB_DIM;
    const bB = idxB * EMB_DIM;
    let sum = 0;
    for (let d = 0; d < EMB_DIM; d++) {
      sum += EMB_I8[bA + d] * EMB_I8[bB + d];
    }
    return sum / (127.0 * 127.0);
  }

  function buildWeightedVector(entries) {
    if (!EMB_I8 || !entries.length) return null;
    const vec = new Float32Array(EMB_DIM);
    let totalW = 0;
    for (const { idx, w } of entries) {
      const base = idx * EMB_DIM;
      for (let d = 0; d < EMB_DIM; d++) {
        vec[d] += (EMB_I8[base + d] / 127.0) * w;
      }
      totalW += Math.abs(w);
    }
    if (totalW === 0) return null;
    let norm = 0;
    for (let d = 0; d < EMB_DIM; d++) norm += vec[d] * vec[d];
    norm = Math.sqrt(norm);
    if (norm < 1e-6) return null;
    for (let d = 0; d < EMB_DIM; d++) vec[d] /= norm;
    return vec;
  }

  /**
   * Core 4-Channel Hybrid Scorer + MMR Sliding-Window Diversifier
   * @param {Object} opts { anchorIdx: number|null, mode: "all"|"art"|"ans", topic: string, shuffleSeed: number }
   */
  function computeRecommendedFeed(opts = {}) {
    const { anchorIdx = null, mode = "all", topic = "", jitter = 0.04 } = opts;

    // 1. Build Short-Term Vector (last 6 interactions with exponential decay)
    const shortEntries = [];
    profile.shortHistory.slice(0, 6).forEach((h, pos) => {
      const idx = ID_TO_IDX.get(h.id);
      if (idx !== undefined) {
        shortEntries.push({ idx, w: h.weight * Math.pow(0.76, pos) });
      }
    });
    if (anchorIdx !== null) {
      shortEntries.unshift({ idx: anchorIdx, w: 3.5 });
    }
    const uShort = buildWeightedVector(shortEntries);

    // 2. Build Long-Term Vector (liked + saved + completed reads)
    const longEntries = [];
    for (const id of Object.keys(profile.liked)) {
      const idx = ID_TO_IDX.get(id);
      if (idx !== undefined) longEntries.push({ idx, w: 3.0 });
    }
    for (const id of Object.keys(profile.saved)) {
      const idx = ID_TO_IDX.get(id);
      if (idx !== undefined) longEntries.push({ idx, w: 2.4 });
    }
    for (const [id, r] of Object.entries(profile.read)) {
      const idx = ID_TO_IDX.get(id);
      if (idx !== undefined && (r.dwell >= 15 || r.count >= 1)) {
        longEntries.push({ idx, w: 1.2 });
      }
    }
    const uLong = buildWeightedVector(longEntries.slice(-30));

    // 3. Build Negative Suppression Vector
    const negEntries = [];
    for (const id of Object.keys(profile.dismissed)) {
      const idx = ID_TO_IDX.get(id);
      if (idx !== undefined) negEntries.push({ idx, w: 1.0 });
    }
    const uNeg = buildWeightedVector(negEntries.slice(-15));

    // 4. Build 1-Hop Graph Propagation Map from recent positive seeds
    const graphBoost = new Map(); // candIdx -> { score, seedTitle }
    if (NEIGHBORS) {
      const seeds = anchorIdx !== null
        ? [{ idx: anchorIdx, w: 1.3 }]
        : shortEntries.slice(0, 4).map(e => ({ idx: e.idx, w: Math.min(1.2, e.w / 2.5) }));
      for (const s of seeds) {
        const seedTitle = ITEMS[s.idx]?.t || "";
        const nbs = NEIGHBORS[s.idx] || [];
        for (const [nbIdx, wInt] of nbs) {
          const val = (wInt / 100.0) * s.w;
          const prev = graphBoost.get(nbIdx);
          if (!prev || val > prev.score) {
            graphBoost.set(nbIdx, { score: val, seedTitle, seedAuthor: ITEMS[s.idx]?.a });
          }
        }
      }
    }

    const hasProfile = Boolean(uShort || uLong || Object.keys(profile.tagWeights).length > 0);
    const candidates = [];

    for (let i = 0; i < ITEMS.length; i++) {
      if (i === anchorIdx) continue;
      const it = ITEMS[i];
      if (profile.dismissed[it.id]) continue;
      if (mode === "art" && it.k !== "art") continue;
      if (mode === "ans" && it.k !== "ans") continue;
      if (topic && !(it.kw?.includes(topic) || it.c.includes(topic))) continue;

      const isRead = Boolean(profile.read[it.id]);
      const impCount = profile.impressions[it.id] || 0;

      const simShort = uShort ? dotItemWithFloatVec(i, uShort) : 0;
      const simLong = uLong ? dotItemWithFloatVec(i, uLong) : 0;
      const simNeg = uNeg ? Math.max(0, dotItemWithFloatVec(i, uNeg)) : 0;
      const gInfo = graphBoost.get(i);
      const gScore = gInfo ? gInfo.score : 0;

      // Explicit Tag & Author Affinity
      let tagAff = 0;
      let bestTag = "";
      let bestTagW = 0.4;
      for (const kw of it.kw || []) {
        const tw = profile.tagWeights[kw] || 0;
        tagAff += tw;
        if (tw > bestTagW) {
          bestTagW = tw;
          bestTag = kw;
        }
      }
      tagAff = Math.max(-0.5, Math.min(1.0, tagAff / 5.0));
      const authAff = Math.max(-0.4, Math.min(0.6, (profile.authorWeights[it.a] || 0) / 6.0));

      let rel = 0;
      if (hasProfile) {
        rel = 0.38 * simShort
            + 0.22 * simLong
            + 0.16 * gScore
            + 0.12 * (tagAff + authAff * 0.5)
            + 0.18 * it.q
            - 0.32 * simNeg;
      } else {
        // Cold-start quality + curated exploration
        rel = 0.78 * it.q + 0.22 * Math.min(1, it.deg / 30);
      }

      // Penalize already-read items (push way down so unread gems surface first) and impression fatigue
      if (isRead) rel -= 0.65;
      rel -= Math.min(0.32, impCount * 0.055);
      // Controlled exploration noise (Bandit feel on refresh)
      rel += (Math.random() - 0.5) * jitter;

      // Explainable Recommendation Reason Badge
      let reason = "";
      let badgeClass = "";
      if (gInfo && gInfo.score > 0.52 && ITEMS[anchorIdx ?? -1]?.a !== it.a && it.k === "art") {
        reason = `跨界共鸣 · ${it.a}`;
        badgeClass = "bridge";
      } else if (gInfo && gInfo.score > 0.48 && gInfo.seedTitle) {
        const shortSeed = gInfo.seedTitle.length > 10 ? gInfo.seedTitle.slice(0, 10) + "…" : gInfo.seedTitle;
        reason = `顺着《${shortSeed}》延伸`;
      } else if (bestTag) {
        reason = `近期偏好 · ${bestTag}`;
      } else if (simShort > 0.42 && shortEntries.length > 0) {
        const topKw = it.kw?.[0] || it.c;
        reason = `语义近邻 · ${topKw}`;
      } else if (it.k === "ans") {
        reason = `知乎问答实战 · 引 ${it.deg} 篇译文`;
        badgeClass = "qa";
      } else if (it.v >= 500) {
        reason = `全站高赞经典 · ${it.v} 赞`;
      } else {
        reason = `探索发现 · ${it.c}`;
        badgeClass = "bridge";
      }

      candidates.push({ idx: i, rel, reason, badgeClass });
    }

    candidates.sort((a, b) => b.rel - a.rel);

    // 5. MMR Sliding-Window Reranking on Top 180 candidates (breaks 92% same-author echo chamber!)
    const poolTop = candidates.slice(0, 180);
    const tail = candidates.slice(180);
    const selected = [];
    const used = new Uint8Array(poolTop.length);
    const lambda = 0.72;

    const targetLen = Math.min(80, poolTop.length);
    for (let step = 0; step < targetLen; step++) {
      let bestPos = -1;
      let bestMmr = -1e9;
      const windowStart = Math.max(0, selected.length - 4);

      // Gentle 3:1 article-to-answer cadence when mode === "all"
      const wantAnswer = mode === "all" && step % 4 === 2;

      for (let p = 0; p < poolTop.length; p++) {
        if (used[p]) continue;
        const cand = poolTop[p];
        const cItem = ITEMS[cand.idx];

        let maxPenalty = 0;
        let sameAuthorCount = 0;
        for (let w = windowStart; w < selected.length; w++) {
          const prevIdx = selected[w].idx;
          const prevItem = ITEMS[prevIdx];
          const vecSim = EMB_I8 ? Math.max(0, dotItems(cand.idx, prevIdx)) : 0;
          const sameAuth = (cItem.a && cItem.a === prevItem.a) ? 0.55 : 0;
          const sameCol = (cItem.c && cItem.c === prevItem.c && cItem.k === "art") ? 0.25 : 0;
          if (cItem.a && cItem.a === prevItem.a) sameAuthorCount++;
          const pen = 0.50 * vecSim + sameAuth + sameCol;
          if (pen > maxPenalty) maxPenalty = pen;
        }

        // Hard block >= 2 consecutive same author in last 3 cards
        if (sameAuthorCount >= 2) maxPenalty += 0.8;

        let cadenceBonus = 0;
        if (mode === "all") {
          if (wantAnswer && cItem.k === "ans") cadenceBonus = 0.06;
          if (!wantAnswer && cItem.k === "ans" && selected.length > 0 && ITEMS[selected[selected.length - 1].idx].k === "ans") {
            cadenceBonus = -0.14; // Avoid two answers back-to-back
          }
        }

        const mmr = lambda * cand.rel - (1 - lambda) * maxPenalty + cadenceBonus;
        if (mmr > bestMmr) {
          bestMmr = mmr;
          bestPos = p;
        }
      }

      if (bestPos === -1) break;
      used[bestPos] = 1;
      selected.push(poolTop[bestPos]);
    }

    for (let p = 0; p < poolTop.length; p++) {
      if (!used[p]) selected.push(poolTop[p]);
    }
    return selected.concat(tail);
  }

  // Escape HTML safely
  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Render single editorial .post-row card
  function renderCardHTML(entry) {
    const it = ITEMS[entry.idx];
    const isLiked = Boolean(profile.liked[it.id]);
    const isSaved = Boolean(profile.saved[it.id]);
    const isRead = Boolean(profile.read[it.id]);
    const kindLabel = it.k === "ans" ? "知乎回答" : "译文专栏";

    return `
      <article class="post-row ${isRead ? "is-read" : ""}" data-item-id="${esc(it.id)}" data-item-idx="${entry.idx}">
        <div class="post-meta">
          ${entry.reason ? `<span class="rec-badge ${esc(entry.badgeClass || "")}">${esc(entry.reason)}</span>` : ""}
          <a class="meta-author" href="#/author/${encodeURIComponent(it.a)}">${esc(it.a)}</a>
          <a class="meta-col" href="#/column/${encodeURIComponent(it.c)}">${esc(it.c)}</a>
          <span class="meta-stats">${esc(it.d)}</span>
          <span class="meta-stats">▲ ${it.v + (isLiked ? 1 : 0)} 赞 · 约 ${it.m} 分钟</span>
        </div>
        <div class="post-summary">
          <div class="post-head-line">
            <span class="kind-tag ${it.k === "ans" ? "ans" : ""}">${kindLabel}</span>
            ${isRead ? `<span>已读</span>` : ""}
          </div>
          <h2>
            <a href="#/post/${esc(it.id)}">
              <span>${esc(it.t)}</span>
              <span class="arr" aria-hidden="true">↗</span>
            </a>
          </h2>
          <p class="excerpt">${esc(it.s)}</p>
          <div class="card-foot">
            <p class="tags">
              ${(it.kw || []).slice(0, 4).map(k => `<a href="#/tag/${encodeURIComponent(k)}">#${esc(k)}</a>`).join("")}
            </p>
            <div class="card-actions">
              <button type="button" class="act-chip js-like ${isLiked ? "active" : ""}" data-id="${esc(it.id)}">${isLiked ? "★ 已感兴趣" : "👍 感兴趣"}</button>
              <button type="button" class="act-chip js-save ${isSaved ? "saved" : ""}" data-id="${esc(it.id)}">${isSaved ? "🔖 已收藏" : "稍后再读"}</button>
              <button type="button" class="act-chip dismiss js-dismiss" data-id="${esc(it.id)}" title="减少此类推荐">✕</button>
            </div>
          </div>
        </div>
        <i class="rule" aria-hidden="true"></i>
      </article>
    `;
  }

  // Render Home Personalized Feed
  function renderHomeView(jitter = 0.04) {
    setActiveNav("feed");
    document.title = "漫游 · Thoughts Memo 译文与问答推荐流";

    const topTags = Object.entries(profile.tagWeights)
      .filter(([, w]) => w > 0.2)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const readCount = Object.keys(profile.read).length;
    const likedCount = Object.keys(profile.liked).length;
    const topInterestLabel = topTags.length ? `#${topTags[0][0]}` : "冷启动探索";

    // Build proportional .cat-bar segments from user affinities or default top columns
    const barItems = topTags.length >= 2
      ? topTags.map(([name, w]) => ({ name, val: Math.max(1, Math.round(w * 10)) }))
      : FEED_DATA.columns.slice(0, 5).map(c => ({ name: c.name, val: c.count }));

    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <div class="ledger">
          <div class="ledger-meta">
            <span class="ledger-label">RECOMMEND / 漫游引擎</span>
            <span id="profile-summary-line">${readCount + likedCount > 0 ? `已学习 ${readCount + likedCount} 条阅读信号` : "高赞先验 + 跨学派探索"}</span>
            <div class="ledger-actions">
              <button type="button" class="textbtn" id="btn-refresh-feed">换一批 ↻</button>
              ${readCount + likedCount > 0 ? `<button type="button" class="textbtn" id="btn-reset-profile">重置画像</button>` : ""}
            </div>
          </div>
          <div class="ledger-body">
            <dl class="ledger-kpis">
              <div><dt>精校译文</dt><dd>${FEED_DATA.stats.articles.toLocaleString()}<span class="kpi-u">篇</span></dd></div>
              <div><dt>知乎高赞回答</dt><dd>${FEED_DATA.stats.answers.toLocaleString()}<span class="kpi-u">篇</span></dd></div>
              <div><dt>已读 / 感兴趣</dt><dd>${readCount} / ${likedCount}<span class="kpi-u">篇</span></dd></div>
              <div><dt>当前焦点偏好</dt><dd class="interest-text">${esc(topInterestLabel)}</dd></div>
            </dl>
            <div class="ledger-cats">
              <div class="cat-bar" aria-hidden="true">
                ${barItems.map((b, i) => `<i class="c${i % 5}" style="flex:${b.val}" title="${esc(b.name)}"></i>`).join("")}
              </div>
              <div class="feed-toolbar">
                <div class="topic-pills" role="group" aria-label="兴趣话题快筛">
                  <button type="button" class="pill-btn ${activeTopic === "" ? "active" : ""}" data-topic="">全部推荐</button>
                  ${QUICK_TOPICS.map(t => `<button type="button" class="pill-btn ${activeTopic === t ? "active" : ""}" data-topic="${esc(t)}">${esc(t)}</button>`).join("")}
                </div>
                <div class="mode-switch" role="group" aria-label="体裁筛选">
                  <button type="button" class="${feedMode === "all" ? "active" : ""}" data-mode="all">混排</button>
                  <button type="button" class="${feedMode === "art" ? "active" : ""}" data-mode="art">仅译文</button>
                  <button type="button" class="${feedMode === "ans" ? "active" : ""}" data-mode="ans">仅回答</button>
                </div>
              </div>
            </div>
          </div>
          <i class="rule" aria-hidden="true"></i>
        </div>

        <div id="feed-list"></div>
        <div class="feed-more" id="feed-sentinel">
          <button type="button" id="btn-load-more">继续向下加载更多推荐 ↓</button>
        </div>
      </section>
    `;

    currentFeedPool = computeRecommendedFeed({ mode: feedMode, topic: activeTopic, jitter });
    renderedCount = 0;
    appendNextFeedBatch(14);
    setupInfiniteObserver();

    document.getElementById("btn-refresh-feed")?.addEventListener("click", () => {
      // Mark currently visible top unclicked items with light impression decay so fresh cards surface
      currentFeedPool.slice(0, Math.min(renderedCount, 8)).forEach(e => {
        const id = ITEMS[e.idx].id;
        if (!profile.read[id] && !profile.liked[id]) {
          profile.impressions[id] = (profile.impressions[id] || 0) + 1;
        }
      });
      saveProfile();
      scrollTo({ top: 0, behavior: "smooth" });
      renderHomeView(0.11);
    });

    document.getElementById("btn-reset-profile")?.addEventListener("click", () => {
      if (confirm("确定清空本地阅读偏好画像吗？（收藏夹将保留）")) {
        const keptSaved = profile.saved;
        profile = defaultProfile();
        profile.saved = keptSaved;
        saveProfile();
        renderHomeView(0.04);
      }
    });

    main.querySelectorAll(".pill-btn[data-topic]").forEach(btn => {
      btn.addEventListener("click", () => {
        const t = btn.dataset.topic || "";
        activeTopic = t;
        if (t) {
          profile.tagWeights[t] = Number(((profile.tagWeights[t] || 0) + 0.6).toFixed(2));
          saveProfile();
        }
        renderHomeView(0.03);
      });
    });

    main.querySelectorAll(".mode-switch button[data-mode]").forEach(btn => {
      btn.addEventListener("click", () => {
        feedMode = btn.dataset.mode || "all";
        renderHomeView(0.03);
      });
    });

    document.getElementById("btn-load-more")?.addEventListener("click", () => appendNextFeedBatch(12));
  }

  function appendNextFeedBatch(batchSize = 12) {
    const listEl = document.getElementById("feed-list");
    const sentinel = document.getElementById("feed-sentinel");
    if (!listEl) return;

    const slice = currentFeedPool.slice(renderedCount, renderedCount + batchSize);
    if (!slice.length) {
      if (sentinel) sentinel.innerHTML = `<span>已刷完当前筛选下的全部 ${renderedCount} 篇内容</span>`;
      return;
    }

    listEl.insertAdjacentHTML("beforeend", slice.map(renderCardHTML).join(""));
    renderedCount += slice.length;

    if (renderedCount >= currentFeedPool.length && sentinel) {
      sentinel.innerHTML = `<span>已刷完当前筛选下的全部 ${renderedCount} 篇内容</span>`;
    }
  }

  function setupInfiniteObserver() {
    if (feedObserver) feedObserver.disconnect();
    const sentinel = document.getElementById("feed-sentinel");
    if (!sentinel) return;
    feedObserver = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        appendNextFeedBatch(12);
      }
    }, { rootMargin: "420px 0px" });
    feedObserver.observe(sentinel);
  }

  // Fetch single post JSON (from repo ./posts/<id>.json)
  async function fetchPost(id) {
    if (POST_CACHE.has(id)) return POST_CACHE.get(id);
    const res = await fetch(`posts/${encodeURIComponent(id)}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    POST_CACHE.set(id, data);
    return data;
  }

  // Settle active reading dwell telemetry when navigating away from a post
  function flushActiveDwell() {
    if (!activeDwell || activeDwell.settled) return;
    activeDwell.settled = true;
    const elapsedSec = (Date.now() - activeDwell.startTime) / 1000;
    const prev = profile.read[activeDwell.id] || { count: 1, dwell: 0, ts: Date.now() };
    prev.dwell = Math.round((prev.dwell || 0) + elapsedSec);
    profile.read[activeDwell.id] = prev;

    if (elapsedSec >= 22 || activeDwell.maxScroll >= 0.55) {
      const bonus = activeDwell.maxScroll >= 0.82 ? 2.4 : 1.4;
      recordSignal(activeDwell.id, bonus, "dwell");
    } else if (elapsedSec < 4 && activeDwell.maxScroll < 0.12) {
      recordSignal(activeDwell.id, -0.3, "bounce");
    } else {
      saveProfile();
    }
    activeDwell = null;
  }

  // Render Full-Text Article / Zhihu Answer Reader + 3-Slot Recs + Infinite Continuation Feed
  async function renderPostView(id) {
    setActiveNav("");
    const main = document.getElementById("main");
    main.innerHTML = `<p class="empty">正在打开文章存档 #${esc(id)}…</p>`;
    scrollTo({ top: 0, behavior: "instant" });

    try {
      const post = await fetchPost(id);
      const idx = ID_TO_IDX.get(String(id));
      document.title = `${post.title} — 雨夹雪 · 译文漫游`;

      // Log initial open + start dwell tracker
      recordSignal(id, 0.9, "read");
      activeDwell = { id: String(id), idx, startTime: Date.now(), maxScroll: 0, settled: false };

      const isLiked = Boolean(profile.liked[post.id]);
      const isSaved = Boolean(profile.saved[post.id]);
      const isAnswer = post.kind === "answer";

      const tocItemsHTML = (post.toc || [])
        .map(h => `<li class="level-${h.level}"><a href="#${esc(h.id)}" data-toc-target="${esc(h.id)}">${esc(h.title)}</a></li>`)
        .join("");

      const origSourcesHTML = (post.original_source_urls || [])
        .map(u => `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(u)} ↗</a>`)
        .join("<br>");

      const renderRecSlotItems = (arr) => {
        if (!arr || !arr.length) return `<li class="rec-item"><span class="rec-item-meta">暂无直接关联条目</span></li>`;
        return arr.map(r => `
          <li class="rec-item">
            <a href="#/post/${esc(r.id)}" data-rec-click="${esc(r.id)}">
              <div class="rec-item-title">${esc(r.title)}</div>
              <div class="rec-item-meta">
                <span class="rec-item-reason">${esc(r.reason)}</span>
                <span>▲ ${r.votes} 赞 · ${r.minutes} 分钟</span>
              </div>
            </a>
          </li>
        `).join("");
      };

      main.innerHTML = `
        <article class="post-article">
          <header class="article-header">
            <p class="article-meta">
              <button type="button" class="back-btn" id="btn-back-feed">← 返回推荐流</button>
              <span class="sep" aria-hidden="true">/</span>
              <time datetime="${esc(post.date)}">${esc(post.date)}</time>
              <a class="tag" href="#/column/${encodeURIComponent(post.column)}">${esc(post.column)}</a>
              <a class="tag" href="#/author/${encodeURIComponent(post.author)}">${esc(post.author)}</a>
              <span>约 ${post.minutes} 分钟（${post.chars.toLocaleString()} 字）</span>
              <span>▲ ${post.votes} 赞同</span>
            </p>
            <h1>${esc(post.title)}</h1>
            ${!isAnswer && post.summary ? `
              <details class="summary-fold">
                <summary>核心论点精校导读（${post.summary.length} 字）</summary>
                <div class="summary-fold-body">${esc(post.summary)}</div>
              </details>
            ` : ""}
            ${isAnswer && post.question_detail_html ? `
              <details class="summary-fold">
                <summary>查看知乎原问题描述</summary>
                <div class="summary-fold-body prose">${post.question_detail_html}</div>
              </details>
            ` : ""}
            ${(post.keywords && post.keywords.length) ? `
              <p class="tags">
                ${post.keywords.map(k => `<a href="#/tag/${encodeURIComponent(k)}">#${esc(k)}</a>`).join("")}
              </p>
            ` : ""}
            <i class="rule" aria-hidden="true"></i>
          </header>

          ${tocItemsHTML ? `
            <details class="toc-mobile">
              <summary>目录（${post.toc.length} 节）</summary>
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

            <aside class="side-rail">
              <div class="rail-actions">
                <span class="rail-actions-title">阅读反馈 · 实时调优推荐</span>
                <div class="rail-btn-row">
                  <button type="button" class="act-chip js-like ${isLiked ? "active" : ""}" data-id="${esc(post.id)}">${isLiked ? "★ 已感兴趣" : "👍 感兴趣"}</button>
                  <button type="button" class="act-chip js-save ${isSaved ? "saved" : ""}" data-id="${esc(post.id)}">${isSaved ? "🔖 已收藏" : "稍后再读"}</button>
                </div>
                <div class="rail-ext">
                  <span>发布者：${esc(post.publisher)}</span>
                  <a href="${esc(post.zhihu_url)}" target="_blank" rel="noopener noreferrer">知乎原帖 ↗</a>
                </div>
              </div>
              ${tocItemsHTML ? `
                <nav class="toc" aria-label="目录">
                  <p>目录</p>
                  <ol id="desktop-toc-list">${tocItemsHTML}</ol>
                </nav>
              ` : ""}
            </aside>

            <footer class="colophon">
              <dl>
                <div>
                  <dt>原作者</dt>
                  <dd>${(post.original_authors || [post.author]).map(a => `<a href="#/author/${encodeURIComponent(a)}">${esc(a)}</a>`).join(" · ")}</dd>
                </div>
                ${origSourcesHTML ? `<div><dt>英文原链</dt><dd>${origSourcesHTML}</dd></div>` : ""}
                <div>
                  <dt>知乎存档</dt>
                  <dd>${esc(post.publisher)} · <a href="${esc(post.zhihu_url)}" target="_blank" rel="noopener noreferrer">${esc(post.zhihu_url)} ↗</a></dd>
                </div>
                <div>
                  <dt>权利声明</dt>
                  <dd>摘要与索引源自 <a href="https://github.com/L-M-Sherlock/tm-index" target="_blank" rel="noopener">tm-index</a>，正文备份源自 <a href="https://github.com/L-M-Sherlock/ZhiHuArchive" target="_blank" rel="noopener">ZhiHuArchive</a>（整理结构遵循 CC0 1.0；原著与译文著作权归原作者及译者所有）。</dd>
                </div>
              </dl>
            </footer>
          </div>

          <section class="rec-section" aria-label="文末延伸推荐">
            <i class="rule top" aria-hidden="true"></i>
            <div class="rec-section-head">
              <h3>顺藤摸瓜与跨界延伸</h3>
              <span>基于双向引用图谱 + 回答共被引 + 128维语义向量生成</span>
            </div>
            <div class="rec-grid-3">
              <div class="rec-col">
                <span class="rec-col-kicker">01 / CITATION CHAIN</span>
                <div class="rec-col-title">${isAnswer ? "本回答直接引用的译文" : "顺藤摸瓜 · 上下文与互引"}</div>
                <ul class="rec-list">${renderRecSlotItems(post.rec_chain)}</ul>
              </div>
              <div class="rec-col bridge">
                <span class="rec-col-kicker">02 / CROSS-AUTHOR BRIDGE</span>
                <div class="rec-col-title">跨界共鸣 · 异作者破茧视角</div>
                <ul class="rec-list">${renderRecSlotItems(post.rec_bridge)}</ul>
              </div>
              <div class="rec-col qa">
                <span class="rec-col-kicker">03 / ZHIHU Q&amp;A PRACTICE</span>
                <div class="rec-col-title">${isAnswer ? "相似痛点的高赞问答" : "知乎实战 · 引用本文的回答"}</div>
                <ul class="rec-list">${renderRecSlotItems(post.rec_answers)}</ul>
              </div>
            </div>
            <i class="rule" aria-hidden="true"></i>
          </section>
        </article>

        <div class="cont-header">CONTINUE READING / 根据本文与你的画像继续往下刷</div>
        <div id="feed-list"></div>
        <div class="feed-more" id="feed-sentinel">
          <button type="button" id="btn-load-more">加载更多延伸推荐 ↓</button>
        </div>
      `;

      document.getElementById("btn-back-feed")?.addEventListener("click", () => {
        if (history.length > 1) history.back();
        else location.hash = "#/";
      });

      // Clicking TOC smooth-scrolls inside page without breaking hash router
      main.querySelectorAll("[data-toc-target]").forEach(a => {
        a.addEventListener("click", e => {
          e.preventDefault();
          const el = document.getElementById(a.dataset.tocTarget);
          if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });

      // Footnote anchor smooth-scroll inside prose
      main.querySelectorAll('.prose a[href^="#ref-"], .prose a[href^="#cite-"]').forEach(a => {
        a.addEventListener("click", e => {
          e.preventDefault();
          const targetId = a.getAttribute("href").slice(1);
          const el = document.getElementById(targetId);
          if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
        });
      });

      // Clicking bottom 3-slot recs gives extra positive signal
      main.querySelectorAll("[data-rec-click]").forEach(a => {
        a.addEventListener("click", () => {
          recordSignal(a.dataset.recClick, 1.5, "chain");
        });
      });

      // Populate infinite continuation stream anchored on current post!
      currentFeedPool = computeRecommendedFeed({ anchorIdx: idx ?? null, mode: "all", jitter: 0.03 });
      renderedCount = 0;
      appendNextFeedBatch(10);
      setupInfiniteObserver();
      document.getElementById("btn-load-more")?.addEventListener("click", () => appendNextFeedBatch(10));
      setupTocScrollSpy();
    } catch (err) {
      main.innerHTML = `<section class="page-head"><p class="eyebrow">ERROR</p><h1>无法加载内容 #${esc(id)}</h1><p class="dek">${esc(err.message)}</p><p style="margin-top:18px"><a class="textbtn" href="#/">← 返回推荐首页</a></p></section>`;
    }
  }

  // Render Hot Leaderboard
  function renderHotView(sortKey = "votes") {
    setActiveNav("hot");
    document.title = "高赞与枢纽榜 — 雨夹雪 · 译文漫游";
    const sorted = ITEMS.map((it, idx) => ({ it, idx })).sort((a, b) => {
      if (sortKey === "citations") return b.it.deg - a.it.deg || b.it.v - a.it.v;
      if (sortKey === "answers") return (b.it.k === "ans" ? b.it.v : -1) - (a.it.k === "ans" ? a.it.v : -1);
      return b.it.v - a.it.v;
    });

    currentFeedPool = sorted.map(({ it, idx }, rank) => ({
      idx,
      reason: `TOP #${rank + 1} · ${sortKey === "citations" ? `被引用 ${it.deg} 次` : `${it.v} 赞同`}`,
      badgeClass: it.k === "ans" ? "qa" : "",
    }));
    renderedCount = 0;

    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <header class="page-head">
          <p class="eyebrow">LEADERBOARD / 全站高赞与知识枢纽榜</p>
          <h1>高赞热榜</h1>
          <p class="dek">按历史知乎赞同数或知识图谱被引用次数排序的镇站之作。</p>
          <div class="topic-pills" style="margin-top:18px">
            <button type="button" class="pill-btn ${sortKey === "votes" ? "active" : ""}" data-sort="votes">按知乎赞同数</button>
            <button type="button" class="pill-btn ${sortKey === "citations" ? "active" : ""}" data-sort="citations">按图谱被引用数（枢纽度）</button>
            <button type="button" class="pill-btn ${sortKey === "answers" ? "active" : ""}" data-sort="answers">高赞知乎回答榜</button>
          </div>
          <i class="rule" aria-hidden="true"></i>
        </header>
        <div id="feed-list"></div>
        <div class="feed-more" id="feed-sentinel"><button type="button" id="btn-load-more">加载更多 ↓</button></div>
      </section>
    `;
    appendNextFeedBatch(15);
    setupInfiniteObserver();
    main.querySelectorAll("[data-sort]").forEach(b => b.addEventListener("click", () => renderHotView(b.dataset.sort)));
    document.getElementById("btn-load-more")?.addEventListener("click", () => appendNextFeedBatch(15));
  }

  // Render Columns, Canonical Keywords & Original Authors Directory
  function renderTopicsView() {
    setActiveNav("topics");
    document.title = "专栏、标签与原作者 — 雨夹雪 · 译文漫游";
    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <header class="page-head">
          <p class="eyebrow">TAXONOMY / 31 个专栏 · 4,356 个规范词 · 172 位原作者</p>
          <h1>专栏与知识地图</h1>
          <p class="dek">点击任意专栏、标签或原作者即可进入对应专题流，并自动计入你的推荐偏好。</p>
          <i class="rule" aria-hidden="true"></i>
        </header>
        <div class="terms">
          <h2>专栏目录（Columns）</h2>
          <ul class="term-list">
            ${FEED_DATA.columns.map(c => `<li><a href="#/column/${encodeURIComponent(c.name)}">${esc(c.name)}<span>${c.count}</span></a></li>`).join("")}
          </ul>
          <h2>核心原作者（Original Authors）</h2>
          <ul class="term-list">
            ${FEED_DATA.top_authors.slice(0, 24).map(a => `<li><a href="#/author/${encodeURIComponent(a.name)}">${esc(a.name)}<span>${a.count}</span></a></li>`).join("")}
          </ul>
          <h2>高频规范标签（Canonical Keywords）</h2>
          <ul class="term-cloud">
            ${FEED_DATA.top_keywords.map(k => `<li><a href="#/tag/${encodeURIComponent(k.name)}">#${esc(k.name)}<span>${k.count}</span></a></li>`).join("")}
          </ul>
        </div>
      </section>
    `;
  }

  // Render Filtered Term View (tag / column / author)
  function renderTermView(kind, value) {
    setActiveNav("topics");
    const labelMap = { tag: "标签", column: "专栏", author: "作者" };
    document.title = `${value} — ${labelMap[kind] || "专题"}`;

    const matched = [];
    ITEMS.forEach((it, idx) => {
      const ok = kind === "tag"
        ? (it.kw || []).includes(value)
        : (kind === "column" ? it.c === value : it.a === value || it.pub === value);
      if (ok) {
        matched.push({ idx, reason: `${labelMap[kind]} · ${it.v} 赞`, badgeClass: it.k === "ans" ? "qa" : "" });
      }
    });
    matched.sort((a, b) => ITEMS[b.idx].v - ITEMS[a.idx].v);
    currentFeedPool = matched;
    renderedCount = 0;

    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <header class="page-head">
          <p class="eyebrow"><a href="#/topics">← 返回专栏与标签</a> / ${esc(labelMap[kind])}</p>
          <h1>${kind === "tag" ? "#" : ""}${esc(value)}</h1>
          <p class="dek">共收录 ${matched.length} 篇相关译文与回答，按赞同数排列。</p>
          <i class="rule" aria-hidden="true"></i>
        </header>
        <div id="feed-list"></div>
        <div class="feed-more" id="feed-sentinel"><button type="button" id="btn-load-more">加载更多 ↓</button></div>
      </section>
    `;
    appendNextFeedBatch(16);
    setupInfiniteObserver();
    document.getElementById("btn-load-more")?.addEventListener("click", () => appendNextFeedBatch(16));
  }

  // Render Saved Bookmarks, Liked & User Interest Profile Inspector
  function renderSavedView(subtab = "saved") {
    setActiveNav("saved");
    document.title = "收藏夹与阅读画像 — 雨夹雪 · 译文漫游";

    const idMap = subtab === "liked" ? profile.liked : (subtab === "history" ? profile.read : profile.saved);
    const entries = Object.entries(idMap)
      .map(([id, val]) => ({ id, ts: typeof val === "object" ? val.ts : val, idx: ID_TO_IDX.get(id) }))
      .filter(x => x.idx !== undefined)
      .sort((a, b) => (b.ts || 0) - (a.ts || 0));

    currentFeedPool = entries.map(e => ({
      idx: e.idx,
      reason: subtab === "saved" ? "🔖 稍后再读" : (subtab === "liked" ? "★ 已标感兴趣" : "🕒 阅读足迹"),
      badgeClass: "",
    }));
    renderedCount = 0;

    const topTags = Object.entries(profile.tagWeights).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]).slice(0, 12);
    const topAuthors = Object.entries(profile.authorWeights).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]).slice(0, 8);

    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <header class="page-head">
          <p class="eyebrow">LOCAL PROFILE / 100% 存储于本地浏览器</p>
          <h1>收藏与个人阅读画像</h1>
          <p class="dek">所有阅读行为、停留时长与偏好权重均保存在本地 localStorage 中，实时驱动混合推荐引擎。</p>
          ${topTags.length ? `
            <p class="tags" style="margin-top:14px">
              ${topTags.map(([k, w]) => `<a href="#/tag/${encodeURIComponent(k)}">#${esc(k)} (${w.toFixed(1)})</a>`).join("")}
              ${topAuthors.map(([a, w]) => `<a href="#/author/${encodeURIComponent(a)}">@${esc(a)} (${w.toFixed(1)})</a>`).join("")}
            </p>
          ` : ""}
          <div class="topic-pills" style="margin-top:18px">
            <button type="button" class="pill-btn ${subtab === "saved" ? "active" : ""}" data-sub="saved">稍后再读 (${Object.keys(profile.saved).length})</button>
            <button type="button" class="pill-btn ${subtab === "liked" ? "active" : ""}" data-sub="liked">感兴趣 (${Object.keys(profile.liked).length})</button>
            <button type="button" class="pill-btn ${subtab === "history" ? "active" : ""}" data-sub="history">阅读历史 (${Object.keys(profile.read).length})</button>
          </div>
          <i class="rule" aria-hidden="true"></i>
        </header>
        <div id="feed-list">${entries.length === 0 ? `<p class="empty">当前列表为空，去推荐流刷几篇感兴趣的内容吧。</p>` : ""}</div>
        <div class="feed-more" id="feed-sentinel" ${entries.length === 0 ? "hidden" : ""}><button type="button" id="btn-load-more">加载更多 ↓</button></div>
      </section>
    `;
    if (entries.length) {
      appendNextFeedBatch(15);
      setupInfiniteObserver();
    }
    main.querySelectorAll("[data-sub]").forEach(b => b.addEventListener("click", () => renderSavedView(b.dataset.sub)));
  }

  // Render Instant Full-Index Search
  function renderSearchView(initialQuery = "") {
    setActiveNav("search");
    document.title = "搜索 — 雨夹雪 · 译文漫游";
    const main = document.getElementById("main");
    main.innerHTML = `
      <section class="writing">
        <header class="page-head">
          <p class="eyebrow">SEARCH / 3,580 篇译文与回答即时检索</p>
          <h1>搜索</h1>
          <form class="search-form" role="search" id="search-form">
            <input id="search-input" type="search" placeholder="输入关键词、标题、原作者或专栏（支持空格多词组合，如：间隔重复 Matuschak）" autocomplete="off" spellcheck="false" value="${esc(initialQuery)}">
          </form>
          <p class="search-status" id="search-status" aria-live="polite"></p>
          <i class="rule" aria-hidden="true"></i>
        </header>
        <div id="feed-list"></div>
        <div class="feed-more" id="feed-sentinel" hidden><button type="button" id="btn-load-more">加载更多结果 ↓</button></div>
      </section>
    `;

    const input = document.getElementById("search-input");
    const status = document.getElementById("search-status");
    const listEl = document.getElementById("feed-list");
    const sentinel = document.getElementById("feed-sentinel");

    const runQuery = (q) => {
      const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
      listEl.innerHTML = "";
      if (!terms.length) {
        status.textContent = "输入任意词条即时筛选 2,187 篇译文与 1,393 篇知乎回答。";
        sentinel.hidden = true;
        return;
      }
      const hits = [];
      for (let i = 0; i < ITEMS.length; i++) {
        const it = ITEMS[i];
        const hay = `${it.t} ${it.a} ${it.c} ${(it.kw || []).join(" ")} ${it.s} ${it.id}`.toLowerCase();
        if (terms.every(t => hay.includes(t))) {
          const titleBoost = terms.every(t => it.t.toLowerCase().includes(t)) ? 2.0 : 0;
          hits.push({ idx: i, score: titleBoost + it.q, reason: `匹配检索 · ${it.v} 赞`, badgeClass: it.k === "ans" ? "qa" : "" });
        }
      }
      hits.sort((a, b) => b.score - a.score);
      status.textContent = `找到 ${hits.length} 条匹配结果`;
      currentFeedPool = hits;
      renderedCount = 0;
      sentinel.hidden = hits.length <= 15;
      appendNextFeedBatch(15);
      setupInfiniteObserver();
    };

    document.getElementById("search-form")?.addEventListener("submit", e => e.preventDefault());
    input?.addEventListener("input", () => runQuery(input.value));
    input?.focus();
    runQuery(initialQuery);
  }

  function setActiveNav(key) {
    document.querySelectorAll("#main-nav a[data-nav]").forEach(a => {
      if (a.dataset.nav === key) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }

  // Scroll-spy for Reader TOC & Global Progress Bar
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

  // Hover Preview Card (.cite-card) for BOTH footnotes [n] AND internal tm-index links!
  function clearHoverCard() {
    hoverCard?.remove();
    hoverCard = null;
  }

  function handlePointerOver(e) {
    if (e.pointerType && e.pointerType !== "mouse") return;
    const internalLink = e.target.closest?.("a[data-tm-id]");
    const citeLink = e.target.closest?.(".cite a");

    if (internalLink) {
      const tmId = internalLink.dataset.tmId;
      const idx = ID_TO_IDX.get(tmId);
      if (idx === undefined || hoverCard?._for === internalLink) return;
      clearHoverCard();
      const it = ITEMS[idx];
      hoverCard = document.createElement("div");
      hoverCard.className = "cite-card";
      hoverCard._for = internalLink;
      hoverCard.innerHTML = `
        <div class="cite-card-head">
          <span>${it.k === "ans" ? "知乎回答" : "站内译文"} · ${esc(it.a)}</span>
          <span>▲ ${it.v} 赞 · ${it.m} 分钟</span>
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
      hoverCard.innerHTML = `<div class="cite-card-head"><span>参考注释 ${esc(citeLink.textContent)}</span></div>`;
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

  // Router
  function route() {
    clearHoverCard();
    flushActiveDwell();
    window._tocSpyHandler = null;
    if (feedObserver) feedObserver.disconnect();

    const hash = location.hash || "#/";
    if (hash === "#/" || hash === "#") {
      renderHomeView(0.04);
    } else if (hash.startsWith("#/post/")) {
      const id = decodeURIComponent(hash.slice("#/post/".length));
      renderPostView(id);
    } else if (hash === "#/hot") {
      renderHotView("votes");
    } else if (hash === "#/topics") {
      renderTopicsView();
    } else if (hash.startsWith("#/tag/")) {
      const tag = decodeURIComponent(hash.slice("#/tag/".length));
      profile.tagWeights[tag] = Number(((profile.tagWeights[tag] || 0) + 0.5).toFixed(2));
      saveProfile();
      renderTermView("tag", tag);
    } else if (hash.startsWith("#/column/")) {
      const col = decodeURIComponent(hash.slice("#/column/".length));
      renderTermView("column", col);
    } else if (hash.startsWith("#/author/")) {
      const au = decodeURIComponent(hash.slice("#/author/".length));
      renderTermView("author", au);
    } else if (hash === "#/saved") {
      renderSavedView("saved");
    } else if (hash === "#/search") {
      renderSearchView("");
    } else {
      renderHomeView(0.04);
    }
  }

  // Global Event Delegation (Theme, Like, Save, Dismiss, Scroll, Hover)
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

    document.addEventListener("click", e => {
      const likeBtn = e.target.closest?.(".js-like");
      if (likeBtn) {
        const id = likeBtn.dataset.id;
        if (profile.liked[id]) {
          delete profile.liked[id];
          recordSignal(id, -1.5, "unlike");
          likeBtn.classList.remove("active");
          likeBtn.textContent = "👍 感兴趣";
        } else {
          profile.liked[id] = Date.now();
          recordSignal(id, 3.2, "like");
          likeBtn.classList.add("active");
          likeBtn.textContent = "★ 已感兴趣";
        }
        saveProfile();
        return;
      }

      const saveBtn = e.target.closest?.(".js-save");
      if (saveBtn) {
        const id = saveBtn.dataset.id;
        if (profile.saved[id]) {
          delete profile.saved[id];
          saveBtn.classList.remove("saved");
          saveBtn.textContent = "稍后再读";
        } else {
          profile.saved[id] = Date.now();
          recordSignal(id, 2.0, "save");
          saveBtn.classList.add("saved");
          saveBtn.textContent = "🔖 已收藏";
        }
        saveProfile();
        return;
      }

      const dismissBtn = e.target.closest?.(".js-dismiss");
      if (dismissBtn) {
        const id = dismissBtn.dataset.id;
        profile.dismissed[id] = Date.now();
        recordSignal(id, -2.8, "dismiss");
        const row = dismissBtn.closest(".post-row");
        if (row) row.remove();
        return;
      }
    });

    addEventListener("scroll", () => {
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - innerHeight);
      const ratio = Math.min(1, Math.max(0, scrollY / maxScroll));
      const bar = document.getElementById("scroll-progress");
      if (bar) bar.style.transform = `scaleX(${ratio.toFixed(4)})`;

      const topBtn = document.getElementById("to-top-btn");
      if (topBtn) topBtn.classList.toggle("show", scrollY > 360);

      if (activeDwell && ratio > activeDwell.maxScroll) {
        activeDwell.maxScroll = ratio;
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
    addEventListener("beforeunload", flushActiveDwell);
  }

  // Boot Application
  async function boot() {
    bindGlobalEvents();
    updateSavedBadge();
    try {
      const indexRes = await fetch("data/feed-index.json");
      if (!indexRes.ok) throw new Error(`加载索引失败: HTTP ${indexRes.status}`);
      FEED_DATA = await indexRes.json();
      ITEMS = FEED_DATA.items;
      ITEMS.forEach((it, idx) => ID_TO_IDX.set(String(it.id), idx));

      // Render first screen immediately before waiting on binary vectors!
      route();

      // Load 128-d Int8 embeddings & graph neighbors asynchronously in background
      Promise.all([
        fetch("data/embeddings.bin").then(r => r.ok ? r.arrayBuffer() : null),
        fetch("data/neighbors.json").then(r => r.ok ? r.json() : null),
      ]).then(([buf, nbs]) => {
        if (buf) EMB_I8 = new Int8Array(buf);
        if (nbs) NEIGHBORS = nbs;
      }).catch(() => {});
    } catch (err) {
      const bootEl = document.getElementById("boot-loading");
      if (bootEl) bootEl.textContent = `初始化失败：${err.message}`;
    }
  }

  boot();
})();
