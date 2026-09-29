const state = {
  data: null,
  view: "groups",
  setupById: new Map(),
  groupById: new Map(),
  sourceByKey: new Map(),
  groupSetupSort: new Map(),
  openKind: null,
  openId: null,
  sourceDialogLimit: 40,
  pages: { groups: 1, setups: 1, sources: 1 },
  pageSizes: { groups: 24, setups: 20, sources: 20 },
  episodePayloads: new Map(),
  scenePageCache: new Map(),
  dataPromises: new Map(),
  sceneCatalogLoaded: false,
  setupsLoaded: false,
  sourcesLoaded: false,
  historyReady: false,
};

const setupCopy = {
  "SETUP-001": ["서랍 작업대 / 여러 소형 부품", "작업대와 주요 물체 구성이 유지되며 물체 위치와 instruction 변화는 분할하지 않았습니다."],
  "SETUP-002": ["서랍 선반 / 파란 직육면체 블록", "파란 직육면체 작업 물체가 유지되는 구간입니다."],
  "SETUP-003": ["서랍 선반 / 회색 원통형 물체", "회색 원통형 물체가 추가되어 앞 구간과 분리했습니다."],
  "SETUP-004": ["분홍 그릇 / 노란 컵", "분홍 그릇과 노란 컵이 반복 확인되는 구간입니다."],
  "SETUP-005": ["분홍 그릇 / 회색 컵", "분홍 그릇은 같지만 노란 컵이 회색 컵으로 교체됩니다."],
  "SETUP-006": ["분홍 그릇 / 주황 컵 전환", "회색 컵 구간과 검은 그릇 구간 사이의 전환 Episode입니다."],
  "SETUP-007": ["검은 그릇 / 주황 컵 / 초록 토큰", "검은 그릇과 주황 컵 구성이 유지됩니다."],
  "SETUP-008": ["색깔 컵 4개", "빨강, 파랑, 노랑, 초록 컵 구성이 유지됩니다."],
  "SETUP-009": ["색깔 컵 4개 / 검은 그릇", "컵 4개에 검은 그릇이 추가되어 별도 asset 구성으로 분리했습니다."],
  "SETUP-010": ["회색 머그 / 색깔 공과 컵", "회색 머그와 작은 색깔 물체 세트로 전체 구성이 바뀝니다."],
  "SETUP-011": ["나무 슬롯 보드 / 주황 원판 3개", "나무 보드와 주황 원판 구성이 유지됩니다."],
  "SETUP-012": ["검은 마우스 / 작은 버튼 부품", "병이 없는 시간 구간은 같은 마우스 object 구성입니다."],
  "SETUP-013": ["검은 마우스 / 베이지색 병", "베이지색 squeeze bottle이 존재해 별도 Setup으로 분리했습니다."],
  "SETUP-014": ["검은 그릇 / 작은 장난감 묶음", "봉제 장난감과 작은 장난감, 검은 그릇 구성이 유지됩니다."],
  "SETUP-015": ["검은 그릇 / 색깔 식기", "장난감이 제거되고 색깔 식기로 교체됩니다."],
  "SETUP-016": ["세탁실 / 상단 건조기 2번", "추가 이동 물체 없이 같은 건조기와 조작 패널을 사용합니다."],
  "SETUP-017": ["세탁실 / 세제와 옷 묶음", "같은 세제통, 병, 봉투와 옷 묶음이 이동·가림·투입됩니다."],
  "SETUP-018": ["세탁실 / 베이지색 천 투입", "이 Episode만 베이지색 천 가방을 사용합니다."],
  "SETUP-019": ["세탁실 / 세탁기 문", "별도 이동 물체 없이 같은 세탁기 문을 사용합니다."],
  "SETUP-020": ["세탁실 / 상단 건조기", "버튼과 문은 같은 건조기 asset의 구성 요소입니다."],
  "SETUP-021": ["세탁실 / 8번 세탁기 물품", "8번 세탁기와 세제통, 흰색·보라색 옷 구성이 유지됩니다."],
  "SETUP-022": ["세탁실 / 9번 세탁기 물품", "작업 위치가 9번 세탁기로 이동하고 물체 구성이 바뀝니다."],
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");
const number = (value) => new Intl.NumberFormat("ko-KR").format(value || 0);
const setupTitle = (setup) => setupCopy[setup.setup_id]?.[0] || setup.title;
const setupReason = (setup) => setupCopy[setup.setup_id]?.[1] || setup.reason;
const familyName = (value) => String(value || "other").replaceAll("__", " + ").replaceAll("_", " ");
const complexityScore = (episode) => Number(episode?.complexity?.score || 0);
const compareId = (left, right) => String(left).localeCompare(String(right), "en", { numeric: true });

function indexGroup(group) {
  group._search = JSON.stringify(group).toLowerCase();
  state.groupById.set(group.scene_id, group);
  return group;
}

function indexSetup(setup) {
  setup._search = JSON.stringify({ ...setup, episodes: undefined, display_title: setupTitle(setup) }).toLowerCase();
  state.setupById.set(setup.setup_id, setup);
  return setup;
}

function indexSource(source) {
  source._search = JSON.stringify({ ...source, episodes: undefined }).toLowerCase();
  state.sourceByKey.set(source.raw_scene_key, source);
  return source;
}

async function fetchDataPayload(url) {
  if (!state.dataPromises.has(url)) {
    state.dataPromises.set(url, fetch(url, { cache: "force-cache" }).then(async (response) => {
      if (!response.ok) throw new Error(`Data HTTP ${response.status}: ${url}`);
      return response.json();
    }).catch((error) => {
      state.dataPromises.delete(url);
      throw error;
    }));
  }
  return state.dataPromises.get(url);
}

function scenePaginationConfig() {
  return state.data?.data_loading?.scene_pagination || null;
}

async function ensureScenePage(pageNumber) {
  if (state.scenePageCache.has(pageNumber)) return state.scenePageCache.get(pageNumber);
  const config = scenePaginationConfig();
  if (!config) return state.data.scene_groups;
  const url = config.page_urls?.[pageNumber - 1];
  if (!url) throw new Error(`Scene page ${pageNumber} URL이 없습니다.`);
  const payload = await fetchDataPayload(url);
  if (!Array.isArray(payload.scene_groups)) throw new Error("Scene page data is malformed");
  const groups = payload.scene_groups.map(indexGroup);
  state.scenePageCache.set(pageNumber, groups);
  return groups;
}

async function ensureSceneCatalog() {
  if (state.sceneCatalogLoaded) return state.data.scene_groups;
  const url = scenePaginationConfig()?.catalog_url;
  if (!url) {
    state.sceneCatalogLoaded = true;
    return state.data.scene_groups;
  }
  const payload = await fetchDataPayload(url);
  if (!Array.isArray(payload.scene_groups)) throw new Error("Scene catalog is malformed");
  state.data.scene_groups = payload.scene_groups.map(indexGroup);
  state.sceneCatalogLoaded = true;
  return state.data.scene_groups;
}

async function ensureSetups() {
  if (state.setupsLoaded) return state.data.setups;
  const url = state.data?.data_loading?.setup_index_url;
  if (!url) {
    state.setupsLoaded = true;
    return state.data.setups;
  }
  const payload = await fetchDataPayload(url);
  if (!Array.isArray(payload.setups)) throw new Error("Setup index is malformed");
  state.data.setups = payload.setups.map(indexSetup);
  state.setupsLoaded = true;
  return state.data.setups;
}

async function ensureSources() {
  if (state.sourcesLoaded) return state.data.source_scenes;
  const url = state.data?.data_loading?.source_index_url;
  if (!url) {
    state.sourcesLoaded = true;
    return state.data.source_scenes;
  }
  const payload = await fetchDataPayload(url);
  if (!Array.isArray(payload.source_scenes)) throw new Error("Source Scene index is malformed");
  state.data.source_scenes = payload.source_scenes.map(indexSource);
  state.sourcesLoaded = true;
  return state.data.source_scenes;
}

async function ensureSceneSetups(group) {
  const present = group.setup_ids.every((setupId) => state.setupById.has(setupId));
  if (present) return group.setup_ids.map((setupId) => state.setupById.get(setupId));
  if (!group.setups_url) {
    await ensureSetups();
    return group.setup_ids.map((setupId) => state.setupById.get(setupId));
  }
  const payload = await fetchDataPayload(group.setups_url);
  if (payload.scene_id !== group.scene_id || !Array.isArray(payload.setups)) {
    throw new Error(`${group.scene_id} Setup data is malformed`);
  }
  const setups = payload.setups.map(indexSetup);
  if (setups.length !== Number(group.setup_count) || !group.setup_ids.every((id) => state.setupById.has(id))) {
    throw new Error(`${group.scene_id} Setup count mismatch`);
  }
  return setups;
}

async function fetchEpisodePayload(url) {
  if (!state.episodePayloads.has(url)) {
    state.episodePayloads.set(url, fetch(url, { cache: "force-cache" }).then(async (response) => {
      if (!response.ok) throw new Error(`Episode data HTTP ${response.status}`);
      const payload = await response.json();
      if (!Array.isArray(payload.episodes)) throw new Error("Episode data is malformed");
      return payload.episodes;
    }));
  }
  return state.episodePayloads.get(url);
}

async function ensureEpisodes(owner) {
  if (Array.isArray(owner.episodes)) return owner.episodes;
  if (owner._episodesPromise) return owner._episodesPromise;
  const urls = owner.episode_urls || (owner.episodes_url ? [owner.episodes_url] : []);
  owner._episodesPromise = Promise.all(urls.map(fetchEpisodePayload)).then((groups) => {
    const episodes = groups.flat().filter((episode) => (
      !owner.setup_id || episode.setup_id === owner.setup_id
    ));
    episodes.sort((left, right) => {
      const sequenceKey = owner.setup_id ? "sequence_in_setup" : "sequence_in_source_scene";
      return Number(left[sequenceKey] || 0) - Number(right[sequenceKey] || 0)
        || Number(left.dataset_row_index || 0) - Number(right.dataset_row_index || 0);
    });
    if (episodes.length !== Number(owner.episode_count || 0)) {
      throw new Error(
        `${owner.setup_id || owner.raw_scene_key} expected ${owner.episode_count} Episodes, loaded ${episodes.length}`
      );
    }
    owner.episodes = episodes;
    return episodes;
  }).catch((error) => {
    owner._episodesPromise = null;
    throw error;
  });
  return owner._episodesPromise;
}

function paginationMarkup(kind, totalItems) {
  const pageSize = state.pageSizes[kind];
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const current = Math.min(Math.max(1, state.pages[kind]), totalPages);
  state.pages[kind] = current;
  if (totalPages <= 1) return "";
  const visible = [...new Set([1, current - 1, current, current + 1, totalPages])]
    .filter((page) => page >= 1 && page <= totalPages)
    .sort((left, right) => left - right);
  let previous = 0;
  const numbered = visible.map((page) => {
    const gap = previous && page - previous > 1
      ? '<span class="pagination-gap" aria-hidden="true">...</span>'
      : "";
    previous = page;
    return `${gap}<button class="pagination-page${page === current ? " is-current" : ""}" type="button" data-page-kind="${kind}" data-page="${page}"${page === current ? ' aria-current="page"' : ""}>${page}</button>`;
  }).join("");
  return `<nav class="pagination" aria-label="${kind} 페이지">
    <button class="pagination-arrow" type="button" data-page-kind="${kind}" data-page="${current - 1}" aria-label="이전 페이지"${current === 1 ? " disabled" : ""}>&lsaquo;</button>
    <div class="pagination-pages">${numbered}</div>
    <span class="pagination-status">${number(current)} / ${number(totalPages)}</span>
    <button class="pagination-arrow" type="button" data-page-kind="${kind}" data-page="${current + 1}" aria-label="다음 페이지"${current === totalPages ? " disabled" : ""}>&rsaquo;</button>
  </nav>`;
}

function paginated(items, kind) {
  const pageSize = state.pageSizes[kind];
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  state.pages[kind] = Math.min(Math.max(1, state.pages[kind]), totalPages);
  const start = (state.pages[kind] - 1) * pageSize;
  return { rows: items.slice(start, start + pageSize), start };
}

function pageRangeLabel(totalItems, kind) {
  if (!totalItems) return "0";
  const start = (state.pages[kind] - 1) * state.pageSizes[kind] + 1;
  const end = Math.min(totalItems, start + state.pageSizes[kind] - 1);
  return `${number(start)}-${number(end)} / ${number(totalItems)}`;
}

function setView(view, preserveDialog = false) {
  state.view = view;
  $$(".tab").forEach((button) => button.classList.toggle("is-active", button.dataset.view === view));
  $$(".view").forEach((section) => {
    section.hidden = section.id !== `${view}View`;
    section.classList.toggle("is-active", section.id === `${view}View`);
  });
  if (!preserveDialog && $("#detailDialog").open) {
    $("#detailDialog").close();
    state.openKind = null;
    state.openId = null;
  }
}

function returnLabelFor(navigation) {
  if (navigation.openKind === "group") return "Scene 상세로 돌아가기";
  if (navigation.openKind === "source") return "Source Scene 상세로 돌아가기";
  if (navigation.view === "setups") return "Setup 목록으로 돌아가기";
  if (navigation.view === "sources") return "Source Scene 목록으로 돌아가기";
  if (navigation.view === "method") return "분류 기준으로 돌아가기";
  return "Scene 목록으로 돌아가기";
}

function expandedSetupIds() {
  return [...new Set(
    $$('.toggle-setup-episodes[aria-expanded="true"]')
      .map((button) => button.dataset.setupId || button.closest(".setup-expand-host")?.dataset.setupHostId)
      .filter(Boolean)
  )];
}

function navigationSnapshot() {
  const dialog = $("#detailDialog");
  const stored = window.history.state?.droidReview ? window.history.state : {};
  return {
    droidReview: true,
    view: state.view,
    openKind: dialog.open ? state.openKind : null,
    openId: dialog.open ? state.openId : null,
    focusSetupId: null,
    expandedSetupIds: expandedSetupIds(),
    scrollY: window.scrollY,
    dialogScrollTop: $("#detailDialogBody").scrollTop,
    hasReviewBack: Boolean(stored.hasReviewBack),
    backLabel: stored.backLabel || null,
  };
}

function navigationUrl(navigation) {
  const url = new URL(window.location.href);
  ["view", "scene", "source", "setup"].forEach((key) => url.searchParams.delete(key));
  if (navigation.view && navigation.view !== "groups") url.searchParams.set("view", navigation.view);
  if (navigation.openKind === "group" && navigation.openId) url.searchParams.set("scene", navigation.openId);
  if (navigation.openKind === "source" && navigation.openId) url.searchParams.set("source", navigation.openId);
  if (navigation.focusSetupId) url.searchParams.set("setup", navigation.focusSetupId);
  return `${url.pathname}${url.search}${url.hash}`;
}

function restoreExpandedSetups(ids, root = document) {
  const wanted = new Set(ids || []);
  if (!wanted.size) return;
  $$("[data-setup-host-id]", root).forEach((host) => {
    if (!wanted.has(host.dataset.setupHostId)) return;
    const button = $(".toggle-setup-episodes", host);
    if (button && button.getAttribute("aria-expanded") !== "true") toggleSetupPanel(button, true);
  });
}

async function prepareNavigation(target) {
  if (target.view === "setups" || target.focusSetupId) await ensureSetups();
  if (target.view === "sources" || target.openKind === "source") await ensureSources();
  if (target.openKind === "group") {
    if (!state.groupById.has(target.openId)) await ensureSceneCatalog();
    const group = state.groupById.get(target.openId);
    if (group) await ensureSceneSetups(group);
  }
}

async function applyNavigation(navigation) {
  const target = {
    droidReview: true,
    view: navigation.view || "groups",
    openKind: navigation.openKind || null,
    openId: navigation.openId || null,
    focusSetupId: navigation.focusSetupId || null,
    expandedSetupIds: navigation.expandedSetupIds || [],
    scrollY: Number(navigation.scrollY || 0),
    dialogScrollTop: Number(navigation.dialogScrollTop || 0),
    hasReviewBack: Boolean(navigation.hasReviewBack),
    backLabel: navigation.backLabel || null,
  };
  const dialog = $("#detailDialog");
  setView(target.view, true);
  state.openKind = null;
  state.openId = null;
  dialog.classList.remove("source-dialog-mode");

  if (target.view === "setups" && !state.setupsLoaded) {
    $("#setupList").innerHTML = '<div class="loading">Setup 인덱스를 불러오는 중...</div>';
  }
  if (target.view === "sources" && !state.sourcesLoaded) {
    $("#sourceSceneGrid").innerHTML = '<div class="loading">Source Scene 인덱스를 불러오는 중...</div>';
  }
  if (target.openKind && !dialog.open) {
    $("#detailDialogBody").innerHTML = '<div class="loading">상세 데이터를 불러오는 중...</div>';
    dialog.showModal();
  }

  await prepareNavigation(target);

  if (target.view === "groups") await renderGroups();
  else if (target.view === "setups") await renderSetups(target.focusSetupId);
  else if (target.view === "sources") await renderSources();

  if (target.openKind === "group" && state.groupById.has(target.openId)) {
    state.openKind = "group";
    state.openId = target.openId;
    renderGroupDialog(state.groupById.get(target.openId));
    if (!dialog.open) dialog.showModal();
  } else if (target.openKind === "source" && state.sourceByKey.has(target.openId)) {
    state.openKind = "source";
    state.openId = target.openId;
    state.sourceDialogLimit = 40;
    dialog.classList.add("source-dialog-mode");
    renderSourceDialog(state.sourceByKey.get(target.openId));
    if (!dialog.open) dialog.showModal();
  } else {
    if (dialog.open) dialog.close();
    $("#dialogBackButton").hidden = true;
  }

  if (dialog.open) {
    $("#dialogBackButton").hidden = false;
    $("#dialogBackLabel").textContent = target.backLabel || (
      target.openKind === "source" ? "Source Scene 목록으로 돌아가기" : "Scene 목록으로 돌아가기"
    );
  }

  window.requestAnimationFrame(() => {
    const root = dialog.open ? $("#detailDialogBody") : document;
    restoreExpandedSetups(target.expandedSetupIds, root);
    $("#detailDialogBody").scrollTop = target.dialogScrollTop;
    if (target.focusSetupId) {
      const card = $$("[data-setup-card-id]", $("#setupList"))
        .find((item) => item.dataset.setupCardId === target.focusSetupId);
      if (card) {
        const button = $(".toggle-setup-episodes", card);
        if (button) toggleSetupPanel(button, true);
        card.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
    }
    window.scrollTo({ top: target.scrollY, behavior: "auto" });
  });
}

async function navigateTo(navigation, { replace = false } = {}) {
  const current = navigationSnapshot();
  const target = {
    droidReview: true,
    view: navigation.view || "groups",
    openKind: navigation.openKind || null,
    openId: navigation.openId || null,
    focusSetupId: navigation.focusSetupId || null,
    expandedSetupIds: navigation.expandedSetupIds || [],
    scrollY: Number(navigation.scrollY || 0),
    dialogScrollTop: Number(navigation.dialogScrollTop || 0),
    hasReviewBack: replace ? Boolean(current.hasReviewBack) : true,
    backLabel: navigation.backLabel || returnLabelFor(current),
  };
  if (state.historyReady) {
    window.history.replaceState(current, "", navigationUrl(current));
    if (replace) window.history.replaceState(target, "", navigationUrl(target));
    else window.history.pushState(target, "", navigationUrl(target));
  }
  await applyNavigation(target);
}

function leaveDetail() {
  if (window.history.state?.droidReview && window.history.state.hasReviewBack) {
    window.history.back();
    return;
  }
  const fallbackView = state.openKind === "source" ? "sources" : "groups";
  navigateTo({ view: fallbackView, backLabel: null }, { replace: true });
}

function objectTags(objects) {
  if (!objects?.length) return '<span class="empty-value">object label 없음</span>';
  return objects.map((item) => `<span class="object-tag">${esc(item)}</span>`).join("");
}

function targetAssetPanel(owner, variant) {
  if (!Array.isArray(owner?.target_assets)) return "";
  const assets = owner.target_assets;
  const isScene = variant.startsWith("scene");
  const scopeCount = Number(
    owner.target_asset_scope_count
    || state.data?.target_asset_inventory?.summary?.scene_count
    || state.data?.summary?.target_asset_labeled_scene_count
    || 0
  );
  const title = isScene
    ? `TOP ${number(scopeCount)} #${number(owner.target_asset_rank)} · SCENE TARGET ASSET SET`
    : "SETUP TARGET ASSETS";
  const tags = assets.length
    ? assets.map((asset) => `<span class="target-asset-tag" data-asset-id="${esc(asset.asset_id)}" title="${esc(asset.label_en)}"><b>${esc(asset.label_ko)}</b><small>${esc(asset.label_en)}</small></span>`).join("")
    : '<span class="target-asset-empty">이동형 target asset 없음</span>';
  const note = owner.review_note
    ? `<p class="target-asset-review-note">${esc(owner.review_note)}</p>`
    : "";
  return `<section class="target-asset-panel target-asset-panel--${esc(variant)}" data-target-asset-count="${assets.length}">
    <header><span>${esc(title)}</span><strong><b>${number(assets.length)}</b> 종류</strong></header>
    <div class="target-asset-tags">${tags}</div>
    ${note}
  </section>`;
}

function setupObjectInventory(setup, variant) {
  if (!Array.isArray(setup.target_assets)) {
    return `<div class="object-list">${objectTags(setup.objects)}</div>`;
  }
  return `${targetAssetPanel(setup, variant)}
    <details class="source-object-details"><summary>검토 당시 보인 전체 물체 라벨</summary><div class="object-list">${objectTags(setup.source_object_labels || setup.objects)}</div></details>`;
}

function sourceBadge(success) {
  return `<span class="source-badge ${success ? "success" : "failure"}">source ${success ? "success" : "failure"}</span>`;
}

function reviewBadge(setup) {
  if (setup.confidence < 0.75) return '<span class="audit-badge transition">전환 구간 / 재검수</span>';
  if (setup.confidence < 0.85) return '<span class="audit-badge provisional">시간순 검수 / 보통</span>';
  return '<span class="audit-badge reviewed">시간순 검수 완료</span>';
}

function complexityBadge(episode, compact = false) {
  const components = episode?.complexity?.components || {};
  const specificity = Number(components.task_specificity_multiplier ?? 1);
  const discount = specificity < 1 ? ` · freeform ×${specificity.toFixed(2)}` : "";
  const detail = `조작 ${components.manipulation_phases ?? 0} · object ${components.interacted_object_load ?? 0} · 경로 ${components.path_constraints ?? 0} · ${Number(components.duration_sec || 0).toFixed(1)}초${discount}`;
  return `<span class="complexity-badge${compact ? " compact" : ""}${specificity < 1 ? " specificity-discounted" : ""}" title="${esc(detail)}"><b>${complexityScore(episode).toFixed(1)}</b>${compact ? "" : " complexity"}${specificity < 1 && !compact ? ` · freeform ×${specificity.toFixed(2)}` : ""}</span>`;
}

function objectCircleMarkup(episode) {
  const annotations = episode?.object_annotations || [];
  if (!annotations.length) return "";
  const circles = annotations.map((item, index) => {
    const x = Math.max(0, Math.min(1, Number(item.x || 0))) * 100;
    const y = Math.max(0, Math.min(1, Number(item.y || 0))) * 100;
    const diameter = Math.max(3, Math.min(60, Number(item.r || 0) * 200));
    const label = item.label || `object ${index + 1}`;
    return `<span class="object-circle" style="--object-x:${x}%;--object-y:${y}%;--object-diameter:${diameter}%" data-index="${index + 1}" data-label="${esc(label)}" title="${esc(label)}" aria-label="${esc(label)}"></span>`;
  }).join("");
  return `${circles}<span class="object-circle-count">주요 object ${annotations.length}</span>`;
}

function screenshotMarkup(episode, className = "") {
  if (!episode?.screenshot) return `<div class="screenshot-missing ${esc(className)}"><span>Screenshot 없음</span></div>`;
  const annotated = episode.object_annotations?.length ? " is-object-annotated" : "";
  return `<span class="screenshot-frame${annotated}"><img class="${esc(className)}" src="${esc(episode.screenshot)}" alt="${esc(episode.uuid)} Episode screenshot" loading="lazy">${objectCircleMarkup(episode)}</span>`;
}

function videoMarkup(episode, className = "") {
  const url = episode?.preferred_video_url || episode?.ext1_url;
  if (!url) return '<div class="video-missing"><span>Episode 영상 없음</span></div>';
  const poster = episode.screenshot ? ` poster="${esc(episode.screenshot)}"` : "";
  return `<video class="${esc(className)}" controls playsinline preload="none"${poster}><source src="${esc(url)}" type="video/mp4"></video>`;
}

function mediaPair(episode, className = "") {
  const camera = episode?.preferred_video_camera || "ext1";
  return `<div class="media-pair ${esc(className)}">
    <figure><figcaption>Episode screenshot</figcaption>${screenshotMarkup(episode, "episode-screenshot")}</figure>
    <figure><figcaption>Episode video · ${esc(camera)}</figcaption>${videoMarkup(episode, "episode-video")}</figure>
  </div>`;
}

function isNoSuccessFallback(owner) {
  return owner?.representative_selection_status === "fallback_no_success_episode";
}

function provenanceMarkup(owner, label = null) {
  const provenance = owner.representative_provenance;
  if (!provenance) return "";
  const resolvedLabel = label || (
    isNoSuccessFallback(owner)
      ? "대표가 아닌 검수용 Failure Episode"
      : "대표로 선택된 Success Episode"
  );
  return `<div class="representative-origin${isNoSuccessFallback(owner) ? " is-fallback" : ""}">
    <span>${esc(resolvedLabel)}</span>
    <strong>${esc(provenance.setup_id)}의 Episode #${number(provenance.setup_episode_number)}</strong>
    <small>Source Scene Episode #${number(provenance.source_scene_episode_number)} · ${esc(provenance.episode_uuid)}</small>
  </div>`;
}

function instructionMarkup(episode, label = "이 대표 Episode의 Instruction") {
  return `<div class="instruction-box"><span>${esc(label)}</span><p>${esc(episode?.task || "Instruction metadata 없음")}</p></div>`;
}

function sourceSceneButton(rawSceneKey, compact = false) {
  return `<button class="source-scene-link open-source-scene${compact ? " compact" : ""}" type="button" data-source-key="${esc(rawSceneKey)}"><span>Source Scene</span><code>${esc(rawSceneKey)}</code></button>`;
}

function sceneButton(sceneId) {
  const group = state.groupById.get(sceneId);
  return `<button class="scene-reference open-group" type="button" data-group-id="${esc(sceneId)}">${esc(sceneId)} · ${esc(group?.title_ko || group?.title || "")}</button>`;
}

function videoLinks(episode) {
  return `<div class="video-links" aria-label="Episode 영상 다운로드">
    <a href="${esc(episode.ext1_url)}" target="_blank" rel="noreferrer">ext1</a>
    <a href="${esc(episode.ext2_url)}" target="_blank" rel="noreferrer">ext2</a>
    <a href="${esc(episode.wrist_url)}" target="_blank" rel="noreferrer">wrist</a>
  </div>`;
}

function representativePanel(owner, heading) {
  const rep = owner.representative;
  const fallback = isNoSuccessFallback(owner);
  const resolvedHeading = fallback
    ? "SUCCESS EPISODE 없음 · 대표 미선정"
    : heading;
  const fallbackNote = fallback
    ? '<div class="representative-fallback-note"><strong>대표 영상 아님</strong><span>이 묶음에는 성공 Episode가 없어 검수용 Failure 자료만 표시합니다.</span></div>'
    : "";
  return `<section class="representative-panel${fallback ? " is-no-success-fallback" : ""}" data-representative-status="${fallback ? "no-success-fallback" : "success"}">
    <div class="representative-panel-media">${mediaPair(rep, "representative-media")}</div>
    <div class="representative-panel-copy">
      <div class="section-heading-row"><p class="section-label">${esc(resolvedHeading)}</p>${complexityBadge(rep)}</div>
      ${fallbackNote}
      ${provenanceMarkup(owner)}
      <div class="rep-meta">${sourceBadge(rep.source_success)}<span class="family-badge">${esc(familyName(rep.task_family))}</span><span>${Number(rep.trajectory_duration_sec || 0).toFixed(2)}s</span></div>
      ${sourceSceneButton(rep.raw_scene_key, true)}
      ${instructionMarkup(rep, fallback ? "검수용 Failure Episode의 Instruction" : "이 대표 Episode의 Instruction")}
    </div>
  </section>`;
}

function episodeItem(episode) {
  const components = episode.complexity?.components || {};
  return `<article class="episode-item">
    <header class="episode-item-header">
      <div>
        <strong>${esc(episode.setup_id)} Episode #${number(episode.sequence_in_setup)}</strong>
        <span>Source Scene Episode #${number(episode.sequence_in_source_scene)}</span>
      </div>
      <div class="episode-header-badges">${complexityBadge(episode, true)}${sourceBadge(episode.source_success)}</div>
    </header>
    <div class="episode-item-body">
      ${mediaPair(episode, "episode-row-media")}
      <div class="episode-item-copy">
        ${sourceSceneButton(episode.raw_scene_key, true)}
        <dl class="episode-facts">
          <div><dt>UUID</dt><dd><code>${esc(episode.uuid)}</code></dd></div>
          <div><dt>Timestamp</dt><dd><code>${esc(episode.timestamp)}</code></dd></div>
          <div><dt>Task family</dt><dd>${esc(familyName(episode.task_family))}</dd></div>
          <div><dt>복잡도 구성</dt><dd>조작 ${esc(components.manipulation_phases ?? 0)} · object ${esc(components.interacted_object_load ?? 0)} · 경로 ${esc(components.path_constraints ?? 0)} · ${Number(components.duration_sec || 0).toFixed(1)}초${Number(components.task_specificity_multiplier ?? 1) < 1 ? ` · freeform ×${Number(components.task_specificity_multiplier).toFixed(2)}` : ""}</dd></div>
        </dl>
        ${instructionMarkup(episode, "이 Episode의 Instruction")}
        <div class="episode-downloads"><span>영상 다운로드</span>${videoLinks(episode)}</div>
      </div>
    </div>
  </article>`;
}

function episodeListMarkup(episodes, title, subtitle = "") {
  return `<section class="episode-list-section">
    <header class="episode-list-heading"><div><p class="section-label">EPISODES</p><h4>${esc(title)}</h4>${subtitle ? `<p>${esc(subtitle)}</p>` : ""}</div><strong>${number(episodes.length)}</strong></header>
    <div class="episode-list">${episodes.map(episodeItem).join("")}</div>
  </section>`;
}

function sourceKeys(setup) {
  return [...new Set(setup.members.map((member) => member.raw_scene_key))];
}

function setupSortValue(setup, mode) {
  if (mode === "episodes") return -setup.episode_count;
  if (mode === "complexity") return -complexityScore(setup.representative);
  return 0;
}

function sortSetups(setups, mode) {
  return [...setups].sort((left, right) => {
    if (mode === "complexity" || mode === "episodes") {
      return setupSortValue(left, mode) - setupSortValue(right, mode) || compareId(left.setup_id, right.setup_id);
    }
    if (mode === "source") {
      return compareId(sourceKeys(left)[0], sourceKeys(right)[0]) || compareId(left.setup_id, right.setup_id);
    }
    if (mode === "scene") {
      return compareId(left.scene_id, right.scene_id) || compareId(left.setup_id, right.setup_id);
    }
    return compareId(left.setup_id, right.setup_id);
  });
}

function setupSourceButtons(setup) {
  return sourceKeys(setup).map((key) => sourceSceneButton(key, true)).join("");
}

function setupInlineButton(setup) {
  return `<div class="episode-toggle-bar">
    <span class="episode-toggle-count"><b>${number(setup.episode_count)}</b><span>Episodes</span></span>
    <button class="episode-chevron-button toggle-setup-episodes" type="button" data-setup-id="${esc(setup.setup_id)}" aria-expanded="false" aria-label="${esc(setup.setup_id)} Episode 목록 펼치기" title="Episode 목록 펼치기"><span class="episode-chevron-icon" aria-hidden="true"></span></button>
  </div>`;
}

function setupInlineRow(setup) {
  return `<article class="scene-setup-row setup-expand-host" data-setup-host-id="${esc(setup.setup_id)}">
    <div class="scene-setup-media">${mediaPair(setup.representative, "setup-row-representative")}</div>
    <div class="scene-setup-content">
      <header><div><p class="setup-id">${esc(setup.setup_id)}</p><h4>${esc(setupTitle(setup))}</h4></div><div class="setup-row-summary"><strong>${number(setup.episode_count)}</strong><span>episodes</span></div></header>
      ${provenanceMarkup(setup)}
      <div class="setup-representative-meta">${complexityBadge(setup.representative)}<span class="family-badge">${esc(familyName(setup.representative.task_family))}</span></div>
      <div class="setup-source-provenance"><span>포함 Source Scene</span><div>${setupSourceButtons(setup)}</div></div>
      ${instructionMarkup(setup.representative)}
      ${setupObjectInventory(setup, "setup-inline")}
      ${setupInlineButton(setup)}
    </div>
    <div class="inline-episode-panel" hidden></div>
  </article>`;
}

function setupCard(setup) {
  return `<article class="setup-card setup-expand-host" data-setup-card-id="${esc(setup.setup_id)}" data-setup-host-id="${esc(setup.setup_id)}">
    <header class="setup-header-v9">
      <div><p class="setup-id">${esc(setup.setup_id)}</p><h3>${esc(setupTitle(setup))}</h3><button class="scene-reference open-group" type="button" data-group-id="${esc(setup.scene_id)}">${esc(setup.scene_id)} · ${esc(setup.scene_title)}</button></div>
      <dl><div><dt>Source</dt><dd>${number(setup.source_scene_count)}</dd></div><div><dt>Episode</dt><dd>${number(setup.episode_count)}</dd></div><div><dt>Complexity</dt><dd>${complexityScore(setup.representative).toFixed(1)}</dd></div></dl>
    </header>
    <section class="setup-card-body">
      ${representativePanel(setup, "SETUP REPRESENTATIVE · 대표로 선택된 1개 Episode")}
      <div class="setup-card-metadata">
        ${setupObjectInventory(setup, "setup-card")}
        <div><p class="section-label">Setup 분할 근거</p><p>${esc(setupReason(setup))}</p></div>
        <div class="setup-badges">${reviewBadge(setup)}<span class="confidence-badge">confidence ${Math.round(setup.confidence * 100)}%</span></div>
        <div class="setup-source-provenance"><span>포함 Source Scene</span><div>${setupSourceButtons(setup)}</div></div>
      </div>
    </section>
    ${setupInlineButton(setup)}
    <div class="inline-episode-panel" hidden></div>
  </article>`;
}

function setSetupToggleState(button, status) {
  const open = status !== "closed";
  button.setAttribute("aria-expanded", String(open));
  button.classList.toggle("is-loading", status === "loading");
  const action = open ? "접기" : "펼치기";
  button.setAttribute("aria-label", `${button.dataset.setupId} Episode 목록 ${action}`);
  button.title = `Episode 목록 ${action}`;
}

async function toggleSetupPanel(button, forceOpen = false) {
  const host = button.closest(".setup-expand-host");
  if (!host) return;
  const setup = state.setupById.get(button.dataset.setupId || host.dataset.setupHostId);
  const panel = $(":scope > .inline-episode-panel", host);
  if (!setup || !panel) return;
  if (forceOpen && !panel.hidden) return;
  const shouldOpen = forceOpen || panel.hidden;
  if (shouldOpen) {
    panel.hidden = false;
    host.classList.add("is-expanded");
    setSetupToggleState(button, "loading");
    panel.innerHTML = '<div class="loading inline-loading">Episode 목록을 불러오는 중...</div>';
    try {
      const episodes = await ensureEpisodes(setup);
      if (panel.hidden) return;
      panel.innerHTML = episodeListMarkup(
        episodes,
        `${setup.setup_id}의 모든 Episode`,
        "각 행의 screenshot, ext1 영상, 해당 Episode instruction과 Source Scene provenance"
      );
      setSetupToggleState(button, "open");
    } catch (error) {
      if (panel.hidden) return;
      panel.innerHTML = `<div class="empty lazy-load-error"><strong>Episode 목록을 불러오지 못했습니다.</strong><span>${esc(error.message)}</span></div>`;
      setSetupToggleState(button, "open");
    }
  } else {
    $$("video", panel).forEach((video) => video.pause());
    panel.hidden = true;
    panel.innerHTML = "";
    host.classList.remove("is-expanded");
    setSetupToggleState(button, "closed");
  }
}

function setupCollectionMarkup(setups, mode) {
  const rows = sortSetups(setups, mode);
  if (mode !== "source" && mode !== "scene") return rows.map(setupCard).join("");
  let previous = null;
  return rows.map((setup) => {
    const key = mode === "source" ? sourceKeys(setup)[0] : setup.scene_id;
    const heading = key !== previous
      ? `<div class="setup-sort-group"><span>${mode === "source" ? "SOURCE SCENE" : "PHYSICAL SCENE"}</span><code>${esc(key)}</code></div>`
      : "";
    previous = key;
    return heading + setupCard(setup);
  }).join("");
}

function sceneCard(group) {
  const rep = group.representative;
  return `<article class="scene-feed-card">
    <header class="scene-feed-header"><div><h3>${esc(group.scene_id)}</h3><p>${esc(group.title_ko || group.title)}</p></div><dl class="scene-feed-stats"><div><dt>Source</dt><dd>${number(group.source_scene_count)}</dd></div><div><dt>Setup</dt><dd>${number(group.setup_count)}</dd></div><div><dt>Episode</dt><dd>${number(group.episode_count)}</dd></div></dl></header>
    <button class="scene-cover open-group" type="button" data-group-id="${esc(group.scene_id)}" aria-label="${esc(group.scene_id)} 상세 열기">${screenshotMarkup(rep, "scene-cover-image")}<span class="scene-cover-score">complexity ${complexityScore(rep).toFixed(1)}</span></button>
    ${targetAssetPanel(group, "scene-card")}
    <div class="scene-card-representative">${provenanceMarkup(group)}${instructionMarkup(rep)}</div>
  </article>`;
}

function sortGroups(groups, mode) {
  return [...groups].sort((left, right) => {
    if (mode === "complexity") return complexityScore(right.representative) - complexityScore(left.representative) || compareId(left.scene_id, right.scene_id);
    if (mode === "episodes") return right.episode_count - left.episode_count || compareId(left.scene_id, right.scene_id);
    if (mode === "setups") return right.setup_count - left.setup_count || compareId(left.scene_id, right.scene_id);
    if (mode === "sources") return right.source_scene_count - left.source_scene_count || compareId(left.scene_id, right.scene_id);
    return compareId(left.scene_id, right.scene_id);
  });
}

async function renderGroups() {
  const token = Symbol("groups");
  state.groupRenderToken = token;
  const query = $("#groupSearch").value.trim().toLowerCase();
  const filter = $("#groupMergeFilter").value;
  const mode = $("#groupSort").value;
  const config = scenePaginationConfig();
  const defaultListing = Boolean(config) && !query && filter === "all" && mode === "complexity";

  if (defaultListing) {
    const total = Number(config.total);
    const totalPages = Math.max(1, Number(config.page_count));
    state.pages.groups = Math.min(Math.max(1, state.pages.groups), totalPages);
    const requestedPage = state.pages.groups;
    if (!state.scenePageCache.has(requestedPage)) {
      $("#groupSceneGrid").innerHTML = '<div class="loading">Scene 페이지를 불러오는 중...</div>';
    }
    const rows = await ensureScenePage(requestedPage);
    if (state.groupRenderToken !== token) return;
    $("#groupCount").textContent = `${pageRangeLabel(total, "groups")} · 전체 ${number(total)}개 Scene · 현재 페이지만 로드`;
    $("#groupSceneGrid").innerHTML = rows.length ? rows.map(sceneCard).join("") : '<div class="empty scene-feed-empty">조건에 맞는 Scene이 없습니다.</div>';
    $("#groupPagination").innerHTML = paginationMarkup("groups", total);
    return;
  }

  const catalog = await ensureSceneCatalog();
  if (state.groupRenderToken !== token) return;
  const groups = sortGroups(catalog.filter((group) => {
    const statusMatch = filter === "all" || (filter === "merged" && group.is_source_scene_merge) || (filter === "single" && !group.is_source_scene_merge);
    return statusMatch && (!query || group._search.includes(query));
  }), mode);
  const page = paginated(groups, "groups");
  $("#groupCount").textContent = `${pageRangeLabel(groups.length, "groups")} · 전체 ${number(catalog.length)}개 Scene`;
  $("#groupSceneGrid").innerHTML = page.rows.length ? page.rows.map(sceneCard).join("") : '<div class="empty scene-feed-empty">조건에 맞는 Scene이 없습니다.</div>';
  $("#groupPagination").innerHTML = paginationMarkup("groups", groups.length);
}

async function renderSetups(focusSetupId = null) {
  const token = Symbol("setups");
  state.setupRenderToken = token;
  if (!state.setupsLoaded) {
    $("#setupList").innerHTML = '<div class="loading">Setup 인덱스를 불러오는 중...</div>';
  }
  await ensureSetups();
  if (state.setupRenderToken !== token) return;
  const query = $("#setupSearch").value.trim().toLowerCase();
  const mode = $("#setupSort").value;
  const setups = sortSetups(
    state.data.setups.filter((setup) => !query || setup._search.includes(query)),
    mode
  );
  if (focusSetupId) {
    const index = setups.findIndex((setup) => setup.setup_id === focusSetupId);
    if (index >= 0) state.pages.setups = Math.floor(index / state.pageSizes.setups) + 1;
  }
  const page = paginated(setups, "setups");
  $("#setupCount").textContent = `${pageRangeLabel(setups.length, "setups")} · 전체 ${number(state.data.setups.length)}개 Setup · ${mode === "source" ? "Source Scene별 묶음" : "선택한 기준으로 정렬"}`;
  $("#setupList").innerHTML = page.rows.length ? setupCollectionMarkup(page.rows, mode) : '<div class="empty">조건에 맞는 Setup이 없습니다.</div>';
  $("#setupPagination").innerHTML = paginationMarkup("setups", setups.length);
}

function sourceTitle(source) {
  const meta = source.source_metadata;
  return `${meta.lab} · ${meta.building}`;
}

function sourceCard(source, index) {
  const rep = source.representative;
  const meta = source.source_metadata;
  return `<article class="source-ledger-row">
    <div class="source-ledger-index"><span>DROID RAW</span><strong>${String(index + 1).padStart(2, "0")}</strong></div>
    <button class="source-ledger-cover open-source-scene" type="button" data-source-key="${esc(source.raw_scene_key)}" aria-label="Source Scene 원본 Episode 열기">${screenshotMarkup(rep, "source-cover-image")}<span>대표 복잡도 ${complexityScore(rep).toFixed(1)}</span></button>
    <div class="source-ledger-body">
      <header><div><p class="section-label">ORIGINAL SOURCE UID</p><h3>${esc(sourceTitle(source))}</h3><code class="source-full-key">${esc(source.raw_scene_key)}</code></div><span class="split-badge ${source.is_split ? "split" : "single"}">${source.is_split ? `${source.setup_count} Setups로 분할` : "1 Setup"}</span></header>
      <div class="source-lineage-values"><span><b>LAB</b>${esc(meta.lab)}</span><span><b>BUILDING</b>${esc(meta.building)}</span><span><b>ROBOT</b>${esc(meta.robot_serial)}</span><span><b>UID</b>${esc(meta.source_scene_uid)}</span></div>
      <dl class="source-ledger-stats"><div><dt>연결 Physical Scene</dt><dd>${number(source.scene_ids.length)}</dd></div><div><dt>분류된 Setup</dt><dd>${number(source.setup_count)}</dd></div><div><dt>원본 Episode</dt><dd>${number(source.episode_count)}</dd></div></dl>
      <div class="source-ledger-connections"><span>RECLASSIFIED PHYSICAL SCENE</span><div>${source.scene_ids.map(sceneButton).join("")}</div></div>
      ${provenanceMarkup(source)}
      ${instructionMarkup(rep)}
      <button class="source-ledger-open open-source-scene" type="button" data-source-key="${esc(source.raw_scene_key)}"><span>원본 Episode 묶음 열기</span><b aria-hidden="true">→</b></button>
    </div>
  </article>`;
}

function sortSources(sources, mode) {
  return [...sources].sort((left, right) => {
    if (mode === "complexity") return complexityScore(right.representative) - complexityScore(left.representative) || compareId(left.raw_scene_key, right.raw_scene_key);
    if (mode === "episodes") return right.episode_count - left.episode_count || compareId(left.raw_scene_key, right.raw_scene_key);
    if (mode === "setups") return right.setup_count - left.setup_count || compareId(left.raw_scene_key, right.raw_scene_key);
    return compareId(left.raw_scene_key, right.raw_scene_key);
  });
}

async function renderSources() {
  const token = Symbol("sources");
  state.sourceRenderToken = token;
  if (!state.sourcesLoaded) {
    $("#sourceSceneGrid").innerHTML = '<div class="loading">Source Scene 인덱스를 불러오는 중...</div>';
  }
  await ensureSources();
  if (state.sourceRenderToken !== token) return;
  const query = $("#sourceSearch").value.trim().toLowerCase();
  const filter = $("#sourceSplitFilter").value;
  const mode = $("#sourceSort").value;
  const sources = sortSources(state.data.source_scenes.filter((source) => {
    const statusMatch = filter === "all" || (filter === "split" && source.is_split) || (filter === "single" && !source.is_split);
    return statusMatch && (!query || source._search.includes(query));
  }), mode);
  const page = paginated(sources, "sources");
  $("#sourceCount").textContent = `${pageRangeLabel(sources.length, "sources")} · 전체 ${number(state.data.source_scenes.length)}개 Source Scene · DROID provenance UID`;
  $("#sourceSceneGrid").innerHTML = page.rows.length ? page.rows.map((source, index) => sourceCard(source, page.start + index)).join("") : '<div class="empty scene-feed-empty">조건에 맞는 Source Scene이 없습니다.</div>';
  $("#sourcePagination").innerHTML = paginationMarkup("sources", sources.length);
}

function sceneSetupRows(group, mode) {
  const setups = sortSetups(
    group.setup_ids.map((id) => state.setupById.get(id)).filter(Boolean),
    mode
  );
  if (mode !== "source") return setups.map(setupInlineRow).join("");
  let previous = null;
  return setups.map((setup) => {
    const key = sourceKeys(setup)[0];
    const heading = key !== previous ? `<div class="scene-setup-group-heading"><span>Source Scene</span><code>${esc(key)}</code></div>` : "";
    previous = key;
    return heading + setupInlineRow(setup);
  }).join("");
}

function renderGroupDialog(group) {
  const mode = state.groupSetupSort.get(group.scene_id) || "complexity";
  $("#dialogEyebrow").textContent = "PHYSICAL SCENE";
  $("#dialogTitle").textContent = `${group.scene_id} · ${group.title_ko || group.title}`;
  $("#dialogKey").textContent = group.environment_signature;
  $("#detailDialogBody").innerHTML = `
    ${representativePanel(group, "SCENE REPRESENTATIVE · Setup 대표 중 선택된 1개 Episode")}
    ${targetAssetPanel(group, "scene")}
    <section class="scene-summary-band"><dl><div><dt>Source Scenes</dt><dd>${number(group.source_scene_count)}</dd></div><div><dt>Setups</dt><dd>${number(group.setup_count)}</dd></div><div><dt>Episodes</dt><dd>${number(group.episode_count)}</dd></div></dl><div><strong>고정 공간</strong><p>${esc(group.fixed_environment)}</p></div></section>
    <details class="scene-evidence-details"><summary>Scene 분류 근거와 고정 배경 metadata</summary><div class="dialog-evidence"><div><strong>같은 Scene으로 묶은 근거</strong><p>${esc(group.reason)}</p></div><div><strong>배경 Landmark</strong><div class="landmark-list">${group.background_landmarks.map((item) => `<span>${esc(item)}</span>`).join("")}</div></div></div></details>
    <section class="scene-setup-feed"><div class="scene-setup-feed-heading"><div><p class="section-label">SETUPS</p><h3>이 Scene의 모든 Setup</h3></div><label class="filter-field compact-filter"><span>Setup 정렬</span><select class="scene-setup-sort"><option value="complexity"${mode === "complexity" ? " selected" : ""}>대표 복잡도 높은 순</option><option value="source"${mode === "source" ? " selected" : ""}>Source Scene별</option><option value="episodes"${mode === "episodes" ? " selected" : ""}>Episode 많은 순</option><option value="id"${mode === "id" ? " selected" : ""}>Setup ID 순</option></select></label></div><div class="scene-setup-rows">${sceneSetupRows(group, mode)}</div></section>`;
}

async function openGroup(groupId) {
  return navigateTo({ view: "groups", openKind: "group", openId: groupId });
}

function sourceDistribution(source) {
  return source.setup_ids.map((setupId) => {
    const setup = state.setupById.get(setupId);
    const detail = setup
      ? `${number(setup.episode_count)} Episodes · ${esc(setupTitle(setup))}`
      : "클릭하면 Setup 상세 정보를 불러옵니다";
    return `<button class="setup-link open-setup-tab" type="button" data-setup-id="${esc(setupId)}"><b>${esc(setupId)}</b><span>${detail}</span></button>`;
  }).join("");
}

function renderSourceDialog(source) {
  const meta = source.source_metadata;
  $("#dialogEyebrow").textContent = "DROID ORIGINAL PROVENANCE";
  $("#dialogTitle").textContent = sourceTitle(source);
  $("#dialogKey").textContent = source.raw_scene_key;
  if (!Array.isArray(source.episodes)) {
    $("#detailDialogBody").innerHTML = '<div class="loading source-episode-loading">원본 Episode 목록을 불러오는 중...</div>';
    ensureEpisodes(source).then(() => {
      if (state.openKind === "source" && state.openId === source.raw_scene_key) {
        renderSourceDialog(source);
      }
    }).catch((error) => {
      if (state.openKind !== "source" || state.openId !== source.raw_scene_key) return;
      $("#detailDialogBody").innerHTML = `<div class="empty lazy-load-error"><strong>원본 Episode 목록을 불러오지 못했습니다.</strong><span>${esc(error.message)}</span><button class="small-button retry-source-episodes" type="button">재시도</button></div>`;
    });
    return;
  }
  const shown = Math.min(state.sourceDialogLimit, source.episodes.length);
  const episodes = source.episodes.slice(0, shown);
  $("#detailDialogBody").innerHTML = `
    <section class="source-origin-banner">
      <div class="source-origin-stamp"><span>DROID</span><b>RAW SOURCE</b></div>
      <dl><div><dt>LAB</dt><dd>${esc(meta.lab)}</dd></div><div><dt>BUILDING</dt><dd>${esc(meta.building)}</dd></div><div><dt>ROBOT</dt><dd>${esc(meta.robot_serial)}</dd></div><div><dt>SOURCE UID</dt><dd>${esc(meta.source_scene_uid)}</dd></div></dl>
    </section>
    ${representativePanel(source, "SOURCE SCENE REPRESENTATIVE · 원본 묶음에서 선택된 1개 Episode")}
    <section class="source-summary-band"><dl><div><dt>Physical Scenes</dt><dd>${number(source.scene_ids.length)}</dd></div><div><dt>Setups</dt><dd>${number(source.setup_count)}</dd></div><div><dt>Episodes</dt><dd>${number(source.episode_count)}</dd></div></dl><div class="source-parent-links"><span>재분류된 물리 Scene</span>${source.scene_ids.map(sceneButton).join("")}</div></section>
    <section class="source-setup-section"><div><p class="section-label">RECLASSIFIED SETUPS</p><h3>이 원본 Source Scene에서 나뉜 Setup</h3></div><div class="setup-distribution">${sourceDistribution(source)}</div></section>
    <section class="source-episode-shell"><header class="episode-list-heading"><div><p class="section-label">RAW SOURCE EPISODES</p><h4>이 Source UID의 모든 Episode</h4><p>${number(shown)} / ${number(source.episodes.length)} 표시</p></div><div class="source-list-actions">${shown < source.episodes.length ? '<button class="small-button source-more" type="button">40개 더</button><button class="small-button source-all" type="button">모두 표시</button>' : ""}</div></header><div class="episode-list">${episodes.map(episodeItem).join("")}</div></section>`;
}

async function openSourceScene(rawSceneKey) {
  if ($("#detailDialog").open && state.openKind === "source" && state.openId === rawSceneKey) return;
  return navigateTo({ view: "sources", openKind: "source", openId: rawSceneKey });
}

async function openSetupInTab(setupId) {
  return navigateTo({
    view: "setups",
    focusSetupId: setupId,
    expandedSetupIds: [setupId],
  });
}

function renderSummary() {
  const summary = state.data.summary;
  const items = [[summary.scene_count, "scenes"], [summary.source_scene_count, "source scenes"], [summary.setup_count, "setups"], [summary.episode_count, "episodes"], [summary.episode_screenshot_count, "screenshots"]];
  $("#summaryStats").innerHTML = items.map(([value, label]) => `<div class="summary-stat"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");
}

function renderMethod() {
  const { summary, method } = state.data;
  $("#methodContent").innerHTML = `
    <section class="method-section"><h3>용어와 계층</h3><dl class="term-list"><div><dt>Scene</dt><dd>Object를 제외하고 연구실, 건물, 로봇, 작업면, 고정 설비와 배경 geometry가 같은 물리 공간입니다.</dd></div><div><dt>Setup</dt><dd>한 Scene 안에서 Blender로 재현해야 할 주요 object 구성이 같은 Episode 묶음입니다.</dd></div><div><dt>Episode</dt><dd>영상 1개, 즉 DROID trajectory 1개입니다.</dd></div><div><dt>Source Scene</dt><dd>DROID의 <code>lab / building / robot_serial / scene_id</code> UID이며 출처 metadata로 보존합니다.</dd></div></dl></section>
    <section class="method-section"><h3>Scene과 Setup 분류</h3><ol class="decision-flow"><li><strong>Scene</strong><span>lab, building, robot, 고정 작업면과 배경 landmark를 비교합니다.</span></li><li><strong>Setup</strong><span>주요 object 종류, variant, 개수가 바뀌는 시간 경계에서 나눕니다.</span></li><li><strong>제외 정보</strong><span>Source Scene UID, object pose, instruction 문구와 source success는 Scene/Setup identity를 결정하지 않습니다.</span></li></ol></section>
    <section class="method-section"><h3>Episode 복잡도 v3</h3><p class="formula-line"><code>${esc(method.complexity.formula)}</code></p><dl class="term-list"><div><dt>O · 40%</dt><dd>상호작용 object 부하입니다. Blender asset 제작 목적에 맞춰 가장 큰 비중입니다.</dd></div><div><dt>P · 25%</dt><dd>조작 단계 수입니다.</dd></div><div><dt>R · 20%</dt><dd>container, stacking, 방향, 지정 위치, 연속 작업과 reset 경로 제약입니다.</dd></div><div><dt>duration · 15%</dt><dd>60초 상한 trajectory 시간이며 실행 경로 proxy입니다.</dd></div><div><dt>S · freeform 0.65</dt><dd>구체적인 object와 작업 경로가 없는 freeform metadata의 불확실성 감점입니다. 명시적 작업은 1.00입니다.</dd></div></dl><p class="method-note">instruction 길이와 source 성공·실패는 복잡도 점수에 사용하지 않습니다. “multiple steps”만 있는 freeform 문구는 임의로 3개 작업으로 가산하지 않습니다.</p></section>
    <section class="method-section"><h3>대표 영상 선정</h3><ol class="decision-flow"><li><strong>Setup 대표</strong><span>성공 Episode 중 screenshot과 영상이 있는 항목을 우선하고, 그 안에서 복잡도가 가장 높은 Episode를 선택합니다.</span></li><li><strong>Failure 제외</strong><span>성공 Episode가 하나라도 있으면 failure는 대표 후보에서 완전히 제외합니다.</span></li><li><strong>성공 Episode 없음</strong><span>대표를 미선정으로 표시하고 failure 한 건은 대표가 아닌 검수용 자료로만 명시합니다.</span></li><li><strong>Scene 대표</strong><span>하위 Setup의 성공 대표 중 복잡도가 가장 높은 Episode를 선택합니다.</span></li></ol></section>`;
  const stats = [["Physical Scenes", summary.scene_count], ["Source Scenes", summary.source_scene_count], ["Setups", summary.setup_count], ["Episodes", number(summary.episode_count)], ["Episode screenshots", `${summary.episode_screenshot_count}/${summary.episode_count}`], ["성공 Setup 대표", summary.setup_success_representative_count], ["대표 없음 Setup", summary.setup_no_success_episode_count], ["성공 Scene 대표", summary.scene_success_representative_count], ["대표 없음 Scene", summary.scene_no_success_episode_count], ["Source Scene 대표", summary.source_scene_representative_count], ["객체 원형 주석", summary.object_annotation_circle_count || 0], ["복잡도 버전", method.complexity.version]];
  $("#runStats").innerHTML = stats.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("");
}

async function changePage(kind, page) {
  const renderers = { groups: renderGroups, setups: renderSetups, sources: renderSources };
  if (!renderers[kind] || !Number.isFinite(page)) return;
  state.pages[kind] = Math.max(1, Math.trunc(page));
  await renderers[kind]();
  const viewIds = { groups: "groupsView", setups: "setupsView", sources: "sourcesView" };
  window.requestAnimationFrame(() => {
    $(`#${viewIds[kind]} .toolbar`)?.scrollIntoView({ behavior: "auto", block: "start" });
  });
}

function routeClick(event) {
  const pageButton = event.target.closest("[data-page-kind][data-page]");
  if (pageButton) return changePage(pageButton.dataset.pageKind, Number(pageButton.dataset.page));
  const sourceButton = event.target.closest(".open-source-scene");
  if (sourceButton) return openSourceScene(sourceButton.dataset.sourceKey);
  const groupButton = event.target.closest(".open-group");
  if (groupButton) return openGroup(groupButton.dataset.groupId);
  const setupTabButton = event.target.closest(".open-setup-tab");
  if (setupTabButton) return openSetupInTab(setupTabButton.dataset.setupId);
  const toggleButton = event.target.closest(".toggle-setup-episodes");
  if (toggleButton) return toggleSetupPanel(toggleButton);
  if (event.target.closest(".source-more") && state.openKind === "source") {
    state.sourceDialogLimit += 40;
    return renderSourceDialog(state.sourceByKey.get(state.openId));
  }
  if (event.target.closest(".source-all") && state.openKind === "source") {
    state.sourceDialogLimit = state.sourceByKey.get(state.openId).episodes.length;
    return renderSourceDialog(state.sourceByKey.get(state.openId));
  }
  if (event.target.closest(".retry-source-episodes") && state.openKind === "source") {
    return renderSourceDialog(state.sourceByKey.get(state.openId));
  }
}

function bindEvents() {
  $$(".tab").forEach((button) => button.addEventListener("click", async () => {
    if (state.view === button.dataset.view && !$("#detailDialog").open) return;
    await navigateTo({ view: button.dataset.view });
  }));
  [["#groupSearch", "groups", renderGroups], ["#groupMergeFilter", "groups", renderGroups], ["#groupSort", "groups", renderGroups], ["#setupSearch", "setups", renderSetups], ["#setupSort", "setups", renderSetups], ["#sourceSearch", "sources", renderSources], ["#sourceSplitFilter", "sources", renderSources], ["#sourceSort", "sources", renderSources]].forEach(([selector, kind, handler]) => $(selector).addEventListener("input", async () => {
    state.pages[kind] = 1;
    await handler();
  }));
  ["#groupSceneGrid", "#groupPagination", "#setupList", "#setupPagination", "#sourceSceneGrid", "#sourcePagination", "#detailDialogBody"].forEach((selector) => $(selector).addEventListener("click", routeClick));
  $("#detailDialogBody").addEventListener("change", (event) => {
    if (!event.target.matches(".scene-setup-sort") || state.openKind !== "group") return;
    state.groupSetupSort.set(state.openId, event.target.value);
    renderGroupDialog(state.groupById.get(state.openId));
  });
  $("#dialogBackButton").addEventListener("click", leaveDetail);
  $("#closeDetailDialog").addEventListener("click", leaveDetail);
  $("#detailDialog").addEventListener("click", (event) => {
    if (event.target === $("#detailDialog")) leaveDetail();
  });
  $("#detailDialog").addEventListener("cancel", (event) => {
    event.preventDefault();
    leaveDetail();
  });
  window.addEventListener("popstate", (event) => {
    if (event.state?.droidReview) {
      applyNavigation(event.state).catch((error) => {
        $("#groupSceneGrid").innerHTML = `<div class="empty">데이터를 불러오지 못했습니다: ${esc(error.message)}</div>`;
      });
    }
  });
}

async function init() {
  $("#groupSceneGrid").append($("#loadingTemplate").content.cloneNode(true));
  try {
    if (window.REVIEW_DATA) state.data = window.REVIEW_DATA;
    else {
      const response = await fetch("review_bootstrap.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.data = await response.json();
    }

    state.data.scene_groups = state.data.scene_groups || [];
    state.data.setups = state.data.setups || [];
    state.data.source_scenes = state.data.source_scenes || [];
    const sceneConfig = scenePaginationConfig();
    if (sceneConfig) {
      state.pageSizes.groups = Number(sceneConfig.page_size || state.pageSizes.groups);
      const firstPage = state.data.scene_groups.map(indexGroup);
      state.scenePageCache.set(1, firstPage);
      state.sceneCatalogLoaded = false;
    } else {
      state.data.scene_groups = state.data.scene_groups.map(indexGroup);
      state.sceneCatalogLoaded = true;
    }
    if (state.data.data_loading?.setup_index_url) {
      state.setupsLoaded = false;
    } else {
      state.data.setups = state.data.setups.map(indexSetup);
      state.setupsLoaded = true;
    }
    if (state.data.data_loading?.source_index_url) {
      state.sourcesLoaded = false;
    } else {
      state.data.source_scenes = state.data.source_scenes.map(indexSource);
      state.sourcesLoaded = true;
    }

    renderSummary();
    renderMethod();
    bindEvents();

    const query = new URLSearchParams(window.location.search);
    const requestedView = query.get("view");
    const initial = {
      droidReview: true,
      view: ["groups", "setups", "sources", "method"].includes(requestedView) ? requestedView : "groups",
      openKind: null,
      openId: null,
      focusSetupId: null,
      expandedSetupIds: [],
      scrollY: 0,
      dialogScrollTop: 0,
      hasReviewBack: false,
      backLabel: null,
    };
    if (query.get("scene")) {
      initial.view = "groups";
      initial.openKind = "group";
      initial.openId = query.get("scene");
    } else if (query.get("source")) {
      initial.view = "sources";
      initial.openKind = "source";
      initial.openId = query.get("source");
    } else if (query.get("setup")) {
      initial.view = "setups";
      initial.focusSetupId = query.get("setup");
      initial.expandedSetupIds = [query.get("setup")];
    }
    await applyNavigation(initial);
    state.historyReady = true;
    const firstEntry = { ...navigationSnapshot(), hasReviewBack: false, backLabel: null };
    window.history.replaceState(firstEntry, "", navigationUrl(firstEntry));
  } catch (error) {
    $("#groupSceneGrid").innerHTML = `<div class="empty">데이터를 불러오지 못했습니다: ${esc(error.message)}</div>`;
  }
}

init();
