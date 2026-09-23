import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const state = {
  scenes: [],
  scene: null,
  setup: null,
  asset: null,
  mediaMode: "video",
  spin: true,
};

const el = (id) => document.getElementById(id);
const sceneList = el("scene-list");
const setupList = el("setup-list");
const assetList = el("asset-list");
const formatter = new Intl.NumberFormat("ko-KR");

el("setup-video").addEventListener("loadedmetadata", () => {
  const status = el("video-status");
  status.textContent = "";
  status.hidden = true;
});

el("setup-video").addEventListener("error", () => {
  const status = el("video-status");
  status.textContent = "Video unavailable";
  status.hidden = false;
});

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function showError(message) {
  el("toast").textContent = message;
  el("toast").hidden = false;
  window.setTimeout(() => { el("toast").hidden = true; }, 5000);
}

function updateUrl() {
  const params = new URLSearchParams();
  if (state.scene) params.set("scene", state.scene.scene_id);
  if (state.setup) params.set("setup", state.setup.setup_id);
  if (state.asset) params.set("asset", state.asset.key);
  history.replaceState(null, "", `${location.pathname}?${params}`);
}

function renderSummary(summary) {
  const rows = [
    ["Scenes", summary.scene_count],
    ["Setups", summary.setup_count],
    ["3D Assets", summary.asset_count],
    ["Missing", summary.missing_label_count],
  ];
  el("summary").innerHTML = rows.map(([label, value]) => `
    <div class="summary-item"><span>${label}</span><strong>${typeof value === "number" ? formatter.format(value) : value}</strong></div>
  `).join("");
}

function renderScenes() {
  el("scene-result-count").textContent = `${formatter.format(state.scenes.length)} total`;
  sceneList.innerHTML = state.scenes.map((scene) => `
    <button class="scene-row ${state.scene?.scene_id === scene.scene_id ? "active" : ""}"
      type="button" data-scene="${escapeHtml(scene.scene_id)}">
      <span class="row-top">
        <span class="row-id">${escapeHtml(scene.scene_id)}</span>
        <span class="row-count">${formatter.format(scene.asset_count)} assets</span>
      </span>
      <span class="row-title">${escapeHtml(scene.title || scene.title_ko)}</span>
      <span class="row-meta">
        <span>${scene.setup_count} setups</span><span>${formatter.format(scene.episode_count)} episodes</span>
      </span>
    </button>
  `).join("");
}

function renderScene() {
  const scene = state.scene;
  el("scene-empty").hidden = true;
  el("scene-content").hidden = false;
  el("scene-id").textContent = scene.scene_id;
  el("scene-title").textContent = scene.title || scene.title_ko;
  el("scene-metrics").innerHTML = [
    `<span class="metric"><strong>${scene.asset_count}</strong>3D assets</span>`,
    `<span class="metric"><strong>${scene.setup_count}</strong>setups</span>`,
    `<span class="metric"><strong>${formatter.format(scene.episode_count)}</strong>episodes</span>`,
  ].join("");
  el("setup-count").textContent = `${scene.setups.length} total`;
  setupList.innerHTML = scene.setups.map((setup) => `
    <button class="setup-row ${state.setup?.setup_id === setup.setup_id ? "active" : ""} ${setup.asset_count ? "" : "no-assets"}"
      type="button" data-setup="${escapeHtml(setup.setup_id)}">
      <span class="row-top">
        <span class="row-id">${escapeHtml(setup.setup_id)}</span>
        <span class="row-count">${setup.asset_count ? `${setup.asset_count} assets` : "no output"}</span>
      </span>
      <span class="row-title">${escapeHtml(setup.title)}</span>
      <span class="row-meta"><span>${setup.episode_count} episodes</span><span>${escapeHtml(setup.audit_status || "")}</span>${setup.missing_labels.length ? `<span class="missing">${setup.missing_labels.length} missing</span>` : ""}</span>
    </button>
  `).join("");
  renderSetupMedia();
}

function renderSetupMedia() {
  if (!state.setup) return;
  const image = el("setup-image");
  const video = el("setup-video");
  const isVideo = state.mediaMode === "video" && Boolean(state.setup.video_url);
  image.hidden = isVideo;
  video.hidden = !isVideo;
  if (isVideo) {
    const changed = video.dataset.url !== state.setup.video_url;
    if (changed) {
      video.dataset.url = state.setup.video_url;
      video.src = state.setup.video_url;
      video.poster = state.setup.source_url;
      video.load();
      el("video-status").hidden = false;
      el("video-status").textContent = "Preparing complete trajectory";
    } else if (video.readyState >= 1) {
      el("video-status").hidden = true;
    }
  } else {
    video.pause();
    el("video-status").hidden = true;
    const requested = state.mediaMode === "detections" ? state.setup.detections_url : state.setup.source_url;
    const available = state.mediaMode === "detections" ? state.setup.has_detections : state.setup.has_source;
    image.src = available ? requested : state.setup.source_url;
  }
  const meta = el("trajectory-meta");
  meta.hidden = false;
  if (state.setup.video_url) {
    const status = el("trajectory-state");
    status.textContent = "SUCCESS";
    status.className = "success";
    const playback = Number(state.setup.video_duration_sec || 0);
    const trajectory = Number(state.setup.video_trajectory_duration_sec || 0);
    const speed = Number(state.setup.video_playback_rate || 1);
    el("trajectory-duration").textContent = playback
      ? `${playback.toFixed(1)} sec playback · ${formatter.format(state.setup.video_trajectory_length || 0)} steps · ${trajectory.toFixed(1)} sec trajectory${speed > 1.01 ? ` · ${speed.toFixed(1)}x` : ""}`
      : "duration unavailable";
    el("trajectory-camera").textContent = state.setup.video_camera || "ext1";
    el("trajectory-task").textContent = state.setup.video_task || "";
  } else {
    const status = el("trajectory-state");
    status.textContent = "NO SUCCESS VIDEO";
    status.className = "unavailable";
    el("trajectory-duration").textContent = "Failure trajectories excluded";
    el("trajectory-camera").textContent = "";
    el("trajectory-task").textContent = "";
  }
  document.querySelectorAll("#media-mode button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.mediaMode);
    button.disabled = button.dataset.mode === "video" && !state.setup.video_url;
  });
}

function renderSetup() {
  const setup = state.setup;
  el("asset-empty").hidden = true;
  el("asset-content").hidden = false;
  el("setup-id").textContent = `${setup.scene_id} / ${setup.setup_id}`;
  el("setup-title").textContent = setup.title;
  el("asset-count").textContent = `${setup.assets.length} total`;
  assetList.innerHTML = setup.assets.map((asset) => {
    const score = Number(asset.score || 0);
    return `
      <button class="asset-row ${state.asset?.key === asset.key ? "active" : ""}" type="button" data-asset="${escapeHtml(asset.key)}">
        <span class="row-top">
          <span class="row-id">#${String(asset.instance + 1).padStart(2, "0")}</span>
          <span class="row-count">${(asset.glb_bytes / 1e6).toFixed(1)} MB</span>
        </span>
        <span class="row-title">${escapeHtml(asset.label)}</span>
        <span class="score-bar"><span style="width:${Math.max(4, score * 100)}%"></span></span>
      </button>`;
  }).join("") || `<div class="empty-state">No generated output is available for this setup.</div>`;
  renderScene();
}

async function selectScene(sceneId, preferredSetup, preferredAsset) {
  state.scene = await api(`/api/scenes/${encodeURIComponent(sceneId)}`);
  state.setup = null;
  state.asset = null;
  renderScenes();
  renderScene();
  const setupId = preferredSetup && state.scene.setups.some((row) => row.setup_id === preferredSetup)
    ? preferredSetup
    : state.scene.setups.find((row) => row.asset_count)?.setup_id || state.scene.setups[0]?.setup_id;
  if (setupId) await selectSetup(setupId, preferredAsset);
  updateUrl();
}

async function selectSetup(setupId, preferredAsset) {
  state.setup = await api(`/api/setups/${encodeURIComponent(setupId)}`);
  state.asset = null;
  state.mediaMode = state.setup.video_url ? "video" : (state.setup.has_detections ? "detections" : "source");
  renderSetup();
  const asset = state.setup.assets.find((row) => row.key === preferredAsset) || state.setup.assets[0];
  if (asset) selectAsset(asset.key);
  else viewer.clear();
  updateUrl();
}

function selectAsset(assetKey) {
  state.asset = state.setup.assets.find((row) => row.key === assetKey);
  if (!state.asset) return;
  renderSetup();
  el("asset-caption").hidden = false;
  el("asset-name").textContent = `${state.asset.label} · instance ${state.asset.instance + 1}`;
  const detection = state.asset.score == null ? "detection n/a" : `detection ${(state.asset.score * 100).toFixed(1)}%`;
  const mask = state.asset.mask_score == null ? "mask n/a" : `mask ${(state.asset.mask_score * 100).toFixed(1)}%`;
  el("asset-scores").textContent = `${detection} · ${mask}`;
  el("download-glb").href = state.asset.glb_url;
  el("download-ply").href = state.asset.ply_url;
  el("result-json-link").href = state.asset.result_url;
  viewer.load(state.asset.glb_url);
  updateUrl();
}

class AssetViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d0e);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1000);
    this.camera.position.set(2.4, 1.7, 2.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);
    this.loader = new GLTFLoader();
    this.model = null;
    this.defaultView = null;
    this.addLights();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this.animate();
  }

  addLights() {
    this.scene.add(new THREE.HemisphereLight(0xe8f4ff, 0x423c35, 2.4));
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(3, 5, 4);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x86bfff, 1.4);
    fill.position.set(-4, 2, -2);
    this.scene.add(fill);
    const floor = new THREE.GridHelper(10, 20, 0x31403a, 0x232a27);
    floor.position.y = -1;
    this.scene.add(floor);
  }

  resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  clear() {
    if (this.model) {
      this.scene.remove(this.model);
      this.model.traverse((node) => {
        node.geometry?.dispose();
        if (node.material) {
          const materials = Array.isArray(node.material) ? node.material : [node.material];
          materials.forEach((material) => material.dispose());
        }
      });
      this.model = null;
    }
    el("viewer-status").textContent = "No generated output";
    el("viewer-status").hidden = false;
    el("asset-caption").hidden = true;
  }

  load(url) {
    this.clear();
    el("viewer-status").textContent = "Loading 3D model";
    this.loader.load(url, (gltf) => {
      this.model = gltf.scene;
      this.scene.add(this.model);
      this.frameModel();
      el("viewer-status").hidden = true;
      el("asset-caption").hidden = false;
    }, undefined, (error) => {
      console.error(error);
      el("viewer-status").textContent = "Unable to load 3D model";
      showError("The GLB could not be loaded.");
    });
  }

  frameModel() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxSize = Math.max(size.x, size.y, size.z) || 1;
    this.model.position.sub(center);
    this.model.position.y += size.y * 0.08;
    const distance = maxSize / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.65;
    const direction = new THREE.Vector3(1.25, 0.8, 1.25).normalize();
    this.camera.position.copy(direction.multiplyScalar(distance));
    this.camera.near = Math.max(distance / 1000, 0.0001);
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.minDistance = distance * 0.25;
    this.controls.maxDistance = distance * 5;
    this.controls.update();
    this.defaultView = { position: this.camera.position.clone(), target: this.controls.target.clone() };
  }

  reset() {
    if (!this.defaultView) return;
    this.camera.position.copy(this.defaultView.position);
    this.controls.target.copy(this.defaultView.target);
    this.controls.update();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (state.spin && this.model) this.model.rotation.y += 0.004;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

const viewer = new AssetViewer(el("viewer"));

sceneList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-scene]");
  if (button) selectScene(button.dataset.scene).catch((error) => showError(error.message));
});
setupList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-setup]");
  if (button) selectSetup(button.dataset.setup).catch((error) => showError(error.message));
});
assetList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-asset]");
  if (button) selectAsset(button.dataset.asset);
});
el("media-mode").addEventListener("click", (event) => {
  const button = event.target.closest("[data-mode]");
  if (!button || button.disabled) return;
  state.mediaMode = button.dataset.mode;
  renderSetupMedia();
});
el("reset-camera").addEventListener("click", () => viewer.reset());
el("toggle-spin").addEventListener("click", (event) => {
  state.spin = !state.spin;
  event.currentTarget.setAttribute("aria-pressed", String(state.spin));
  event.currentTarget.textContent = state.spin ? "Spin on" : "Spin off";
});

let searchTimer;
el("scene-search").addEventListener("input", (event) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const result = await api(`/api/scenes?limit=1000&q=${encodeURIComponent(event.target.value)}`);
    state.scenes = result.items;
    renderScenes();
  }, 180);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && document.activeElement !== el("scene-search")) {
    event.preventDefault();
    el("scene-search").focus();
  }
});

async function boot() {
  try {
    const [summary, scenes] = await Promise.all([api("/api/summary"), api("/api/scenes?limit=1000")]);
    renderSummary(summary);
    state.scenes = scenes.items;
    renderScenes();
    const params = new URLSearchParams(location.search);
    const requestedScene = params.get("scene");
    const sceneId = state.scenes.some((row) => row.scene_id === requestedScene)
      ? requestedScene
      : state.scenes[0]?.scene_id;
    if (sceneId) await selectScene(sceneId, params.get("setup"), params.get("asset"));
  } catch (error) {
    console.error(error);
    showError(`Unable to load the catalog: ${error.message}`);
  }
}

boot();
