"""Build the PhysicsGen scene library + model bench UI (single HTML, inline assets).

Three tabs: scene library (every implemented scene with S-grid previews, GIF pairs,
verification), design goals (P0–P4 ladder, principles, contract, metric structure),
model bench (each evaluated model: mechanism diagram, real examples, numbers, verdict).
Usage: PYTHONPATH=src python src/exp/build_scene_model_ui.py <dst.html>
"""
from __future__ import annotations

import base64, io, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/mnt/nvme/migration/jihun/SimDROID/code_vwm/artifacts")
CODE_OUT = Path("/mnt/nvme/migration/jihun/SimDROID/code/out")


# ---------------------------------------------------------------- assets
def b64_file(path: Path, mime: str) -> str | None:
    if not path.exists():
        return None
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def b64_img(path: Path, max_w: int = 1200, quality: int = 80, fmt: str = "JPEG") -> str | None:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def b64_array(arr: np.ndarray, scale: int = 1, quality: int = 82) -> str:
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def gif_pair(path: Path):
    """(animated gif data uri, first-frame jpeg data uri) or (None, None)."""
    if not path.exists():
        return None, None
    im = Image.open(path); im.seek(0)
    first = b64_array(np.asarray(im.convert("RGB")))
    return b64_file(path, "image/gif"), first


def ensure_pair_gif(root_sub: str, fam: str, lo_id: str, hi_id: str, scale: float = 0.5):
    """Side-by-side GT clip GIF (S<1 | S>1) for a family; built once from the gt mp4s."""
    out = ROOT / root_sub / "gifs" / f"{fam}_pair.gif"
    if out.exists():
        return out
    import imageio.v2 as imageio
    lo = imageio.mimread(ROOT / root_sub / "gt" / f"{lo_id}.mp4", memtest=False)[:21]
    hi = imageio.mimread(ROOT / root_sub / "gt" / f"{hi_id}.mp4", memtest=False)[:21]
    frames = []
    for a, b in zip(lo, hi):
        im = Image.fromarray(np.concatenate([a, np.full((a.shape[0], 6, 3), 35, np.uint8), b], axis=1))
        frames.append(im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=62, loop=0, optimize=True)
    return out


def vera_strip(sid: str, seed: int = 1):
    gp, rp = ROOT / "phase_a/vera" / f"{sid}.npz", ROOT / f"phase_a/vera_out/seed_{seed:02d}" / sid / "rollout.npz"
    if not (gp.exists() and rp.exists()):
        return None
    gt = np.load(gp)["side_view"]; gen = np.load(rp)["side_view"]
    idx = [28, 40, 52, 64, 76, 88, 100]
    row_gt = np.concatenate([gt[min(i, len(gt) - 1)] for i in idx], axis=1)
    row_gen = np.concatenate([gen[min(i, len(gen) - 1)] for i in idx], axis=1)
    return b64_array(np.concatenate([row_gt, row_gen], axis=0), scale=2)


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x, nd=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}%"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- content
def scene_cards(toy, wb, verify_a, verify_p):
    """Scene card definitions: (id, set, title, class chip, formula, rule, preview path, gif path, spec dl)."""
    def bis(fam):
        b = verify_a[fam]["bisection"]
        return f'폐형식 S*=1 대비 시뮬 경계 {b["S_star_sim"]:.3f} (차이 {b["gap_percent"]:.1f}% &lt; 격자 한 칸 {b["grid_cell_percent"]:.0f}%)'
    cards = [
        dict(id="hill_roll", set="Bundle A · toy 외관", title="A1 구름 언덕", cls="P1-A 중력·연속", bundle="A1↔A3 원리 짝",
             formula="S = 0.7·v₀² / (g·h)", rule="S &gt; 1 이면 넘어감(1), 아니면 정점 전에 돌아옴(0). 구름 운동이라 7/10 v₀² = 7/10 v² + g·y.",
             preview=ROOT / "bundle_a/previews/hill_roll.png", gif=ROOT / "bundle_a/gifs/hill_roll_pair.gif",
             spec=[("결정 변수", "언덕 높이 h (v₀ = 1.0 m/s 고정, 폭 0.30 m 고정, sin² 접선 연속 프로파일)"),
                   ("사건", "정점 통과 또는 반환 — 0.6–0.9 s (프레임 10–15)"), ("판독", "빨간 공 중심 궤적 → 방향, (y, v²) 직선 기울기(구름 −14.0 m/s² 기대)"),
                   ("검증", bis("hill_roll") + " · timestep 0.5/1 ms 결과 일치 · μ·밀도 변경에 결과 불변")]),
        dict(id="two_ball", set="Bundle A · toy 외관", title="A2 두 공 정면충돌", cls="P2-B 충격·운동량", bundle="A2-0 벽 반동 사전검사와 짝",
             formula="S = (r₂ / r₁)³  (등밀도 → 질량비)", rule="e ≈ 1 에서 m₁ &lt; m₂ 즉 S &gt; 1 이면 빨간 공이 되돌아옴(1), 아니면 계속 진행(0).",
             preview=ROOT / "bundle_a/previews/two_ball.png", gif=ROOT / "bundle_a/gifs/two_ball_pair.gif",
             spec=[("결정 변수", "파란 공 반지름 r₂ (빨간 공 r₁ = 4 cm, v₀ = 1.0 m/s, 근-무마찰 바닥)"),
                   ("사건", "접촉 0.44 s (프레임 7), 판독 창은 충돌 직후 2–3프레임"), ("판독", "충돌 후 빨간 공 방향 + Σmv 비(운동량 보존, GT 0.98–1.00)"),
                   ("검증", bis("two_ball") + " · 접촉 감쇠·밀도 변경에 결과 불변 · e_eff = 0.996")]),
        dict(id="pendulum_rod", set="Bundle A · toy 외관", title="A3 막대 진자", cls="P1-B 구속 보존", bundle="A1 언덕과 원리 짝",
             formula="S = v₀² / (4·g·L)", rule="S &gt; 1 이면 정점을 넘어 회전(1), 아니면 되돌아옴(0). 높이–속도 관계는 언덕과 같은 에너지 법칙.",
             preview=ROOT / "bundle_a/previews/pendulum_rod.png", gif=ROOT / "bundle_a/gifs/pendulum_rod_pair.gif",
             spec=[("결정 변수", "팔 길이 L (v₀ = 3.6 m/s 고정 — 공통 1.0이면 임계 L이 2.5 cm로 비가시)"),
                   ("사건", "정점 통과 또는 반환 — 0.4–1.1 s (S에 따라 프레임 6–18)"), ("판독", "봅 각도 궤적 → 회전/반환, 각도-속도 에너지 기울기"),
                   ("검증", bis("pendulum_rod") + " · 밀도·시작각 변경에 결과 불변")]),
        dict(id="kin_roll", set="Phase A · 작업대 외관", title="P0 등속 구름 (제어)", cls="P0 등속 연장", bundle="결정 변수 없음",
             formula="x(t) = x₀ + v₀·t", rule="사건이 없다. 등속 외삽이 정답이므로 생성 실패와 물리 실패를 가른다.",
             preview=ROOT / "phase_a/previews/kin_roll.png", gif=None,
             spec=[("스윕", "v₀ ∈ {0.30, 0.45, 0.60, 0.75, 0.90} m/s, 평탄한 검사대"), ("판독", "생성 클립 자신의 조건 프레임으로 등속 외삽한 궤적과의 오차, 변위비 gen/GT"),
                   ("검증", "MuJoCo 속도비 1.000 · slip 2e-7 · 에너지 드리프트 &lt; 0.04%")]),
        dict(id="hill_roll_wb", set="Phase A · 작업대 외관", title="A1 언덕 · 작업대", cls="P1-A 중력·연속", bundle="toy A1과 물리 동일",
             formula="S = 0.7·v₀² / (g·h)", rule="초록 케이블커버 hump. 옵션·접촉쌍·충돌 geom·hfield가 toy와 컴파일 수준에서 동일.",
             preview=ROOT / "phase_a/previews/hill_roll_wb.png", gif=None,
             spec=[("외관", "회색 라미네이트 검사대, 뒷벽, 정지한 Franka Panda(충돌 없음), 트레이·픽스처·빈, 테이프 눈금"),
                   ("검증", "toy↔작업대 물리 서명 identical(접촉쌍 2), 격자 게이트 11/11 통과, 사건 프레임 동일")]),
        dict(id="two_ball_wb", set="Phase A · 작업대 외관", title="A2 두 공 · 작업대", cls="P2-B 충격·운동량", bundle="toy A2와 물리 동일",
             formula="S = (r₂ / r₁)³", rule="같은 재질의 두 부품이 저마찰 검사대 위에서 정면충돌.",
             preview=ROOT / "phase_a/previews/two_ball_wb.png", gif=None,
             spec=[("외관", "작업대 wrapper 공통 + 테이프 눈금 위치만 조정"), ("검증", "물리 서명 identical(접촉쌍 3: floor·gA·gB), 격자 게이트 11/11 통과")]),
        dict(id="pendulum_rod_wb", set="Phase B · 작업대 외관 (2026-09-03)", title="B1 매달린 payload (A3 재스킨)", cls="P1-B 구속 보존", bundle="toy A3와 물리 동일 · A1 언덕과 원리 짝",
             formula="S = v₀²(1 + 0.4r²/L²) / (4·g·L)", rule="steel 포스트에 봉으로 매달린 붉은 payload가 최저점을 지나 올라간다. 관절·질량·타임스텝이 toy A3와 같아 궤적 차가 0이다.",
             preview=ROOT / "phase_b/previews/pendulum_rod_wb.png", gif=ensure_pair_gif("phase_b", "pendulum_rod_wb", "pendulum_rod_wb_02", "pendulum_rod_wb_08"),
             spec=[("결정 변수", "봉 길이 L (v₀ = 3.6 m/s 고정) — S 0.5–2.0 ↔ L 0.66–0.17 m"),
                   ("사건", "정점 통과 또는 반환 — 프레임 6(S=2.0)–12(S≤0.95)"), ("판독", "붉은 payload 궤적 원 적합 → 피벗·각도, 에너지 기울기 (toy와 동일 판독기)"),
                   ("검증", "toy↔작업대 물리 서명 identical · 격자 게이트 11/11 · GT 자가검증 11/11 · 모델 실행은 씬 피드백 후")]),
        dict(id="support_edge_wb", set="Phase B · 작업대 외관 · 신규 씬", title="B2 지지 끝 이탈", cls="P3-A 접촉 상실", bundle="P0 등속과 짝: S &lt; 1 은 등속이 정답",
             formula="S = v₀·T_REF / (x_edge − x_start),  T_REF = 1.176 s", rule="S &gt; 1 이면 지평 안에 steel 플레이트 끝을 지나 검사대로 떨어짐(1), 아니면 끝까지 지지됨(0). 등속 외삽은 끝을 지나서도 같은 높이로 직진하는 '공중 부양'을 예측한다.",
             preview=ROOT / "phase_b/previews/support_edge_wb.png", gif=ensure_pair_gif("phase_b", "support_edge_wb", "support_edge_wb_02", "support_edge_wb_08"),
             spec=[("결정 변수", "플레이트 끝 위치 x_edge (v₀ = 0.35 m/s 고정, 플레이트 높이 0.15 m, 공 6 cm)"),
                   ("사건", "S=2.0 프레임 11 · 1.26 → 17 · 1.12 → 18; S ≤ 0.95 는 지평 끝(프레임 20)까지 지지"),
                   ("판독", "빨간 공 중심 행이 문맥 수준보다 0.6 지름 이상 하강 = 낙하; 낙하 초기 가속도/g (포물선 법칙)"),
                   ("검증", "T_REF 를 시뮬 이분법으로 보정해 경계 S = 0.9998 · 끝 도달 전 속도비 1.0000 · toy↔작업대 서명 identical · GT 자가검증 10/11 (+S = 1.00 나이프에지)")]),
    ]
    return cards


def main(dst):
    toy = load_json(ROOT / "bundle_a/analysis/summary.json")
    wb = load_json(ROOT / "phase_a/analysis/summary.json")
    vera = load_json(ROOT / "phase_a/analysis_vera_s8/summary.json") or load_json(ROOT / "phase_a/analysis_vera/summary.json")
    vera_seeds = round(vera["hill_roll_wb"]["n"] / 6) if vera else 0
    dila_pix = {t: load_json(ROOT / f"dila/phase_a/pixels{t}/summary.json")
                for t in ("", "_zero", "_mean", "_c3", "_rev", "_s2", "_s1", "_s1_zero", "_oracle", "_s1_oracle")}
    dila_lat = {t: load_json(ROOT / f"dila/phase_a/analysis{t}/summary.json")
                for t in ("", "_zero", "_mean", "_c3", "_rev", "_s2", "_s1", "_s1_zero", "_oracle", "_s1_oracle")}
    vj = load_json(ROOT / "vjepa_ac/phase_a/analysis/summary.json")
    vj_rev = load_json(ROOT / "vjepa_ac/phase_a/analysis_rev/summary.json")
    verify_a = load_json(ROOT / "bundle_a/verify.json")
    verify_p = load_json(ROOT / "phase_a/verify.json")
    lib = load_json(ROOT / "scene_library/report.json") or {}

    # ---- scene cards
    cards_html = ""
    for c in scene_cards(toy, wb, verify_a, verify_p):
        prev = b64_img(c["preview"], max_w=1400, quality=78)
        gif, still = gif_pair(c["gif"]) if c["gif"] else (None, None)
        spec = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in c["spec"])
        media = f'<div class="media"><img src="{prev}" alt="{c["title"]} S 격자 프리뷰"></div><p class="cap">열 = S 격자 11점(0.50–2.00), 행 = t=0 / 조건 끝 0.25 s / +0.31 s / 사건 / t=1.25 s. y = GT 결과.</p>' if prev else ""
        if gif:
            media += f'<div class="media gifwrap"><img class="gif" src="{gif}" data-anim="{gif}" data-still="{still}" alt="{c["title"]} S&lt;1 대 S&gt;1 GT 클립"></div><p class="cap">GT 21프레임 (16 fps): 왼쪽 S &lt; 1, 오른쪽 S &gt; 1 — 같은 카메라·물체·초기 운동에서 결정 변수 하나만 다르다.</p>'
        cards_html += f"""<article class="card scene">
  <div class="card-head"><div><span class="eyebrow">{c["set"]}</span><h3>{c["title"]}</h3></div><div class="chips"><span class="chip cls">{c["cls"]}</span><span class="chip">{c["bundle"]}</span></div></div>
  <p class="formula mono">{c["formula"]}</p>
  <p>{c["rule"]}</p>
  {media}
  <dl class="spec">{spec}</dl>
</article>
"""
    # prechecks + vjepa view + early modules
    pre_html = ""
    for gid, lab, cap in (("hill_roll_pre", "A1-0 감속 사전검사 (S = 0.35)", "물리적으로 확실히 되돌아오는 낮은 에너지. 관성 외삽만 하는 모델은 여기서도 '넘어감'을 그린다."),
                          ("wall_bounce_pre", "A2-0 벽 반동 사전검사", "질량비를 읽기 전에 '접촉 후 방향이 바뀐다'를 그릴 수 있는지 보는 정성 검사.")):
        gif, still = gif_pair(ROOT / "bundle_a/gifs" / f"{gid}.gif")
        if gif:
            pre_html += f'<figure class="pre"><div class="media gifwrap"><img class="gif" src="{gif}" data-anim="{gif}" data-still="{still}" alt="{lab}"></div><figcaption><b>{lab}</b> — {cap}</figcaption></figure>'
    vj_prev = b64_img(ROOT / "vjepa_ac/phase_a/previews/hill_roll_wb_04.png", max_w=1200, quality=78)
    early_html = ""
    EARLY = [("seesaw", "시소", "P4 · Phase 2"), ("lean", "기대선 막대", "P4 · Phase 2"), ("tower", "3단 탑", "P4 · Phase 2"),
             ("hill", "가우시안 언덕", "→ A1 전신"), ("collide", "두 공 (마찰 스윕)", "→ A2 전신"), ("domino", "도미노", "보류"),
             ("energy_hill", "원호 위 구슬", "에너지 묶음"), ("energy_ramp", "경사로·게이트", "P3 후보"), ("energy_pendulum", "진자 (초기)", "→ A3 전신")]
    for key, lab, tag in EARLY:
        p = ROOT / "scene_library" / f"{key}.png"
        b = b64_img(p, max_w=600, quality=74)
        if not b:
            continue
        info = lib.get(key, {})
        sub = f'{info.get("n_frames", "?")}프레임 · 결과 {info.get("outcome", "?")} · margin {info.get("margin", float("nan")):+.2f}' if "n_frames" in info else ""
        early_html += f'<figure class="thumb"><img src="{b}" alt="{lab} t=0 / 중간 / 끝"><figcaption><b>{lab}</b><span class="chip tiny">{tag}</span><span class="small muted">{sub}</span></figcaption></figure>'

    # ---- model numbers
    def acc_row(name, summ, fam, n):
        a = summ[fam]["accuracy_nonboundary"]; lo, hi = wilson(a, n)
        return f'<tr><td>{name}</td><td class="num">{pct(a)} <span class="muted small">[{pct(lo)}, {pct(hi)}]</span></td><td class="num">{pct(summ[fam]["encoder_accuracy_nonboundary"])}</td><td class="num">{pct(summ[fam]["undecided_rate"])}</td><td class="num">{pct(summ[fam]["valid_rate"])}</td></tr>'
    cosmos_rows = (acc_row("A1 언덕 · toy", toy, "hill_roll", 240) + acc_row("A3 진자 · toy", toy, "pendulum_rod", 240) + acc_row("A2 두 공 · toy", toy, "two_ball", 240)
                   + acc_row("A1 언덕 · 작업대", wb, "hill_roll_wb", 240) + acc_row("A2 두 공 · 작업대", wb, "two_ball_wb", 240))
    pb = load_json(ROOT / "phase_b/analysis_pilot/summary.json")
    if pb:
        cosmos_rows += (acc_row("B1 매달린 payload · 작업대 (파일럿 3 seed)", pb, "pendulum_rod_wb", 24)
                        + acc_row("B2 지지 끝 이탈 · 작업대 (파일럿 3 seed)", pb, "support_edge_wb", 24))
    ctrl = wb.get("controls", {}).get("kin_roll", {})
    p0_line = " · ".join(f'v₀ {s["v0"]:.2f}: 변위비 {s["speed_ratio"]:.2f}, 오차 {s["kin_err_px"]:.0f} px' for s in ctrl.get("per_sample", []))
    cosmos_ex = ""
    for sid, root, lab in (("hill_roll_wb_02", "phase_a", "A1 작업대 S=0.79 — GT 반환, 생성은 넘어감"), ("hill_roll_wb_08", "phase_a", "A1 작업대 S=1.26 — GT 통과"),
                           ("two_ball_wb_08", "phase_a", "A2 작업대 S=1.26 — GT 반전, 생성은 정체·직진"), ("pendulum_rod_08", "bundle_a", "A3 toy S=1.26 — GT 회전"),
                           ("wall_bounce_wb_pre", "phase_a", "A2-0 작업대 벽 반동 — GT 반동"), ("kin_roll_02", "phase_a", "P0 v₀=0.60 — 등속"),
                           ("support_edge_wb_10", "phase_b", "B2 지지 끝 S=2.00 (파일럿) — GT 낙하, 생성은 플레이트 높이로 '공중 부양'"),
                           ("pendulum_rod_wb_08", "phase_b", "B1 매달린 payload S=1.26 (파일럿) — GT 회전")):
        b = b64_img(ROOT / root / "cosmos_v2w/seed_01" / sid / "comparison.png", max_w=896, quality=80)
        if b:
            cosmos_ex += f'<figure class="ex"><img src="{b}" alt="{lab}"><figcaption>{lab} <span class="muted">(seed 1: 현재 프레임 | 물리 GT +0.32 s | 생성 +0.31 s | 생성 마지막 1.25 s)</span></figcaption></figure>'
    vera_ex = ""
    for sid, lab in (("hill_roll_wb_02", "A1 S=0.79 (GT 반환)"), ("two_ball_wb_08", "A2 S=1.26 (GT 반전)"), ("wall_bounce_wb_pre", "A2-0 벽 반동")):
        b = vera_strip(sid)
        if b:
            vera_ex += f'<figure class="ex"><div class="media scroll"><img src="{b}" alt="{lab} VERA"></div><figcaption>{lab} — 위 GT, 아래 VERA (seed 1). 열 = 100 Hz 캡처 프레임 28(문맥 끝)·40·52·64·76·88·100 = 0.28–1.0 s. 측면 타일 192×128.</figcaption></figure>'
    vera_rows = ""
    if vera:
        for fam, v in vera.items():
            vera_rows += f'<tr><td>{fam}</td><td class="num">{v["n"]}</td><td class="num">{pct(v["accuracy"])}</td><td class="num">{pct(v["undecided"])}</td><td class="num">{pct(v["valid_frac"])}</td></tr>'
    vj_curves = b64_img(ROOT / "vjepa_ac/phase_a/analysis/curves.png", max_w=1000, quality=80)
    vj_rows = ""
    if vj:
        for fam, lab in (("kin_roll", "P0 등속"), ("hill_roll_wb", "A1 언덕"), ("two_ball_wb", "A2 두 공")):
            f = vj["families"].get(fam); r = (vj_rev or {}).get("families", {}).get(fam) if vj_rev else None
            if not f:
                continue
            bs = f["by_step"]; g = f["gated"]
            cos = " / ".join(f'{bs[k]["cos_P_mean"]:+.2f}' for k in ("1", "2", "3", "4"))
            cos_r = " / ".join(f'{r["by_step"][k]["cos_P_mean"]:+.2f}' for k in ("1", "2", "3", "4")) if r else "—"
            dc = "P≡K" if fam == "kin_roll" else f'{g["dcos_mean"]:+.3f} (오라클 ±{g["dcos_oracle_P_mean"]:.2f})'
            vj_rows += f'<tr><td>{lab}</td><td class="num">{pct(f["gate_rate"])}</td><td class="num">{f["all_steps"]["sep_over_floor_median"]:.1f}</td><td class="num">{cos}</td><td class="num">{cos_r}</td><td class="num">{dc}</td></tr>'
    ada = b64_img(CODE_OUT / "rollout_adaworld/viz_s2_lo2hi.png", max_w=1000, quality=76)
    dd = b64_img(CODE_OUT / "rollout_dreamdojo/hand_slide_mu_s3/zoom_hi2lo.png", max_w=1000, quality=76)
    c25 = b64_img(CODE_OUT / "rollout_cosmos/smoke_slide_mu_s3_t24_neutral/zoom_ctx_hi_neutral.png", max_w=1000, quality=76)

    # ---- DiLA (latent dynamics, past-only rollouts with held/zero latent actions)
    DILA_VARIANTS = [("", "stride 4 (4 fps) · action 유지"), ("_zero", "stride 4 · action 0"), ("_mean", "stride 4 · 문맥 평균 action"),
                     ("_c3", "stride 4 · 문맥 3프레임"), ("_rev", "stride 4 · 문맥 반전 (제어)"), ("_s2", "stride 2 (8 fps) · 유지"),
                     ("_s1", "stride 1 (16 fps, 문맥 5프레임) · 유지"), ("_s1_zero", "stride 1 · action 0"),
                     ("_oracle", "누설 참조: 참 미래 latent action · stride 4"), ("_s1_oracle", "누설 참조 · stride 1")]

    def pres(px, fam):
        if not px or fam not in px["families"]:
            return "—"
        return " / ".join(f'{px["families"][fam][k]["present_rate"]:.2f}' if k in px["families"][fam] else "·" for k in ("1", "2", "3", "4"))

    def dc(lat, fam):
        if not lat or fam not in lat["families"]:
            return "—"
        g = lat["families"][fam]["gated"]
        return f'{g["dcos_mean"]:+.3f}' if g.get("dcos_mean") == g.get("dcos_mean") else "—"

    def trk(lat, fam):
        if not lat or fam not in lat["families"]:
            return "—"
        return f'{lat["families"][fam]["all_steps"]["track_mean"]:.1f}'

    dila_rows = ""
    for tag, lab in DILA_VARIANTS:
        px, lat = dila_pix.get(tag), dila_lat.get(tag)
        if not px and not lat:
            continue
        leak = ' <span class="chip tiny wait">누설</span>' if "oracle" in tag else ""
        dila_rows += (f'<tr><td>{lab}{leak}</td><td class="num">{pres(px, "hill_roll_wb")}</td><td class="num">{pres(px, "two_ball_wb")}</td>'
                      f'<td class="num">{pres(px, "kin_roll")}</td><td class="num">{dc(lat, "hill_roll_wb")} / {dc(lat, "two_ball_wb")}</td><td class="num">{trk(lat, "hill_roll_wb")}</td></tr>')
    dila_ex = ""
    for tag, lab in (("", "stride 4 · action 유지"), ("_s1", "stride 1 · 문맥 5프레임 · action 유지"), ("_oracle", "누설 참조 (참 미래 latent action) · stride 4"), ("_s1_oracle", "누설 참조 · stride 1")):
        b = b64_img(ROOT / "dila/phase_a/previews" / f"examples{tag}.png", max_w=1200, quality=78)
        if b:
            dila_ex += (f'<figure class="ex"><img src="{b}" alt="DiLA {lab}"><figcaption>{lab} — 샘플 A1 작업대 S=0.95, A2 작업대 S=0.95. 각 샘플 4행: GT(문맥+표적 4스텝) · 문맥+등속 K · '
                        f'RAE 디코더로 복원한 GT latent(디코더 상한) · <b>DiLA 예측을 디코드한 것</b>(첫 칸은 문맥 마지막 프레임 복원). 표적 시각 0.5 / 0.75 / 1.0 / 1.25 s.</figcaption></figure>')
    ora = dila_pix.get("_oracle")
    ora_txt = ""
    if ora:
        h = ora["families"]["hill_roll_wb"]; t = ora["families"]["two_ball_wb"]
        ora_txt = (f'누설 참조(참 미래 latent action, stride 4)에서 공 존재율은 A1 {h["1"]["present_rate"]:.2f}/{h["2"]["present_rate"]:.2f}/{h["3"]["present_rate"]:.2f}/{h["4"]["present_rate"]:.2f}, '
                   f'A2 {t["1"]["present_rate"]:.2f}/{t["2"]["present_rate"]:.2f}/{t["3"]["present_rate"]:.2f}/{t["4"]["present_rate"]:.2f}')
    dila_html = f"""
<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 4 · latent 동역학 (구조/내용 분리)</span><h2>DiLA (Disentangled Latent Action world model, ICML 2026)</h2></div><span class="chip no">NO-GO · past-only 롤아웃 붕괴</span></div>
  <div class="pipe"><svg viewBox="0 0 1000 170" role="img" aria-label="DiLA 파이프라인">
    <g font-family="Chakra Petch" font-size="13" font-weight="600" fill="var(--ink)">
      <rect x="10" y="20" width="160" height="54" rx="4" fill="var(--red-bg)" stroke="var(--red)"/><text x="90" y="43" text-anchor="middle">문맥 2–5프레임</text>
      <rect x="215" y="20" width="180" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="305" y="43" text-anchor="middle">DINOv2-RAE 인코더</text>
      <rect x="440" y="20" width="200" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="540" y="43" text-anchor="middle">구조 g · 내용 c 분리</text>
      <rect x="685" y="20" width="150" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="760" y="43" text-anchor="middle">역동역학 → a</text>
      <rect x="215" y="100" width="260" height="54" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="345" y="123" text-anchor="middle">g' = g + f(g, a)  ×T (a 유지/0)</text>
      <rect x="520" y="100" width="200" height="54" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="620" y="123" text-anchor="middle">융합 디코더 → ẑ (768×16²)</text>
      <rect x="765" y="100" width="225" height="54" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="877" y="123" text-anchor="middle">latent 판독 + RAE 디코드 → 픽셀 판독</text>
    </g>
    <g font-family="Noto Sans KR" font-size="11.5" fill="var(--sub)"><text x="90" y="64" text-anchor="middle">256², stride 4 / 2 / 1</text><text x="305" y="64" text-anchor="middle">224² 재표본, 통계 정규화</text><text x="540" y="64" text-anchor="middle">ST-Transformer · Mamba 메모리</text><text x="760" y="64" text-anchor="middle">256-d latent action</text>
      <text x="345" y="144" text-anchor="middle">미래 action은 우리가 정해야 함 (past-only 예측 없음)</text><text x="620" y="144" text-anchor="middle">내용 메모리는 예측으로 갱신</text><text x="877" y="144" text-anchor="middle">dcos · 빨간 공 존재율 · 위치</text></g>
    <g stroke="var(--sub)" stroke-width="1.5" fill="var(--sub)"><line x1="170" y1="47" x2="207" y2="47"/><polygon points="207,42 215,47 207,52"/><line x1="395" y1="47" x2="432" y2="47"/><polygon points="432,42 440,47 432,52"/><line x1="640" y1="47" x2="677" y2="47"/><polygon points="677,42 685,47 677,52"/>
      <line x1="760" y1="74" x2="760" y2="88"/><line x1="760" y1="88" x2="345" y2="88"/><line x1="345" y1="88" x2="345" y2="92"/><polygon points="340,92 345,100 350,92"/><line x1="475" y1="127" x2="512" y2="127"/><polygon points="512,122 520,127 512,132"/><line x1="720" y1="127" x2="757" y2="127"/><polygon points="757,122 765,127 757,132"/></g>
  </svg></div>
  <p>관측만으로 학습된 latent action world model. 프레임을 DINOv2 특징으로 바꾼 뒤 <b>구조(배치·운동)</b>와 <b>내용(외관)</b>으로 나누고, 인접 구조 사이의 역동역학으로 latent action을 뽑아 구조 공간에서 자기회귀로 전개한다. 공개 API는 전체 시퀀스(미래 포함)에서 action을 추론하므로 past-only 예측이 없다. 우리는 문맥 프레임만 넣어 문맥의 action을 얻고, 미래 action을 <b>마지막 값 유지</b>(주 정책) 또는 <b>0</b>으로 두어 전개했다. 판독은 V-JEPA와 같은 latent 지표(P·K·J 후보 미래를 같은 인코더로)에 더해, 디코더가 있으므로 <b>픽셀 판독</b>(빨간 공 존재율·위치)을 병기했다. 입력은 V-JEPA 트랙의 256² 번들을 그대로 재사용했다.</p>
  <h4>실제 예시</h4>
  {dila_ex}
  <h4>결과 (Phase A 29샘플 × 3오프셋 = 87 롤아웃/변형)</h4>
  <div class="tbl-wrap"><table><tr><th>변형</th><th class="num">공 존재율 A1 스텝 1/2/3/4</th><th class="num">A2</th><th class="num">P0</th><th class="num">dcos (게이트) A1 / A2</th><th class="num">track A1</th></tr>{dila_rows}</table></div>
  <p class="small muted">공 존재율 = 디코드한 예측에서 빨간 마스크 면적이 문맥 복원 대비 25% 이상인 비율(디코더 상한: GT latent 복원 존재율 0.93–1.0, 중심 오차 ≈ 1 px). track = 예측과 물리 미래의 거리 / freeze와 물리 미래의 거리 (1 미만이어야 문맥보다 미래에 가까움). {ora_txt}</p>
  <div class="verdict"><b>판정 NO-GO (모델 원인 · past-only 부재 + 자기회귀 붕괴).</b> 미래 action을 유지하든 0으로 두든, 시간축을 4·8·16 fps 어느 것으로 잡든, 구조 롤아웃은 1–4스텝 안에 장면 전체를 흐리고 공을 지운다(스텝 3 이후 존재율 0). latent 거리로도 예측은 실제 인코더 latent 전부에서 멀어지고(track 3–7, V-JEPA의 1.2–2.3보다 큼) 물리/등속 선호가 없다(|dcos| ≤ 0.04, 오라클 ±0.4). 문맥 반전 제어에서도 같은 값. 실패 위치는 V-JEPA(운동 전개 없음)보다 앞: <b>미래 latent action이 주어지지 않으면 상태 자체가 유지되지 않는다</b> — 사용자 문서의 "past-only forecasting 가능 여부" 확인 항목이 부정으로 판정됨. 누설 참조(참 미래 action)는 이 붕괴가 action 정책 탓인지 롤아웃 자체 탓인지 가르는 진단으로만 쓴다.</div>
</section>
"""
    n_scenes = 6 + 2 + 2 + len(EARLY)
    html = f"""<title>PhysicsGen 씬·모델 벤치</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Chakra+Petch:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F1F2F4;--panel:#FFFFFF;--panel2:#F7F8FA;--ink:#1C2128;--sub:#6B7580;--line:#DCE0E6;--red:#C94431;--red-bg:#F8E6E2;--blue:#3A6CB4;--blue-bg:#E4ECF7;--green:#2E9A4E;--green-bg:#E2F2E7;--tape:#E9E2C4;--tape-ink:#6E5F2B;--media:#23262C;--focus:#3A6CB4}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15181D;--panel:#1F242B;--panel2:#252B33;--ink:#E9ECF0;--sub:#98A2AE;--line:#2D333C;--red:#E0604C;--red-bg:#3A231F;--blue:#7FA5E0;--blue-bg:#1F2A3B;--green:#5CBF7A;--green-bg:#1D3126;--tape:#4A4530;--tape-ink:#E3D9AE;--media:#0F1114;--focus:#7FA5E0}}}}
:root[data-theme="dark"]{{--bg:#15181D;--panel:#1F242B;--panel2:#252B33;--ink:#E9ECF0;--sub:#98A2AE;--line:#2D333C;--red:#E0604C;--red-bg:#3A231F;--blue:#7FA5E0;--blue-bg:#1F2A3B;--green:#5CBF7A;--green-bg:#1D3126;--tape:#4A4530;--tape-ink:#E3D9AE;--media:#0F1114;--focus:#7FA5E0}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:15px;line-height:1.7;word-break:keep-all}}
.mono{{font-family:"JetBrains Mono",Consolas,monospace}} .disp{{font-family:"Chakra Petch","Noto Sans KR",sans-serif}}
a{{color:var(--blue)}} main{{max-width:1120px;margin:0 auto;padding:28px 22px 80px}}
header.top{{display:grid;grid-template-columns:1fr;gap:6px;padding:8px 0 18px}}
.eyebrow{{font-family:"Chakra Petch",sans-serif;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub)}}
h1{{font-size:30px;line-height:1.25;margin:2px 0 6px;font-weight:700;text-wrap:balance}} h2{{font-size:21px;margin:0 0 6px;font-weight:700;text-wrap:balance}} h3{{font-size:16.5px;margin:0 0 4px;font-weight:700}} h4{{font-size:14px;margin:14px 0 6px;font-weight:700;color:var(--sub)}}
p{{margin:8px 0}} .lead{{font-size:16.5px;max-width:68ch}} .muted{{color:var(--sub)}} .small{{font-size:13px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;align-items:center}} .chip{{display:inline-block;background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:2px 9px;font-size:12.5px;font-family:"Chakra Petch","Noto Sans KR",sans-serif;font-weight:500;letter-spacing:.02em}}
.chip.cls{{background:var(--blue-bg);color:var(--blue);border-color:transparent}} .chip.ok{{background:var(--green-bg);color:var(--green);border-color:transparent}} .chip.no{{background:var(--red-bg);color:var(--red);border-color:transparent}} .chip.wait{{background:var(--tape);color:var(--tape-ink);border-color:transparent}} .chip.tiny{{font-size:11px;padding:0 6px;margin-left:6px}}
nav.tabs{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);display:flex;gap:4px;align-items:center;padding:8px 0;margin:0 0 18px}}
nav.tabs button{{font-family:"Chakra Petch","Noto Sans KR",sans-serif;font-weight:600;font-size:14px;letter-spacing:.03em;background:transparent;color:var(--sub);border:1px solid transparent;border-radius:6px;padding:8px 14px;cursor:pointer}}
nav.tabs button[aria-selected="true"]{{color:var(--ink);background:var(--panel);border-color:var(--line)}} nav.tabs button:focus-visible,#motion:focus-visible{{outline:2px solid var(--focus);outline-offset:2px}}
nav.tabs .spacer{{flex:1}} #motion{{font-family:"Chakra Petch",sans-serif;font-size:12px;background:var(--panel);color:var(--sub);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}}
section.block{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px 22px;margin:0 0 16px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}} .grid3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px 20px;display:flex;flex-direction:column;gap:8px}} .card-head{{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;flex-wrap:wrap}}
.formula{{font-size:15px;color:var(--ink);background:var(--panel2);border-left:3px solid var(--red);padding:6px 12px;border-radius:0 4px 4px 0;margin:2px 0}}
.media{{background:var(--media);border-radius:6px;padding:6px;margin:6px 0 2px}} .media img{{display:block;width:100%;height:auto;border-radius:3px}} .media.scroll{{overflow-x:auto}} .media.scroll img{{width:auto;max-width:none;height:220px}}
.cap{{font-size:12.5px;color:var(--sub);margin:2px 0 8px}}
dl.spec{{display:grid;grid-template-columns:84px 1fr;gap:4px 12px;margin:4px 0 0;font-size:13.5px}} dl.spec dt{{color:var(--sub)}} dl.spec dd{{margin:0}}
.thumbs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}} figure{{margin:0}} .thumb img{{width:100%;height:auto;border-radius:4px;display:block;background:var(--media)}} .thumb figcaption{{font-size:13px;margin-top:4px;display:flex;flex-wrap:wrap;gap:4px 8px;align-items:center}}
.pres{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px}} .pre figcaption{{font-size:13px;margin-top:4px}}
.tbl-wrap{{overflow-x:auto;margin:8px 0}} table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}} th{{color:var(--sub);font-weight:500;font-size:12.5px}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.ex img{{width:100%;height:auto;display:block;border-radius:4px;background:var(--media)}} .ex figcaption{{font-size:13px;margin:4px 0 12px}}
.ladder svg,.pipe svg,.contract svg{{width:100%;height:auto;display:block}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}} .tile{{background:var(--panel2);border-radius:6px;padding:14px 16px}} .tile h3{{font-size:15px;margin-bottom:6px}} .tile h3 span{{color:var(--red);font-family:"Chakra Petch",sans-serif;margin-right:8px}}
ol.goals{{padding-left:22px;margin:6px 0;columns:2;column-gap:28px}} ol.goals li{{break-inside:avoid;margin:0 0 6px}}
.verdict{{border-left:3px solid var(--red);background:var(--panel2);padding:8px 14px;border-radius:0 6px 6px 0;margin:10px 0}}
.model-head{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap}}
footer{{color:var(--sub);font-size:12.5px;margin-top:22px;line-height:1.7}}
[hidden]{{display:none!important}}
@media (max-width:640px){{main{{padding:18px 14px 60px}} .grid2{{grid-template-columns:1fr}} dl.spec{{grid-template-columns:1fr}} ol.goals{{columns:1}} nav.tabs button{{padding:7px 9px;font-size:13px}}}}
@media (prefers-reduced-motion: reduce){{*{{scroll-behavior:auto}}}}
</style>
<main>
<header class="top">
  <div class="eyebrow">SimDROID · PhysicsGen · MuJoCo 임계 씬 · 2026-09-03</div>
  <h1>임계 씬 라이브러리와 모델 벤치</h1>
  <p class="lead">world model에게 <b>미래를 그리게</b> 했을 때, 현재 속도를 늘려 그리는 것으로는 풀리지 않는 순간 — 언덕 정점, 충돌, 정점 통과 — 에서 물리가 시키는 쪽을 고르는지 본다.
  씬마다 결정 변수 하나를 무차원 포화도 <span class="mono">S</span>로 스윕하고(<span class="mono">S = 1</span>이 경계), 사건 전 프레임만 조건으로 준 뒤 21프레임 미래를 판독한다.</p>
  <div class="chips"><span class="chip">씬 {n_scenes}종 구현</span><span class="chip">Cosmos 1,536 롤아웃</span><span class="chip">VERA 45</span><span class="chip">V-JEPA 2-AC 87 + 제어 134</span><span class="chip">DiLA 87 × 8변형</span><span class="chip no">네 모델 모두 문턱 미반영</span></div>
</header>
<nav class="tabs" role="tablist" aria-label="섹션">
  <button role="tab" id="tab-scenes" aria-controls="p-scenes" aria-selected="true">01 씬 라이브러리</button>
  <button role="tab" id="tab-goals" aria-controls="p-goals" aria-selected="false">02 설계 목표</button>
  <button role="tab" id="tab-models" aria-controls="p-models" aria-selected="false">03 모델 벤치</button>
  <span class="spacer"></span><button id="motion" type="button" aria-pressed="false">GIF 정지</button>
</nav>

<div id="p-scenes" role="tabpanel" aria-labelledby="tab-scenes">
<section class="block contract">
  <h2>조건 계약 — 무엇을 보여주고 무엇을 맞히게 하는가</h2>
  <svg viewBox="0 0 1000 150" role="img" aria-label="조건 프레임과 예측 지평 타임라인">
    <rect x="30" y="52" width="176" height="26" rx="3" fill="var(--red-bg)" stroke="var(--red)"/>
    <rect x="206" y="52" width="692" height="26" rx="3" fill="var(--blue-bg)" stroke="var(--blue)"/>
    <text x="118" y="70" text-anchor="middle" font-family="Chakra Petch" font-size="13" fill="var(--red)">조건 5프레임 · 0.31 s</text>
    <text x="552" y="70" text-anchor="middle" font-family="Chakra Petch" font-size="13" fill="var(--blue)">예측 지평 16프레임 · 1.0 s (총 21프레임 = 1.31 s @ 16 fps)</text>
    <g font-family="JetBrains Mono" font-size="11" fill="var(--sub)">
      <text x="30" y="100">t=0</text><text x="206" y="100" text-anchor="middle">0.31</text><text x="552" y="100" text-anchor="middle">0.8</text><text x="898" y="100" text-anchor="end">1.31 s</text>
    </g>
    <g stroke="var(--sub)" stroke-width="1"><line x1="30" y1="82" x2="30" y2="88"/><line x1="206" y1="82" x2="206" y2="88"/><line x1="552" y1="82" x2="552" y2="88"/><line x1="898" y1="82" x2="898" y2="88"/></g>
    <g font-family="Noto Sans KR" font-size="12" fill="var(--ink)">
      <rect x="318" y="20" width="10" height="10" fill="var(--green)"/><text x="334" y="30">A2 접촉 0.44 s</text>
      <rect x="404" y="20" width="10" height="10" fill="var(--red)"/><text x="420" y="30">A1 정점·반환 0.6–0.9 s</text>
      <rect x="622" y="20" width="10" height="10" fill="var(--blue)"/><text x="638" y="30">A3 정점 0.4–1.1 s</text>
    </g>
    <g font-family="JetBrains Mono" font-size="11" fill="var(--sub)"><text x="30" y="132">V-JEPA 2-AC 시간축: 4 fps = 프레임 0, 4 (문맥) → 8, 12, 16, 20 (예측 4스텝)</text></g>
    <g fill="var(--sub)"><circle cx="30" cy="115" r="2.5"/><circle cx="206" cy="115" r="2.5"/><circle cx="379" cy="115" r="2.5"/><circle cx="552" cy="115" r="2.5"/><circle cx="725" cy="115" r="2.5"/><circle cx="898" cy="115" r="2.5"/></g>
  </svg>
  <p class="small muted">조건 프레임은 사건 전에 끝나야 하고(누설 게이트), 사건은 지평 안에 들어와야 한다. S 격자 {{0.50, 0.63, 0.79, 0.89, 0.95, 1.00, 1.05, 1.12, 1.26, 1.59, 2.00}} × seed 24. |S−1| ≤ 0.04는 경계로 두고 정확도에서 제외한다.</p>
</section>
<div class="grid2">
{cards_html}</div>
<section class="block">
  <h2>사전검사 씬</h2>
  <div class="pres">{pre_html}</div>
</section>
<section class="block">
  <h2>latent 트랙용 정방형 시야</h2>
  {"<div class='media'><img src='" + vj_prev + "' alt='V-JEPA 정방형 카메라 렌더'></div>" if vj_prev else ""}
  <p class="cap">A1 작업대 S=0.95를 256×256 전용 카메라로 다시 렌더한 것. 위: 문맥 두 프레임(0, 0.25 s)과 물리 미래 4스텝 · 가운데: 문맥 속도를 그대로 유지한 등속 counterfactual(MuJoCo qpos 재배치) · 아래: 물리 위치에서 빨간 공만 1 px 옮긴 지터(인코더 잡음 바닥). 물리·카메라 외 요소는 Phase A와 동일.</p>
</section>
<section class="block">
  <h2>초기 후보 씬 (2026-08-31 스윕)</h2>
  <p class="small muted">Bundle A 이전에 만든 9종. 재정의 문서에서 P0–P4 계층으로 재배치됐다: 언덕·두 공·진자는 A1·A2·A3의 전신이 됐고, 시소·기대선 막대·탑은 정적 prior와 사건 지연 문제로 Phase 2(P4), 도미노는 보류. 열 = t=0 / 중간 / 끝.</p>
  <div class="thumbs">{early_html}</div>
</section>
</div>

<div id="p-goals" role="tabpanel" aria-labelledby="tab-goals" hidden>
<section class="block">
  <h2>한 줄 요약</h2>
  <p class="lead">씬을 물리 현상의 이름으로 나열하지 않는다. "현재 상태에서 미래 상태로 가는 데 어떤 <b>역학 전이 법칙</b>이 필요한가"로 계층화하고, 원칙은 <b>사실적 외관 + 통제된 역학 + 짝지은 대조</b>다. 가림(occlusion)은 물리 수준이 아니라 별도의 관측 축으로 둔다.</p>
  <p>판별하려는 것: WM의 미래 예측이 외관·궤적의 연장(H<sub>visual</sub>)인지, 상태를 전파하는 것(H<sub>physics</sub>: s<sub>t</sub> = E(x), ŝ = F<sub>physics</sub>(s<sub>t</sub>), x̂ = D(ŝ))인지. 증거는 linear probe가 아니라 <b>예측 출력과 그 출력이 만족하는 dynamics 관계</b>다.</p>
</section>
<section class="block ladder">
  <h2>주축 — 역학 전이 계층 P0–P4</h2>
  <svg viewBox="0 0 1000 300" role="img" aria-label="P0에서 P4까지의 계층 사다리와 씬 배치">
    <g font-family="Chakra Petch" font-size="14" font-weight="600">
      <rect x="20" y="220" width="180" height="60" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="30" y="243" fill="var(--ink)">P0 등속 연장</text>
      <rect x="215" y="180" width="180" height="100" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="225" y="203" fill="var(--blue)">P1 연속 역학</text>
      <rect x="410" y="140" width="180" height="140" rx="4" fill="var(--red-bg)" stroke="var(--red)"/><text x="420" y="163" fill="var(--red)">P2 충격 상호작용</text>
      <rect x="605" y="100" width="180" height="180" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="615" y="123" fill="var(--green)">P3 접촉 전이</text>
      <rect x="800" y="60" width="180" height="220" rx="4" fill="var(--panel2)" stroke="var(--line)" stroke-dasharray="5 4"/><text x="810" y="83" fill="var(--sub)">P4 안정성 상실</text>
    </g>
    <g font-family="Noto Sans KR" font-size="12.5" fill="var(--ink)">
      <text x="30" y="265">P0 등속 구름 [제어]</text>
      <text x="225" y="226">A1 언덕 (중력)</text><text x="225" y="246">A3 진자 · B1 payload (구속 보존)</text><text x="225" y="266" fill="var(--sub)">P1-C 마찰 정지 (제한)</text>
      <text x="420" y="186">A2 두 공 충돌 (주)</text><text x="420" y="206">A2-0 벽 반동 (사전검사)</text>
      <text x="615" y="146">B2 지지 끝 이탈 → 자유낙하 (구현)</text><text x="615" y="166" fill="var(--sub)">가이드 진입 · 접촉 상실 (후속)</text><text x="615" y="192" fill="var(--sub)">Phase B · 씬 완료 · 모델 실행 대기</text>
      <text x="810" y="106" fill="var(--sub)">기울어짐 · 흔들린 탑 · 시소</text><text x="810" y="132" fill="var(--sub)">Phase 2 (정적 prior·사건 지연 해결 후)</text>
    </g>
    <g font-family="JetBrains Mono" font-size="11" fill="var(--sub)"><text x="20" y="24">현재 속도 연장으로 풀림 →</text><text x="980" y="24" text-anchor="end">→ 외삽이 가장 강하게 틀림</text></g>
    <line x1="20" y1="32" x2="980" y2="32" stroke="var(--line)"/>
    <g font-family="Noto Sans KR" font-size="11.5" fill="var(--sub)"><text x="20" y="298">구현·평가 완료: P0, P1(A1·A3), P2(A2·A2-0) · 제외: 변형·좌굴·파단·강성 식별 (내부 상태 비가시, VAE 해상도)</text></g>
  </svg>
</section>
<section class="block">
  <h2>세 원칙 — Phase A에서의 구현</h2>
  <div class="tiles">
    <div class="tile"><h3><span>R</span>사실적 외관</h3><p class="small">작업대·픽스처·정지한 로봇팔·자연스러운 조명과 재질. 단, 잡동사니가 결정 변수를 가려선 안 된다. 구현: 회색 라미네이트 검사대, 뒷벽, 정지 Franka Panda(충돌 없음), 트레이·픽스처·빈, 테이프 눈금.</p></div>
    <div class="tile"><h3><span>C</span>통제된 역학</h3><p class="small">스윕 변수 하나, 나머지는 고정하거나 ablation. 해석적 관계가 있고 사건이 지평 안에 있으며 solver·timestep에 둔감. 구현: h 또는 r₂만 스윕, 경계 이분법·에너지 감사·timestep 일치·nuisance 불변 검사 통과 후에만 모델 실행.</p></div>
    <div class="tile"><h3><span>P</span>짝지은 대조</h3><p class="small">같은 카메라·물체·질감·배경·초기 운동에서 물리 변수 하나만 바꿔 결과 A/B를 만든다. 구현: S 격자의 경계 양쪽(0.95 / 1.05)이 최소 단위, toy↔작업대는 물리를 고정한 외관 대조.</p></div>
  </div>
</section>
<section class="block pipe">
  <h2>측정 경로 x → f → y → g(y)</h2>
  <svg viewBox="0 0 1000 120" role="img" aria-label="입력에서 지표까지의 경로">
    <g font-family="Chakra Petch" font-size="14" font-weight="600" fill="var(--ink)">
      <rect x="10" y="30" width="200" height="58" rx="4" fill="var(--red-bg)" stroke="var(--red)"/><text x="110" y="54" text-anchor="middle">x  시각 history</text>
      <rect x="270" y="30" width="180" height="58" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="360" y="54" text-anchor="middle">f  world model</text>
      <rect x="510" y="30" width="200" height="58" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="610" y="54" text-anchor="middle">y  미래 영상 / latent</text>
      <rect x="770" y="30" width="220" height="58" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="880" y="54" text-anchor="middle">g(y)  판독기</text>
    </g>
    <g font-family="Noto Sans KR" font-size="11.5" fill="var(--sub)">
      <text x="110" y="76" text-anchor="middle">5 / 9 / 13 프레임 (사건 전)</text><text x="360" y="76" text-anchor="middle">Cosmos · VERA · V-JEPA 2-AC</text><text x="610" y="76" text-anchor="middle">21프레임 · 3청크 · 4스텝 latent</text><text x="880" y="76" text-anchor="middle">중심·속도·각도·접촉 → 결과 + 법칙 잔차</text>
    </g>
    <g stroke="var(--sub)" stroke-width="1.5" fill="var(--sub)"><line x1="210" y1="59" x2="262" y2="59"/><polygon points="262,54 270,59 262,64"/><line x1="450" y1="59" x2="502" y2="59"/><polygon points="502,54 510,59 502,64"/><line x1="710" y1="59" x2="762" y2="59"/><polygon points="762,54 770,59 762,64"/></g>
  </svg>
  <div class="tbl-wrap"><table>
    <tr><th>지표 구조</th><th>정의</th><th>이번 구현</th></tr>
    <tr><td>18.1 Validity</td><td>소실·복제·형상·침투·카메라 drift</td><td>색 마스크 면적 0.35–2.8× 게이트(strict), 프레임 유효 비율(valid_frac)</td></tr>
    <tr><td>18.2 Kinematic baseline gap</td><td>G = Score<sub>WM</sub> − Score<sub>kin</sub></td><td>등속 인코더(항상 "넘어감·직진") 정확도 0.50과 비교, P0 제어에서 등속 외삽 오차</td></tr>
    <tr><td>18.3 Outcome accuracy</td><td>방향 선택 정답률</td><td>비경계 S 240행, Wilson 95% CI</td></tr>
    <tr><td>18.4 Law residual</td><td>ε<sub>E</sub>, ε<sub>p</sub>, 법선 속도 부호</td><td>(y, v²) 기울기(구름 −14.0), Σmv 비(GT 0.98–1.00)</td></tr>
    <tr><td>18.5 Threshold response</td><td>P(outcome | S): 단조성·PSE·기울기</td><td>S 격자 11점 곡선, 로지스틱 S50 (진자 0.82)</td></tr>
    <tr><td>18.6 Principle consistency</td><td>같은 원리 씬 묶음의 곡선 일치</td><td>A1↔A3 (언덕 평탄 vs 진자 S자) — 불일치가 결과</td></tr>
    <tr><td>18.7 Occlusion robustness</td><td>R<sub>occ</sub>, 가시 씬과 짝</td><td>Phase C 예정 (O1 그리퍼 / O2 픽스처 / O4 시점)</td></tr>
  </table></div>
</section>
<section class="block">
  <h2>씬 입장 조건과 렌더 원칙</h2>
  <div class="grid3">
    <div class="tile"><h3>입장 조건 7항</h3><p class="small">폐형식 결정 부등식 · 중심 변수 하나(사진 계수 고정) · 무차원 S(S=1 경계) · 조건 창은 외삽 이탈 전 종료 · 이진 사건 판독(경계 근처 critical slowing 유의) · 묶음 내 v₀ 고정 · 사전검사 통과.</p></div>
    <div class="tile"><h3>렌더 원칙</h3><p class="small">척도 단서(눈금·기준 물체), 측면 근직교 카메라, 물체는 두껍게(VAE 8×8 압축에서 얇은 구조 소실), 색 분리(빨강·파랑·초록만), 회전 성분 무시, 마지막 한 장이 아니라 21프레임 궤적으로 판정.</p></div>
    <div class="tile"><h3>가림 축 O0–O4</h3><p class="small">O0 완전 가시 / O1 조작기 / O2 물체·잡동사니 / O3 용기 / O4 시점·자기 가림. 규칙: 가려지기 전 상태 복원 가능, 가려진 동안 전이 발생, 원인은 가려지기 전 가시, 완전 가시 짝과 Δ<sub>occlusion</sub> 측정. 인위적 마스크 금지.</p></div>
  </div>
</section>
<section class="block">
  <h2>설계 목표 우선순위와 단계</h2>
  <ol class="goals">
    <li>예측에서 물리를 검증한다 (표현 probing이 아니라)</li><li>외삽으로 못 푸는 씬</li><li>정답이 history에서 식별 가능</li><li>사실적이되 인과는 단순</li><li>짝지은 대조가 기본 단위</li>
    <li>이진 + 연속 지표 동시</li><li>원리 전이 (씬 묶음)</li><li>자연스러운 가림</li><li>시공간 해상도에 맞는 물리 (VAE 상한 먼저)</li><li>시뮬레이터 인공물 배제 (timestep·solver·에너지 감사·사건 지평)</li>
  </ol>
  <div class="chips" style="margin-top:10px"><span class="chip ok">Phase A · P0 + A1·A2 작업대 — 완료</span><span class="chip ok">Phase B · 매달린 payload · 지지 끝 이탈 — 씬 완료 (09-03), 모델 실행은 피드백 후</span><span class="chip wait">Phase C · 자연 가림 짝</span><span class="chip">Phase D · 기울어짐 · 흔들린 탑 · 시소</span></div>
  <p class="small muted" style="margin-top:8px">최종 결과물은 leaderboard가 아니라 <b>Physical Prediction Profile</b>: 등속 / 연속 / 충격 / 접촉 전이 / 안정성 별 등급 + 원리 전이, 문턱 정밀도, 가림 강건성, 법칙 일관성.</p>
</section>
</div>

<div id="p-models" role="tabpanel" aria-labelledby="tab-models" hidden>
<section class="block">
  <h2>한눈에</h2>
  <div class="tbl-wrap"><table>
    <tr><th>모델</th><th>종류</th><th>입력 → 출력</th><th class="num">규모</th><th>핵심 결과</th><th>판정</th></tr>
    <tr><td>Cosmos-Predict2-2B Video2World</td><td>픽셀 생성 (diffusion, 텍스트+영상 조건)</td><td>5프레임 0.31 s → 21프레임 1.31 s, 480×832</td><td class="num">35+29행 × 24 seed = 1,536</td><td>언덕 전 구간 통과, 두 공 반전 못 그림, 진자만 S자(S50 0.82). 텍스트 지시로도 방향 안 바뀜</td><td><span class="chip no">문턱 미반영 · 생성 한계</span></td></tr>
    <tr><td>VERA DROID planner</td><td>픽셀 생성 (Wan2.1-I2V-14B, 로봇 영상)</td><td>3뷰 29프레임 → 24프레임 × 3청크</td><td class="num">15행 × 3 seed = 45</td><td>반동·반환은 그리지만 S와 무관(무작위), 미결 22–33%, 220 s/샘플</td><td><span class="chip wait">GO/NO-GO 결정 대기</span></td></tr>
    <tr><td>V-JEPA 2-AC</td><td>latent 예측 (ViT-g + action predictor)</td><td>2프레임 latent → 4스텝 latent, 액션 0</td><td class="num">29행 × 3오프셋 = 87 (+제어 134)</td><td>인코더 게이트 가까스로 통과, 예측기는 수동 물체 운동을 외삽하지 않음</td><td><span class="chip no">NO-GO · 운동 전개 없음</span></td></tr>
    <tr><td>DiLA</td><td>latent 동역학 (DINOv2-RAE, 구조/내용 분리, latent action)</td><td>2–5프레임 latent + action 정책(유지/0) → 4–16스텝 latent → RAE 디코드</td><td class="num">29행 × 3오프셋 × 8변형 + 누설 참조 2</td><td>past-only 예측 없음. 유지/0 action 롤아웃은 1–4스텝 안에 장면이 흐려지고 공이 사라짐(스텝 3 존재율 0), 물리/등속 선호 없음</td><td><span class="chip no">NO-GO</span></td></tr>
    <tr><td class="muted">AdaWorld · DreamDojo-Pretrain 2B · Cosmos-Predict2.5 base</td><td class="muted">latent-action 생성 · 액션 조건 생성 · 영상 이어 그리기</td><td class="muted">5-task 푸셔 씬(2026-08-26)</td><td class="num muted">스모크</td><td class="muted">액션 표현 게이트 실패 / 도메인 간극 / 단기만 물리적</td><td><span class="chip">이전 시도</span></td></tr>
  </table></div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 1 · 픽셀 생성</span><h2>Cosmos-Predict2-2B Video2World</h2></div><span class="chip no">문턱 미반영</span></div>
  <div class="pipe"><svg viewBox="0 0 1000 110" role="img" aria-label="Cosmos 파이프라인">
    <g font-family="Chakra Petch" font-size="13" font-weight="600" fill="var(--ink)">
      <rect x="10" y="26" width="170" height="54" rx="4" fill="var(--red-bg)" stroke="var(--red)"/><text x="95" y="49" text-anchor="middle">조건 5프레임</text>
      <rect x="230" y="26" width="200" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="330" y="49" text-anchor="middle">Wan VAE (시간 4×) + DiT</text>
      <rect x="480" y="26" width="170" height="54" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="565" y="49" text-anchor="middle">21프레임 영상</text>
      <rect x="700" y="26" width="290" height="54" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="845" y="49" text-anchor="middle">색 마스크 → 중심 궤적 → 방향·법칙</text>
    </g>
    <g font-family="Noto Sans KR" font-size="11.5" fill="var(--sub)"><text x="95" y="70" text-anchor="middle">= latent 2장, 중립 프롬프트</text><text x="330" y="70" text-anchor="middle">35 step, guidance 0, seed 24</text><text x="565" y="70" text-anchor="middle">1.31 s, 480×832</text><text x="845" y="70" text-anchor="middle">validity 게이트 · Wilson CI · (y, v²) · Σmv</text></g>
    <g stroke="var(--sub)" stroke-width="1.5" fill="var(--sub)"><line x1="180" y1="53" x2="222" y2="53"/><polygon points="222,48 230,53 222,58"/><line x1="430" y1="53" x2="472" y2="53"/><polygon points="472,48 480,53 472,58"/><line x1="650" y1="53" x2="692" y2="53"/><polygon points="692,48 700,53 692,58"/></g>
  </svg></div>
  <p>텍스트·영상 조건 diffusion 생성기. 조건 5프레임은 VAE 시간 압축 후 latent 두 장이라 속도 정보가 겨우 들어간다. 판독은 생성 영상에서 빨강·파랑 마스크로 공 중심을 추적해 방향과 법칙 잔차를 읽는다. 첫 파일럿에서 manifest 필드 충돌로 전 롤아웃이 이미지 조건이 됐던 사고를 잡아낸 뒤 재발사했다(REPLAY 검사: 조건 구간 픽셀 일치).</p>
  <h4>실제 예시 (seed 1)</h4>
  {cosmos_ex}
  <h4>결과</h4>
  <div class="tbl-wrap"><table><tr><th>씬</th><th class="num">비경계 정확도 [95% CI]</th><th class="num">등속 인코더</th><th class="num">미결</th><th class="num">strict valid</th></tr>{cosmos_rows}</table></div>
  <p class="small muted">P0 등속 제어(24 seed): {p0_line}. 저속은 과속, 고속은 감속 — 사건 없는 장면에서도 속도가 ±20% 틀린다(P1·P2 판독의 잡음 바닥). 프롬프트 대조군(정답/오답 서술 288 롤아웃): 세 가족 모두 |Δ| ≤ 0.02 — 텍스트는 방향 선택을 움직이지 못한다.</p>
  <div class="verdict"><b>판정.</b> 언덕은 S 전 구간 96–100% "넘어감"(관성 외삽), 두 공은 S와 무관하게 반전 10–33%(운동량 전달 붕괴), 진자만 등급형 전이가 있으나 문턱이 S 0.82로 이동. 외관을 작업대로 바꿔도 곡선은 CI 안에서 동일 — 실패는 외관이 아니라 생성기의 물리에 있다.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 2 · 로봇 영상 생성</span><h2>VERA DROID planner (Wan2.1-I2V-14B)</h2></div><span class="chip wait">GO/NO-GO 대기</span></div>
  <div class="pipe"><svg viewBox="0 0 1000 110" role="img" aria-label="VERA 파이프라인">
    <g font-family="Chakra Petch" font-size="13" font-weight="600" fill="var(--ink)">
      <rect x="10" y="26" width="200" height="54" rx="4" fill="var(--red-bg)" stroke="var(--red)"/><text x="110" y="49" text-anchor="middle">3뷰 캔버스 29프레임</text>
      <rect x="260" y="26" width="200" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="360" y="49" text-anchor="middle">Wan2.1-I2V-14B + 프롬프트</text>
      <rect x="510" y="26" width="170" height="54" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="595" y="49" text-anchor="middle">24프레임 × 3청크</text>
      <rect x="730" y="26" width="260" height="54" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="860" y="49" text-anchor="middle">측면 타일 재표본 → 동결 판독기</text>
    </g>
    <g font-family="Noto Sans KR" font-size="11.5" fill="var(--sub)"><text x="110" y="70" text-anchor="middle">576×128 (측면·3/4 상방·근상방)</text><text x="360" y="70" text-anchor="middle">40 step, VRAM 48 GB, 74 s/청크</text><text x="595" y="70" text-anchor="middle">autoregressive 0.72 s</text><text x="860" y="70" text-anchor="middle">192×128 → 16 fps 등가</text></g>
    <g stroke="var(--sub)" stroke-width="1.5" fill="var(--sub)"><line x1="210" y1="53" x2="252" y2="53"/><polygon points="252,48 260,53 252,58"/><line x1="460" y1="53" x2="502" y2="53"/><polygon points="502,48 510,53 502,58"/><line x1="680" y1="53" x2="722" y2="53"/><polygon points="722,48 730,53 722,58"/></g>
  </svg></div>
  <p>DROID 로봇 데이터로 학습된 비디오 planner. 문맥 29프레임(15 fps)이 필요한데 우리 씬은 사건 전 history가 0.31 s뿐이라 <b>100 Hz로 캡처해 시간을 늘렸다</b>(재생 시 6.7배 슬로모션). 그 대가로 뷰당 192×128이라 공이 5 px 남짓이고, 판독은 측면 타일만 16 fps 등가로 재표본해 같은 판독기를 쓴다.</p>
  <h4>실제 예시</h4>
  {vera_ex}
  <h4>결과 ({vera_seeds} seed × 15 샘플)</h4>
  <div class="tbl-wrap"><table><tr><th>가족</th><th class="num">n</th><th class="num">정확도(미결 포함)</th><th class="num">미결</th><th class="num">valid_frac</th></tr>{vera_rows}</table></div>
  <div class="verdict"><b>판정 보류.</b> Cosmos가 못 그리던 반동·반환 자체는 그린다(정성 GO). 그러나 반전 선택은 S와 무관해 무작위이고(S&lt;1에서도 반전 4/7), 미결 22–33%, 샘플당 220 s라 벤치마크 급 정량 평가에는 부적합(정량 NO-GO). 확정하려면 격자점당 n ≥ 8이 필요하다.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 3 · latent 예측</span><h2>V-JEPA 2-AC (ViT-g + action-conditioned predictor)</h2></div><span class="chip no">NO-GO · 운동 전개 없음</span></div>
  <div class="pipe"><svg viewBox="0 0 1000 150" role="img" aria-label="V-JEPA latent 트랙 파이프라인">
    <g font-family="Chakra Petch" font-size="13" font-weight="600" fill="var(--ink)">
      <rect x="10" y="20" width="170" height="54" rx="4" fill="var(--red-bg)" stroke="var(--red)"/><text x="95" y="43" text-anchor="middle">문맥 2프레임 (4 fps)</text>
      <rect x="230" y="20" width="170" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="315" y="43" text-anchor="middle">ViT-g 인코더</text>
      <rect x="450" y="20" width="200" height="54" rx="4" fill="var(--panel2)" stroke="var(--line)"/><text x="550" y="43" text-anchor="middle">predictor × 4 (액션 0)</text>
      <rect x="700" y="20" width="290" height="54" rx="4" fill="var(--blue-bg)" stroke="var(--blue)"/><text x="845" y="43" text-anchor="middle">ẑ₁…ẑ₄  (256 토큰 × 1408)</text>
      <rect x="230" y="92" width="420" height="46" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="440" y="112" text-anchor="middle">MuJoCo 후보 미래 P · K · J → 같은 인코더 → z_P, z_K, z_J</text>
      <rect x="700" y="92" width="290" height="46" rx="4" fill="var(--green-bg)" stroke="var(--green)"/><text x="845" y="112" text-anchor="middle">dcos = cos(Δẑ, ΔP) − cos(Δẑ, ΔK)</text>
    </g>
    <g font-family="Noto Sans KR" font-size="11.5" fill="var(--sub)"><text x="95" y="64" text-anchor="middle">256², 프레임 0·4 (0.25 s 간격)</text><text x="315" y="64" text-anchor="middle">프레임당 2프레임 튜블릿</text><text x="550" y="64" text-anchor="middle">EE state 상수, 각 스텝 0.25 s</text><text x="845" y="64" text-anchor="middle">layer-norm latent</text><text x="440" y="130" text-anchor="middle">게이트: d(z_P, z_K) &gt; 3 × d(z_P, z_J)</text><text x="845" y="130" text-anchor="middle">오라클 ±0.45 · freeze 중립</text></g>
    <g stroke="var(--sub)" stroke-width="1.5" fill="var(--sub)"><line x1="180" y1="47" x2="222" y2="47"/><polygon points="222,42 230,47 222,52"/><line x1="400" y1="47" x2="442" y2="47"/><polygon points="442,42 450,47 442,52"/><line x1="650" y1="47" x2="692" y2="47"/><polygon points="692,42 700,47 692,52"/><line x1="845" y1="74" x2="845" y2="84"/><polygon points="840,84 845,92 850,84"/><line x1="650" y1="115" x2="692" y2="115"/><polygon points="692,110 700,115 692,120"/></g>
  </svg></div>
  <p>픽셀을 그리지 않는 예측기. 인코더 latent에서 미래 latent를 자기회귀로 예측하므로 판독도 latent 공간에서 한다: 같은 시각의 물리 미래 P와 등속 counterfactual K를 MuJoCo로 렌더해 인코딩하고, 예측이 어느 쪽으로 움직였는지 변위 방향으로 잰다. 로봇 액션은 0, 문맥은 사건 전 두 프레임 — 누출이 구조적으로 없다.</p>
  <h4>실제 예시</h4>
  {"<div class='media'><img src='" + vj_prev + "' alt='후보 미래 렌더'></div><p class='cap'>A1 작업대 S=0.95. 위: 문맥 2프레임 + 물리 미래 P · 가운데: 등속 K · 아래: 1 px 지터 J. 예측기는 위 두 프레임의 latent만 받는다.</p>" if vj_prev else ""}
  {"<div class='media'><img src='" + vj_curves + "' alt='latent 트랙 곡선'></div><p class='cap'>위: 마지막 스텝 거리차 Δ/sep — 예측기(빨강)는 freeze 기준선(회색)의 축소판. 가운데: 분리도 sep/floor와 게이트선. 아래: 변위 방향 dcos(전체 토큰 ●, P≠K 패치 ▲)와 오라클 범위(띠). 어느 S에서도 0 근처.</p>" if vj_curves else ""}
  <h4>결과</h4>
  <div class="tbl-wrap"><table><tr><th>씬</th><th class="num">게이트 통과</th><th class="num">sep/floor</th><th class="num">cos_P 스텝 1/2/3/4</th><th class="num">문맥 반전 시 cos_P</th><th class="num">dcos (게이트)</th></tr>{vj_rows}</table></div>
  <div class="verdict"><b>판정 NO-GO (모델 원인).</b> 인코더는 두 미래를 1 px 잡음의 2–4.5배로 겨우 구분한다(공 11–17 px). 예측기의 변위 정렬(cos ≈ 0.2)은 문맥 순서를 뒤집어도, 문맥을 3프레임으로 늘려도, 속도를 바꿔도 그대로 — 운동이 아니라 정적 성분에서 오는 정렬이다. 물리·등속 미래 선호 차 |dcos| ≤ 0.05(오라클 ±0.45), S 무관, EE state 무시. DROID 로봇 운동으로 post-train된 AC predictor는 액션이 0인 장면에서 dynamics를 전개하지 않는다.</div>
</section>

{dila_html}
<section class="block">
  <div class="model-head"><div><span class="eyebrow">이전 시도 · 2026-08-26 · 5-task 푸셔 씬</span><h2>액션 표현 게이트에서 막힌 모델들</h2></div><span class="chip">규칙: 이식된 액션이 우리 장면에서 전개돼야 물리 점수를 보고한다</span></div>
  <div class="grid3">
    <figure class="thumb">{"<img src='" + ada + "' alt='AdaWorld 스모크'>" if ada else ""}<figcaption><b>AdaWorld</b><span class="chip tiny no">게이트 실패</span><span class="small muted">latent action이 전역 이동에만 민감 — 예측이 푸셔를 지우고 큐브가 정지.</span></figcaption></figure>
    <figure class="thumb">{"<img src='" + dd + "' alt='DreamDojo 스모크'>" if dd else ""}<figcaption><b>DreamDojo-Pretrain 2B</b><span class="chip tiny no">게이트 실패</span><span class="small muted">in-domain GR-1 클립에서는 액션을 따르지만(PSNR 23.0 vs 19.1) 우리 씬에서는 그리퍼를 사람 손으로 바꿔 그리고 물체는 정지.</span></figcaption></figure>
    <figure class="thumb">{"<img src='" + c25 + "' alt='Cosmos-Predict2.5 base 이어 그리기'>" if c25 else ""}<figcaption><b>Cosmos-Predict2.5-2B base</b><span class="chip tiny wait">단기만 물리적</span><span class="small muted">문맥 5프레임 이어 그리기: 첫 5프레임 오차 10.3 px &lt; freeze 16.8, 이후 드리프트·seed 의존. 이 경험이 Bundle A 설계로 이어졌다.</span></figcaption></figure>
  </div>
</section>
<footer>생성: <span class="mono">SimDROID/code_vwm</span> · 씬 <span class="mono">core/threshold/*</span> · Cosmos <span class="mono">exp/cosmos_v2w_sweep.py</span> · VERA <span class="mono">external/vera_infer.py</span> · V-JEPA <span class="mono">exp/rollout_vjepa_ac.py</span> · 판독 <span class="mono">exp/analyze_bundle_a.py · exp/analyze_vjepa_ac.py</span> · 정본 기록 <span class="mono">SimDROID/PROGRESS.md (35)–(41)</span> · 2026-09-03 · 문지훈/Claude</footer>
</main>
<script>
(function(){{
  var tabs=Array.prototype.slice.call(document.querySelectorAll('nav.tabs [role=tab]'));
  function show(id){{tabs.forEach(function(b){{var on=b.id===id;b.setAttribute('aria-selected',on?'true':'false');document.getElementById(b.getAttribute('aria-controls')).hidden=!on;}});try{{localStorage.setItem('pg_tab',id);}}catch(e){{}}}}
  tabs.forEach(function(b){{b.addEventListener('click',function(){{show(b.id);window.scrollTo({{top:0}});}});b.addEventListener('keydown',function(e){{var i=tabs.indexOf(b);if(e.key==='ArrowRight'){{tabs[(i+1)%tabs.length].focus();show(tabs[(i+1)%tabs.length].id);}}if(e.key==='ArrowLeft'){{tabs[(i+tabs.length-1)%tabs.length].focus();show(tabs[(i+tabs.length-1)%tabs.length].id);}}}});}});
  var saved=null;try{{saved=localStorage.getItem('pg_tab');}}catch(e){{}}
  if(saved&&document.getElementById(saved))show(saved);
  var gifs=Array.prototype.slice.call(document.querySelectorAll('img.gif')),btn=document.getElementById('motion'),still=false;
  function setStill(v){{still=v;gifs.forEach(function(g){{g.src=v?g.getAttribute('data-still'):g.getAttribute('data-anim');}});btn.textContent=v?'GIF 재생':'GIF 정지';btn.setAttribute('aria-pressed',v?'true':'false');}}
  btn.addEventListener('click',function(){{setStill(!still);}});
  if(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches)setStill(true);
}})();
</script>
"""
    Path(dst).write_text(html)
    print(f"OK -> {dst} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
