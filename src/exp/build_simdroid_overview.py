"""Build the SimDROID scene + model overview page (single HTML, inline assets).

Four tabs: overview (question, timeline, scene-generation x model matrix, pending decisions),
scenes (G1 pusher 5-task .. G5 Phase B, with verified strips/previews/GIFs), models (one block per
evaluated or assessed model: contract, scenes, numbers, verdict, failure location), verdicts/next.
Numbers come from artifacts/*/summary.json where they exist; everything else is copied from
PROGRESS.md (35)-(43) and HANDOFF 3.6-3.9 with the entry cited.
Usage: PYTHONPATH=src python src/exp/build_simdroid_overview.py <dst.html>
"""
from __future__ import annotations

import os

import base64, io, json, math, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("VWM_ARTIFACTS", "artifacts"))
CODE_OUT = Path(os.environ.get("SIMDROID_CODE_OUT", "../code/out"))


# ---------------------------------------------------------------- assets
def b64_img(path: Path, max_w: int = 1000, quality: int = 76) -> str | None:
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_array(arr, quality=80):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def small_gif(path: Path, max_w: int = 640, step: int = 2):
    """(animated gif data uri, first-frame jpeg uri) re-encoded smaller; (None, None) if missing."""
    if not path.exists():
        return None, None
    im = Image.open(path)
    frames, k = [], 0
    try:
        while True:
            if k % step == 0:
                f = im.convert("RGB")
                if f.width > max_w:
                    f = f.resize((max_w, round(f.height * max_w / f.width)), Image.LANCZOS)
                frames.append(f)
            k += 1; im.seek(im.tell() + 1)
    except EOFError:
        pass
    still = b64_array(np.asarray(frames[0]))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=62 * step, loop=0, optimize=True)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode(), still


def vera_strip(sid: str, seed: int = 1):
    gp, rp = ROOT / "phase_a/vera" / f"{sid}.npz", ROOT / f"phase_a/vera_out/seed_{seed:02d}" / sid / "rollout.npz"
    if not (gp.exists() and rp.exists()):
        return None
    gt = np.load(gp)["side_view"]; gen = np.load(rp)["side_view"]
    idx = [28, 40, 52, 64, 76, 88, 100]
    row_gt = np.concatenate([gt[min(i, len(gt) - 1)] for i in idx], axis=1)
    row_gen = np.concatenate([gen[min(i, len(gen) - 1)] for i in idx], axis=1)
    arr = np.concatenate([row_gt, row_gen], axis=0)
    im = Image.fromarray(arr); im = im.resize((im.width * 2, im.height * 2), Image.NEAREST)
    return b64_array(np.asarray(im))


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x, nd=0):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}%"


def img(uri, alt, cls="media"):
    return f'<div class="{cls}"><img src="{uri}" alt="{alt}" loading="lazy"></div>' if uri else ""


def gif_html(anim, still, alt):
    if not anim:
        return ""
    return f'<div class="media gifwrap"><img class="gif" src="{anim}" data-anim="{anim}" data-still="{still}" alt="{alt}"></div>'


def dl(rows):
    return '<dl class="spec">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def chip(cls, txt):
    return f'<span class="chip {cls}">{txt}</span>'


def table(head, rows, cls=""):
    th = "".join(f'<th{" class=num" if h.startswith("#") else ""}>{h.lstrip("#")}</th>' for h in head)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f'<td{" class=num" if isinstance(c, tuple) else ""}>{c[0] if isinstance(c, tuple) else c}</td>' for c in r) + "</tr>"
    return f'<div class="tbl-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def N(x):  # numeric cell marker
    return (x,)


# ---------------------------------------------------------------- data
def vera_curve(summ, fam):
    """Per-S decided counts from a VERA summary -> (rows, acc_nb, ci, acc_dec, ci_dec, n_nb, n_dec, lo_rate, hi_rate)."""
    by = {}
    for r in summ[fam]["per_sample"]:
        b = by.setdefault(r["S"], [0, 0, 0])
        b[0 if r["pred"] == 1 else 1 if r["pred"] == 0 else 2] += 1
    rows = []; nb_c = nb_n = d_c = d_n = 0; lo1 = lo = hi1 = hi = 0
    for S in sorted(by):
        d1, d0, u = by[S]; gt = 1 if S > 1 else 0
        p = d1 / (d1 + d0) if d1 + d0 else float("nan")
        rows.append((S, d1, d0, u, p, gt))
        if abs(S - 1) > 0.04:
            nb_n += d1 + d0 + u; nb_c += d1 if gt else d0; d_n += d1 + d0; d_c += d1 if gt else d0
            if S < 1: lo1 += d1; lo += d1 + d0
            else: hi1 += d1; hi += d1 + d0
    a, ad = nb_c / nb_n, d_c / d_n
    return rows, a, wilson(a, nb_n), ad, wilson(ad, d_n), nb_n, d_n, lo1 / max(lo, 1), hi1 / max(hi, 1)


def main(dst):
    toy = load_json(ROOT / "bundle_a/analysis/summary.json")
    wb = load_json(ROOT / "phase_a/analysis/summary.json")
    pb = load_json(ROOT / "phase_b/analysis_pilot/summary.json")
    vera8 = load_json(ROOT / "phase_a/analysis_vera_s8/summary.json")
    vj = load_json(ROOT / "vjepa_ac/phase_a/analysis/summary.json")
    vj_rev = load_json(ROOT / "vjepa_ac/phase_a/analysis_rev/summary.json")
    vjb = load_json(ROOT / "vjepa_ac/phase_b/analysis/summary.json")
    verify_a = load_json(ROOT / "bundle_a/verify.json") or {}
    verify_b = load_json(ROOT / "phase_b/verify.json") or {}
    lib = load_json(ROOT / "scene_library/report.json") or {}
    pusher = load_json(ROOT / "scene_library/pusher5/report.json") or {}
    dila_px = {t: load_json(ROOT / f"dila/phase_a/pixels{t}/summary.json") for t in ("", "_zero", "_s1", "_oracle")}
    dila_pb = load_json(ROOT / "dila/phase_b/pixels_zero/summary.json")

    def acc(summ, fam, n):
        a = summ[fam]["accuracy_nonboundary"]; lo, hi = wilson(a, n)
        return f'{pct(a)} <span class="muted small">[{pct(lo)}, {pct(hi)}]</span>'

    # ------------------------------------------------ G1 pusher strips
    G1 = [
        ("slide", "mu", "밀린 큐브가 미끄러져 멈춤", "질량 · 미끄럼 마찰 μ", "μ 0.15 → 0.37: 정지 거리 11.4 cm 차", "22 / 37"),
        ("roll", "roll_fric", "밀린 구가 굴러감", "질량 · 굴림저항", "굴림저항 0.0006 → 0.0040: 6.0 cm 차", "9 / 23"),
        ("bounce", "damping", "큐브를 떨어뜨림", "접촉 감쇠 · 질량(판별 불가)", "감쇠 13 → 134: 튀는 높이 9.4 cm 차", "73 / 50"),
        ("collide", "mass2", "큐브가 두 번째 큐브를 침", "두 번째 큐브 질량 · 공통 μ", "mass2 0.11 → 0.44 kg: 2.7 cm 차 (약한 셀)", "9 / 10"),
        ("incline", "mu", "경사면에서 놓은 큐브", "μ · 질량(판별 불가)", "μ 0.14 → 0.35: 22.5 cm 차 (T=13)", "26 / 60"),
    ]
    g1_cards = ""
    for fam, param, what, props, pair_txt, gap in G1:
        uri = b64_img(ROOT / "scene_library/pusher5" / f"{fam}_{param}_pair.jpg", max_w=1216, quality=80)
        rep = pusher.get(f"{fam}/{param}", {})
        n = rep.get("n_frames", "?")
        g1_cards += f"""<article class="card scene">
  <div class="card-head"><div><span class="eyebrow">G1 · {fam}</span><h3>{what}</h3></div><div class="chips">{chip("cls", props)}</div></div>
  {img(uri, f"{fam} lo/hi 쌍")}
  <p class="cap">위 = 낮은 분위(15%), 아래 = 높은 분위(85%) — <b>{param}</b>만 다르고 나머지는 동일. 640×480 · 50 Hz · {n}프레임 중 5장.</p>
  {dl([("쌍 대조", pair_txt), ("GT 분기 (px)", f"중심 격차 @중간/@끝 = {gap}"), ("데이터", "train 250 + test 80×3 · 쌍 30/셀 · full state 8열")])}
</article>
"""
    gt_gap_rows = [
        ("slide / mass", N("428"), N("22 / 37"), N("12.3"), N("0.85")), ("slide / mu", N("393"), N("14 / 34"), N("11.4"), N("0.95")),
        ("roll / mass", N("634"), N("11 / 21"), N("5.4"), N("0.94")), ("roll / roll_fric", N("261"), N("9 / 23"), N("6.0"), N("0.99")),
        ("bounce / mass " + chip("no tiny", "판별 불가"), N("0"), N("0 / 0"), N("0"), N("1.00")), ("bounce / damping", N("171"), N("73 / 50"), N("9.4"), N("1.00")),
        ("collide / mass2 " + chip("wait tiny", "약함"), N("136"), N("9 / 10"), N("2.7"), N("1.00")), ("collide / mu " + chip("wait tiny", "약함"), N("587"), N("7 / 9"), N("2.1"), N("0.91")),
        ("incline / mass " + chip("no tiny", "판별 불가"), N("0"), N("0 / 0"), N("0"), N("1.00")), ("incline / mu (T=13)", N("118"), N("26 / 60"), N("22.5"), N("1.00")),
    ]

    # ------------------------------------------------ G2 early scenes
    EARLY = [("seesaw", "시소", "정적 · P4", "가시 토크 균형 → 왼쪽/오른쪽 기움"), ("lean", "기대선 막대", "정적 · P4", "막대 각도·고정 마찰 → 버팀/미끄러짐"),
             ("tower", "3단 탑", "정적 · P4", "누적 무게중심 → 서 있음/무너짐 (o1 고정이라 맨 위 큐브 문제로 축소)"),
             ("hill", "가우시안 언덕", "동적 → A1 전신", "속도·높이 → 반환/통과 (+0.32 s 비교 시점에 결말 미포함)"),
             ("collide", "두 공 (감쇠 스윕)", "동적 → A2 전신", "스윕 축(반발계수)이 조건 5프레임에서 비가시 — 7점 입력 픽셀 동일"),
             ("domino", "도미노", "동적 · 보류", "간격 → 전파 정지/도달 (좁은 2점은 첫 전도가 조건 창 안 = 누설)"),
             ("energy_hill", "원호 위 구슬", "에너지 묶음", "E_avail/E_req − 1, ±0.05–0.30 (|m| ≥ 0.2는 말기에 호 밖)"),
             ("energy_ramp", "경사로·게이트", "에너지 묶음 · P3 후보", "½v² ≥ gh (±0.05는 평가 시점에 미해결)"),
             ("energy_pendulum", "진자 (초기)", "에너지 묶음 → A3 전신", "무감쇠 연속 회전이라 마지막 프레임 각도 앨리어싱")]
    g2_thumbs = ""
    for key, lab, tag, note in EARLY:
        b = b64_img(ROOT / "scene_library" / f"{key}.png", max_w=560, quality=72)
        if not b:
            continue
        info = lib.get(key, {})
        sub = f'{info.get("n_frames", "?")}프레임 · margin {info.get("margin", float("nan")):+.2f} · 사건 프레임 {info.get("event_frame", "?")}' if "n_frames" in info else ""
        g2_thumbs += f'<figure class="thumb"><img src="{b}" alt="{lab}" loading="lazy"><figcaption><b>{lab}</b>{chip("tiny", tag)}<span class="small">{note}</span><span class="small muted mono">{sub}</span></figcaption></figure>'

    # ------------------------------------------------ G3–G5 cards
    def bis(fam):
        b = verify_a.get(fam, {}).get("bisection")
        return f'시뮬 경계 S* = {b["S_star_sim"]:.3f} (폐형식 1 대비 {b["gap_percent"]:.1f}%, 격자 한 칸 {b["grid_cell_percent"]:.0f}%)' if b else "—"

    def scene_card(gen, title, cls, formula, rule, preview, gif, spec, extra=""):
        prev = b64_img(preview, max_w=1100, quality=72) if preview else None
        anim, still = small_gif(gif) if gif else (None, None)
        media = ""
        if prev:
            media += img(prev, f"{title} S 격자 프리뷰") + '<p class="cap">열 = S 격자 11점 (0.50 → 2.00), 행 = t=0 / 조건 끝 0.25 s / +0.31 s / 사건 / 1.25 s. 위 라벨 y = GT 결과.</p>'
        if anim:
            media += gif_html(anim, still, f"{title} GT 쌍") + '<p class="cap">GT 21프레임 · 왼쪽 S &lt; 1, 오른쪽 S &gt; 1 — 결정 변수 하나만 다르다.</p>'
        return f"""<article class="card scene">
  <div class="card-head"><div><span class="eyebrow">{gen}</span><h3>{title}</h3></div><div class="chips">{chip("cls", cls)}</div></div>
  <p class="formula mono">{formula}</p><p>{rule}</p>{media}{dl(spec)}{extra}
</article>
"""
    g3 = (scene_card("G3 · A1", "구름 언덕", "P1-A 중력 · 연속", "S = 0.7·v₀² / (g·h)",
                     "S &gt; 1이면 정점을 넘고(1), 아니면 돌아온다(0). 구름 운동: 7/10 v₀² = 7/10 v² + g·y.",
                     ROOT / "bundle_a/previews/hill_roll.png", ROOT / "bundle_a/gifs/hill_roll_pair.gif",
                     [("결정 변수", "언덕 높이 h (v₀ = 1.0 m/s, 폭 0.30 m, sin² 접선 연속)"), ("사건", "정점 통과/반환 0.6–0.9 s (프레임 10–15)"),
                      ("판독", "빨간 공 중심 궤적 → 방향, (y, v²) 기울기 (구름 −14.0 m/s² 기대)"), ("검증", bis("hill_roll") + " · dt 0.5/1 ms 일치 · μ·밀도 불변")])
          + scene_card("G3 · A2", "두 공 정면충돌", "P2-B 충격 · 운동량", "S = (r₂ / r₁)³ (등밀도 → 질량비)",
                       "e ≈ 1에서 m₁ &lt; m₂, 즉 S &gt; 1이면 빨간 공이 되돌아온다(1). 바닥은 근-무마찰이라 Σmv를 영상에서 감사할 수 있다.",
                       ROOT / "bundle_a/previews/two_ball.png", ROOT / "bundle_a/gifs/two_ball_pair.gif",
                       [("결정 변수", "파란 공 반지름 r₂ (r₁ = 4 cm, v₀ = 1.0 m/s)"), ("사건", "접촉 0.44 s (프레임 7), 판독 창 충돌 직후 2–3프레임"),
                        ("판독", "충돌 후 빨간 공 방향 + Σmv 비 (GT 0.98–1.00)"), ("검증", bis("two_ball") + " · e_eff = 0.996 · 감쇠·밀도 불변")])
          + scene_card("G3 · A3", "막대 진자", "P1-B 구속 보존", "S = v₀² / (4·g·L)",
                       "S &gt; 1이면 정점을 넘어 회전(1), 아니면 되돌아온다(0). 언덕과 같은 에너지 법칙, 다른 메커니즘.",
                       ROOT / "bundle_a/previews/pendulum_rod.png", ROOT / "bundle_a/gifs/pendulum_rod_pair.gif",
                       [("결정 변수", "팔 길이 L (v₀ = 3.6 m/s — 공통 1.0이면 임계 L 2.5 cm로 비가시)"), ("사건", "정점 통과/반환 0.4–1.1 s"),
                        ("판독", "봅 각도 궤적 → 회전/반환, 에너지 기울기"), ("검증", bis("pendulum_rod") + " · 에너지 드리프트 ≤ 0.08%")]))
    pre_a, pre_a_still = small_gif(ROOT / "bundle_a/gifs/hill_roll_pre.gif")
    pre_b, pre_b_still = small_gif(ROOT / "bundle_a/gifs/wall_bounce_pre.gif")
    g3_pre = (f'<figure class="pre">{gif_html(pre_a, pre_a_still, "A1-0 감속 사전검사")}<figcaption><b>A1-0 감속 사전검사 (S = 0.35)</b> — 확실히 되돌아오는 낮은 에너지. 관성 외삽만 하는 모델은 여기서도 넘어감을 그린다 (Cosmos 20/24 넘어감).</figcaption></figure>'
              f'<figure class="pre">{gif_html(pre_b, pre_b_still, "A2-0 벽 반동 사전검사")}<figcaption><b>A2-0 벽 반동 사전검사</b> — 질량비를 읽기 전에 "접촉 후 방향이 바뀐다"를 그릴 수 있는지 (Cosmos toy 5/24, 작업대 11/24, VERA 3/7).</figcaption></figure>')
    g4 = (scene_card("G4 · P0", "등속 구름 (제어)", "P0 등속 연장", "x(t) = x₀ + v₀·t",
                     "사건이 없다. 등속 외삽이 정답이므로 생성 실패(소실·드리프트·감속)와 물리 실패를 가른다.",
                     ROOT / "phase_a/previews/kin_roll.png", None,
                     [("스윕", "v₀ ∈ {0.30, 0.45, 0.60, 0.75, 0.90} m/s · 평탄 검사대"), ("판독", "생성 클립의 조건 프레임으로 등속 외삽한 궤적과의 오차, 변위비 gen/GT"),
                      ("검증", "속도비 1.000 · slip 2e-7 · 에너지 드리프트 &lt; 0.04%")])
          + scene_card("G4 · A1", "언덕 · 작업대", "P1-A 중력 · 연속", "S = 0.7·v₀² / (g·h)",
                       "초록 케이블커버 hump. 옵션·접촉쌍·충돌 geom·hfield가 toy와 컴파일 수준에서 동일 — 순수 외관 대조.",
                       ROOT / "phase_a/previews/hill_roll_wb.png", None,
                       [("외관", "회색 라미네이트 검사대 · 뒷벽 · 정지 Franka Panda(충돌 없음) · 트레이·픽스처·빈 · 테이프 눈금"),
                        ("검증", "toy↔작업대 물리 서명 identical · 격자 게이트 11/11 · 사건 프레임 동일")])
          + scene_card("G4 · A2", "두 공 · 작업대", "P2-B 충격 · 운동량", "S = (r₂ / r₁)³",
                       "같은 재질의 두 부품이 저마찰 검사대 위에서 정면충돌. 벽 반동 사전검사는 steel 스토퍼 블록.",
                       ROOT / "phase_a/previews/two_ball_wb.png", None,
                       [("외관", "작업대 wrapper 공통 + 눈금 위치만 조정"), ("검증", "물리 서명 identical(접촉쌍 3) · 격자 게이트 11/11")]))
    sb = verify_b.get("support_edge_boundary", {})
    g5 = (scene_card("G5 · B1", "매달린 payload (A3 재스킨)", "P1-B 구속 보존", "S = v₀²(1 + 0.4r²/L²) / (4·g·L)",
                     "steel 포스트에 봉으로 매달린 붉은 payload가 최저점을 지나 올라간다. 관절·질량·타임스텝이 toy A3와 같아 궤적 차 0.",
                     ROOT / "phase_b/previews/pendulum_rod_wb.png", ROOT / "phase_b/gifs/pendulum_rod_wb_pair.gif",
                     [("결정 변수", "봉 길이 L (v₀ = 3.6 m/s) — S 0.5–2.0 ↔ L 0.66–0.17 m"), ("사건", "프레임 6 (S=2.0) – 12 (S ≤ 0.95)"),
                      ("판독", "payload 궤적 원 적합 → 피벗·각도, 에너지 기울기 (toy와 동일 판독기)"),
                      ("검증", "toy↔작업대 서명 identical · 궤적 차 0.0 · GT 자가검증 11/11")],
                     '<p class="note"><b>피드백 항목.</b> 파일럿에서 strict valid 0.30·미결 21% — payload가 봉·포스트·Franka와 겹쳐 마스크 면적 게이트를 자주 넘긴다. (a) payload 색·크기 또는 포스트 위치 조정 / (b) 면적 게이트 완화 중 결정 필요.</p>')
          + scene_card("G5 · B2", "지지 끝 이탈 (신규)", "P3-A 접촉 상실", f"S = v₀·T_REF / (x_edge − x_start), T_REF = {sb.get('T_REF', 1.176)} s",
                       "S &gt; 1이면 지평 안에 steel 플레이트 끝을 지나 검사대로 떨어진다(1), 아니면 끝까지 지지된다(0). 등속 외삽은 끝을 지나서도 같은 높이로 직진하는 <b>공중 부양</b>을 예측한다.",
                       ROOT / "phase_b/previews/support_edge_wb.png", ROOT / "phase_b/gifs/support_edge_wb_pair.gif",
                       [("결정 변수", "플레이트 끝 위치 x_edge (v₀ = 0.35 m/s, 플레이트 높이 0.15 m, 공 6 cm) — 정적 가시"),
                        ("사건", "S=2.0 프레임 11 · 1.26 → 17 · 1.12 → 18; S ≤ 0.95는 지평 끝(20)까지 지지"),
                        ("판독", "빨간 공 중심 행이 문맥 수준보다 0.6 지름 이상 하강 = 낙하; 낙하 초기 가속도/g"),
                        ("검증", f"시뮬 이분법으로 T_REF 보정 → 경계 S = {sb.get('S_boundary_sim', 0.9998):.4f} · 끝 도달 전 속도비 1.0000 · GT 자가검증 10/11 (+S=1.00 나이프에지)")]))

    # ------------------------------------------------ model example images
    ada = b64_img(CODE_OUT / "rollout_adaworld/viz_s2_lo2hi.png", max_w=1000, quality=74)
    dd_hand = b64_img(CODE_OUT / "rollout_dreamdojo/hand_slide_mu_s3/zoom_hi2lo.png", max_w=1000, quality=74)
    dd_rod = b64_img(CODE_OUT / "rollout_dreamdojo/smoke_slide_mu_s3/zoom_lo2hi.png", max_w=1000, quality=74)
    hand_cmp = b64_img(CODE_OUT / "hand_look/slide_lo_rod_vs_hand.png", max_w=900, quality=74)
    c25 = b64_img(CODE_OUT / "rollout_cosmos/smoke_slide_mu_s3_t24_neutral/zoom_ctx_hi_neutral.png", max_w=1000, quality=74)
    curves_toy = b64_img(ROOT / "bundle_a/analysis/curves.png", max_w=1000, quality=78)
    curves_wb = b64_img(ROOT / "phase_a/analysis/curves.png", max_w=1000, quality=78)
    curves_pb = b64_img(ROOT / "phase_b/analysis_pilot/curves.png", max_w=1000, quality=78)
    vj_prev = b64_img(ROOT / "vjepa_ac/phase_a/previews/hill_roll_wb_04.png", max_w=1000, quality=74)
    vj_curves = b64_img(ROOT / "vjepa_ac/phase_a/analysis/curves.png", max_w=1000, quality=76)
    dila_ex = b64_img(ROOT / "dila/phase_a/previews/examples_s1.png", max_w=1100, quality=74)
    cosmos_ex = ""
    for sid, root, lab in (("hill_roll_wb_02", "phase_a", "A1 작업대 S=0.79 — GT 반환, 생성은 넘어감"), ("two_ball_wb_08", "phase_a", "A2 작업대 S=1.26 — GT 반전, 생성은 정체·직진"),
                           ("wall_bounce_wb_pre", "phase_a", "A2-0 작업대 벽 반동 — GT 반동"), ("kin_roll_02", "phase_a", "P0 v₀=0.60 — 등속"),
                           ("support_edge_wb_10", "phase_b", "B2 지지 끝 S=2.00 (파일럿) — GT 낙하, 생성은 플레이트 높이로 '공중 부양'"),
                           ("pendulum_rod_wb_08", "phase_b", "B1 매달린 payload S=1.26 (파일럿) — GT 회전")):
        b = b64_img(ROOT / root / "cosmos_v2w/seed_01" / sid / "comparison.png", max_w=896, quality=74)
        if b:
            cosmos_ex += f'<figure class="ex"><img src="{b}" alt="{lab}" loading="lazy"><figcaption>{lab} <span class="muted">(seed 1: 현재 | 물리 GT +0.32 s | 생성 +0.31 s | 생성 마지막 1.25 s)</span></figcaption></figure>'
    vera_ex = ""
    for sid, lab in (("hill_roll_wb_02", "A1 S=0.79 (GT 반환)"), ("two_ball_wb_08", "A2 S=1.26 (GT 반전)")):
        b = vera_strip(sid)
        if b:
            vera_ex += f'<figure class="ex"><div class="media scroll"><img src="{b}" alt="{lab} VERA" loading="lazy"></div><figcaption>{lab} — 위 GT, 아래 VERA (seed 1). 열 = 100 Hz 캡처 프레임 28(문맥 끝)·40·52·64·76·88·100. 측면 타일 192×128.</figcaption></figure>'

    # ------------------------------------------------ model numbers
    cosmos_rows = [
        ("A1 언덕 · toy", N(acc(toy, "hill_roll", 240)), N(pct(toy["hill_roll"]["undecided_rate"])), N(pct(toy["hill_roll"]["valid_rate"])), "p(넘어감) 0.91–1.00 전 구간 평탄 · 사전검사 S=0.35도 20/24 넘어감"),
        ("A3 진자 · toy", N(acc(toy, "pendulum_rod", 240)), N(pct(toy["pendulum_rod"]["undecided_rate"])), N(pct(toy["pendulum_rod"]["valid_rate"])), "등급형 S자 · 로지스틱 S50 = 0.82 (GT 1.0) · 결정된 것만 0.75"),
        ("A2 두 공 · toy", N(acc(toy, "two_ball", 240)), N(pct(toy["two_ball"]["undecided_rate"])), N(pct(toy["two_ball"]["valid_rate"])), f'p(반전) 0.11–0.20 평탄 · Σmv 비 {toy["two_ball"]["momentum_ratio_mean"]:.2f} (GT 0.98–1.00) · 벽 반동 5/24'),
        ("A1 언덕 · 작업대", N(acc(wb, "hill_roll_wb", 240)), N(pct(wb["hill_roll_wb"]["undecided_rate"])), N(pct(wb["hill_roll_wb"]["valid_rate"])), "p(넘어감) 0.96–1.00 · 사전검사 23/24 넘어감 · (y,v²) 기울기 −7.8 vs GT −13.6"),
        ("A2 두 공 · 작업대", N(acc(wb, "two_ball_wb", 240)), N(pct(wb["two_ball_wb"]["undecided_rate"])), N(pct(wb["two_ball_wb"]["valid_rate"])), f'p(반전) 0.10–0.33 평탄 · Σmv 비 {wb["two_ball_wb"]["momentum_ratio_mean"]:.2f} · 벽 반동 11/24'),
    ]
    if pb:
        cosmos_rows += [
            ("B2 지지 끝 · 작업대 (파일럿 3 seed)", N(acc(pb, "support_edge_wb", 30)), N(pct(pb["support_edge_wb"]["undecided_rate"])), N(pct(pb["support_edge_wb"]["valid_rate"])), "S ≤ 1.12 전부 '안 떨어짐' · S ≥ 1.26 각 1/3 — 비경계 S&gt;1 12건 중 9건이 공중 직진(부양)"),
            ("B1 payload · 작업대 (파일럿 3 seed)", N(acc(pb, "pendulum_rod_wb", 30)), N(pct(pb["pendulum_rod_wb"]["undecided_rate"])), N(pct(pb["pendulum_rod_wb"]["valid_rate"])), f'S ≥ 0.89에서 회전 선호(등급형) · 에너지 기울기 GT {pb["pendulum_rod_wb"]["gt_law_slope_mean"]:.1f} vs 생성 {pb["pendulum_rod_wb"]["gen_law_slope_mean"]:.1f} · 판독 약함'),
        ]
    ctrl = wb.get("controls", {}).get("kin_roll", {})
    p0_line = " · ".join(f'v₀ {s["v0"]:.2f}: 변위비 {s["speed_ratio"]:.2f}, 오차 {s["kin_err_px"]:.0f} px' for s in ctrl.get("per_sample", []))

    # VERA seed 1-8
    vera_rows, vera_tbl_S = [], ""
    vera_verdict_nums = {}
    if vera8:
        for fam, lab in (("hill_roll_wb", "A1 언덕 · 작업대"), ("two_ball_wb", "A2 두 공 · 작업대")):
            rows, a, (lo, hi), ad, (lo2, hi2), n_nb, n_dec, lo_rate, hi_rate = vera_curve(vera8, fam)
            vera_verdict_nums[fam] = (a, lo, hi, ad, lo2, hi2, lo_rate, hi_rate)
            vera_rows.append((lab, N(n_nb), N(f'{pct(a)} <span class="muted small">[{pct(lo)}, {pct(hi)}]</span>'), N(f'{pct(ad)} <span class="muted small">[{pct(lo2)}, {pct(hi2)}] (n={n_dec})</span>'),
                              N(pct(vera8[fam]["undecided"])), N(f"{lo_rate:.2f} vs {hi_rate:.2f}"),
                              " / ".join("—" if math.isnan(r[4]) else f"{r[4]:.2f}" for r in rows)))
        k = vera8["kin_roll"]; w = vera8["wall_bounce_wb_pre"]
        vera_rows.append(("P0 등속 (제어)", N(k["n"]), N(pct(k["accuracy"])), N("—"), N(pct(k["undecided"])), N("—"), "14/16 계속 굴러감"))
        vera_rows.append(("A2-0 벽 반동", N(w["n"]), N(pct(w["accuracy"])), N("—"), N(pct(w["undecided"])), N("—"), "반동 3/7"))

    # V-JEPA 2-AC rows
    vj_rows = []
    if vj:
        for fam, lab in (("kin_roll", "P0 등속"), ("hill_roll_wb", "A1 언덕 · 작업대"), ("two_ball_wb", "A2 두 공 · 작업대")):
            f = vj["families"].get(fam); r = (vj_rev or {}).get("families", {}).get(fam)
            if not f:
                continue
            bs = f["by_step"]; g = f["gated"]
            cos = " / ".join(f'{bs[k]["cos_P_mean"]:+.2f}' for k in ("1", "2", "3", "4"))
            cos_r = " / ".join(f'{r["by_step"][k]["cos_P_mean"]:+.2f}' for k in ("1", "2", "3", "4")) if r else "—"
            dc = "P ≡ K" if fam == "kin_roll" else f'{g["dcos_mean"]:+.3f} (오라클 ±{g["dcos_oracle_P_mean"]:.2f})'
            vj_rows.append((lab, N(pct(f["gate_rate"])), N(f'{f["all_steps"]["sep_over_floor_median"]:.1f}'), N(cos), N(cos_r), N(dc)))
        if vjb:
            f = vjb["families"]["support_edge_wb"]; g = f["gated"]
            vj_rows.append(("B2 지지 끝 · 작업대 (Phase B)", N(pct(f["gate_rate"])), N(f'{f["all_steps"]["sep_over_floor_median"]:.1f}'),
                            N(" / ".join(f'{f["by_step"][k]["cos_P_mean"]:+.2f}' for k in ("1", "2", "3", "4"))), N("—"), N(f'{g["dcos_mean"]:+.3f} (오라클 ±{g["dcos_oracle_P_mean"]:.2f})')))

    # DiLA rows
    def pres(px, fam):
        if not px or fam not in px["families"]:
            return "—"
        return " / ".join(f'{px["families"][fam][k]["present_rate"]:.2f}' if k in px["families"][fam] else "·" for k in ("1", "2", "3", "4"))
    dila_rows = [(lab + (" " + chip("wait tiny", "누설") if "oracle" in tag else ""), N(pres(dila_px[tag], "hill_roll_wb")), N(pres(dila_px[tag], "two_ball_wb")), N(pres(dila_px[tag], "kin_roll")))
                 for tag, lab in (("", "stride 4 (4 fps) · action 유지"), ("_zero", "stride 4 · action 0"), ("_s1", "stride 1 (16 fps, 문맥 5프레임) · 유지"), ("_oracle", "누설 참조: 참 미래 latent action")) if dila_px.get(tag)]
    if dila_pb:
        dila_rows.append(("Phase B 지지 끝 · stride 4 · action 0", N("—"), N("—"), N(pres(dila_pb, "support_edge_wb").replace("—", "") or "—")))

    # ------------------------------------------------ overview matrix
    M = {  # (chip class, text)
        "G1": {"probe": ("rep", "R² 0.91–0.99 (표현)"), "ada": ("no", "게이트 실패 · LAM"), "dd": ("no", "게이트 실패 · 도메인"), "c25": ("wait", "단기만 물리적 · 124회"),
               "v2w": ("dash", "—"), "vera": ("dash", "—"), "vjac": ("dash", "—"), "dila": ("dash", "—"), "a2w": ("dash", "—")},
        "G2": {"probe": ("dash", "—"), "ada": ("dash", "—"), "dd": ("dash", "—"), "c25": ("dash", "—"), "v2w": ("no", "Tower 44% · Hill 47% (제현)"), "vera": ("dash", "—"), "vjac": ("dash", "—"), "dila": ("dash", "—"), "a2w": ("dash", "—")},
        "G3": {"probe": ("dash", "—"), "ada": ("dash", "—"), "dd": ("dash", "—"), "c25": ("dash", "—"), "v2w": ("no", "0.46 / 0.65 / 0.39 · 840회"), "vera": ("dash", "—"), "vjac": ("no", "동일 패턴 · 72회"), "dila": ("no", "붕괴 1스텝 늦음"), "a2w": ("dash", "—")},
        "G4": {"probe": ("dash", "—"), "ada": ("dash", "—"), "dd": ("dash", "—"), "c25": ("dash", "—"), "v2w": ("no", "0.49 / 0.43 · 696회"), "vera": ("wait", "0.52 / 0.35 · 120회 · NO-GO 권고"), "vjac": ("no", "NO-GO · 87+134회"), "dila": ("no", "NO-GO · 87×8"), "a2w": ("dash", "—")},
        "G5": {"probe": ("dash", "—"), "ada": ("dash", "—"), "dd": ("dash", "—"), "c25": ("dash", "—"), "v2w": ("wait", "파일럿 66회 · 부양 9/12"), "vera": ("dash", "—"), "vjac": ("no", "낙하/부양 무구분"), "dila": ("no", "freeze"), "a2w": ("wait", "Phase B/C 이후 예정")},
    }
    COLS = [("probe", "V-JEPA 2 물성 probe"), ("ada", "AdaWorld"), ("dd", "DreamDojo 2B"), ("c25", "Cosmos-2.5 base"), ("v2w", "Cosmos-2 V2W"), ("vera", "VERA"), ("vjac", "V-JEPA 2-AC"), ("dila", "DiLA"), ("a2w", "A2World · DW05")]
    ROWS = [("G1", "5-task 푸셔 씬", "slide · roll · bounce · collide · incline"), ("G2", "초기 임계 씬 9종", "seesaw · lean · tower · hill · collide · domino + 에너지 3"),
            ("G3", "Bundle A toy", "언덕 · 두 공 · 진자 (+사전검사 2)"), ("G4", "Phase A 작업대", "P0 등속 · 언덕 · 두 공 (+사전검사 2)"), ("G5", "Phase B 작업대", "매달린 payload · 지지 끝 이탈")]
    matrix = '<div class="tbl-wrap"><table class="matrix"><thead><tr><th>씬 세대</th>' + "".join(f"<th>{c}</th>" for _, c in COLS) + "</tr></thead><tbody>"
    for g, name, sub in ROWS:
        matrix += f'<tr><th scope="row"><span class="gtag">{g}</span> {name}<br><span class="small muted">{sub}</span></th>' + "".join(f'<td><span class="cell {M[g][k][0]}">{M[g][k][1]}</span></td>' for k, _ in COLS) + "</tr>"
    matrix += "</tbody></table></div>"

    TIMELINE = [("08-24", "박제현 · 5-task 씬 + V-JEPA 2 probe 브랜치 (R² 0.91–0.99)"), ("08-26", "R3-79 · dense 데이터 3,050 · AdaWorld·DreamDojo 게이트 실패 · Cosmos-2.5 base 첫 물리층 도달 → 스위프"),
                ("08-31", "박제현 · Cosmos V2W 문턱 씬 6종 판독 (설계 이슈 4건)"), ("09-01", "Bundle A toy 3종 + Cosmos 840 롤아웃 · 프롬프트 대조 288"),
                ("09-02", "PhysicsGen 재정의 P0–P4 · Phase A 작업대 696 · VERA 준비"), ("09-03", "V-JEPA 2-AC NO-GO · DiLA NO-GO · Phase B 씬 2종 + 파일럿 · VERA 8 seed"), ("09-04", "VERA 정량 판독 · Cosmos-2.5 스위프 집계 · 이 총람")]
    timeline = '<ol class="timeline">' + "".join(f'<li><span class="mono date">{d}</span><span class="dot"></span><span class="lab">{t}</span></li>' for d, t in TIMELINE) + "</ol>"

    # ------------------------------------------------ HTML
    css = """
:root{--bg:#EEF0F3;--panel:#FFFFFF;--panel2:#F6F7F9;--ink:#1B2028;--sub:#667080;--line:#D9DEE5;--red:#C94431;--red-bg:#F8E6E2;--blue:#3A6CB4;--blue-bg:#E4ECF7;--green:#2E9A4E;--green-bg:#E2F2E7;--amber:#A8721C;--amber-bg:#F6ECD6;--media:#23262C;--focus:#3A6CB4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14171C;--panel:#1E2329;--panel2:#262C34;--ink:#E8ECF1;--sub:#97A1AD;--line:#2E353E;--red:#E0604C;--red-bg:#3A231F;--blue:#7FA5E0;--blue-bg:#1F2A3B;--green:#5CBF7A;--green-bg:#1D3126;--amber:#D9A441;--amber-bg:#3A3120;--media:#0F1114;--focus:#7FA5E0}}
:root[data-theme="dark"]{--bg:#14171C;--panel:#1E2329;--panel2:#262C34;--ink:#E8ECF1;--sub:#97A1AD;--line:#2E353E;--red:#E0604C;--red-bg:#3A231F;--blue:#7FA5E0;--blue-bg:#1F2A3B;--green:#5CBF7A;--green-bg:#1D3126;--amber:#D9A441;--amber-bg:#3A3120;--media:#0F1114;--focus:#7FA5E0}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:15px;line-height:1.7;word-break:keep-all}
.mono{font-family:"JetBrains Mono",Consolas,monospace} a{color:var(--blue)}
main{max-width:1180px;margin:0 auto;padding:26px 22px 80px}
header.top{display:grid;gap:6px;padding:6px 0 16px}
.eyebrow{font-family:"Chakra Petch",sans-serif;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--sub)}
h1{font-size:30px;line-height:1.25;margin:2px 0 6px;font-weight:700;text-wrap:balance} h2{font-size:21px;margin:0 0 8px;font-weight:700;text-wrap:balance} h3{font-size:16.5px;margin:0 0 4px;font-weight:700} h4{font-size:13.5px;margin:14px 0 6px;font-weight:700;color:var(--sub);letter-spacing:.02em}
p{margin:8px 0} .lead{font-size:16.5px;max-width:70ch} .muted{color:var(--sub)} .small{font-size:13px}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{display:inline-block;background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:2px 9px;font-size:12.5px;font-family:"Chakra Petch","Noto Sans KR",sans-serif;font-weight:500;letter-spacing:.02em;white-space:nowrap}
.chip.cls{background:var(--blue-bg);color:var(--blue);border-color:transparent} .chip.ok{background:var(--green-bg);color:var(--green);border-color:transparent} .chip.no{background:var(--red-bg);color:var(--red);border-color:transparent} .chip.wait{background:var(--amber-bg);color:var(--amber);border-color:transparent} .chip.rep{background:var(--blue-bg);color:var(--blue);border-color:transparent} .chip.tiny{font-size:11px;padding:0 6px;margin-left:6px}
nav.tabs{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);display:flex;gap:4px;align-items:center;padding:8px 0;margin:0 0 18px;overflow-x:auto}
nav.tabs button{font-family:"Chakra Petch","Noto Sans KR",sans-serif;font-weight:600;font-size:14px;letter-spacing:.03em;background:transparent;color:var(--sub);border:1px solid transparent;border-radius:6px;padding:8px 14px;cursor:pointer;white-space:nowrap}
nav.tabs button[aria-selected="true"]{color:var(--ink);background:var(--panel);border-color:var(--line)} nav.tabs button:focus-visible,#motion:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
nav.tabs .spacer{flex:1} #motion{font-family:"Chakra Petch",sans-serif;font-size:12px;background:var(--panel);color:var(--sub);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;white-space:nowrap}
section.block{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px 22px;margin:0 0 16px}
section.gen{border-left:4px solid var(--blue);padding-left:20px}
.gen-head{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;align-items:baseline;margin-bottom:12px}
.gen-head .gtag{font-family:"Chakra Petch",sans-serif;font-weight:600;font-size:22px;color:var(--blue)}
.gen-head .meta{grid-column:2;display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--sub)}
.gtag{font-family:"Chakra Petch",sans-serif;font-weight:600;color:var(--blue)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px} .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px 20px;display:flex;flex-direction:column;gap:8px} .card-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;flex-wrap:wrap}
.formula{font-size:15px;background:var(--panel2);border-left:3px solid var(--red);padding:6px 12px;border-radius:0 4px 4px 0;margin:2px 0}
.media{background:var(--media);border-radius:6px;padding:6px;margin:6px 0 2px} .media img{display:block;width:100%;height:auto;border-radius:3px} .media.scroll{overflow-x:auto} .media.scroll img{width:auto;max-width:none;height:220px}
.cap{font-size:12.5px;color:var(--sub);margin:2px 0 8px} .note{font-size:13.5px;background:var(--amber-bg);color:var(--ink);border-radius:6px;padding:8px 12px;margin:6px 0 0}
dl.spec{display:grid;grid-template-columns:96px 1fr;gap:4px 12px;margin:4px 0 0;font-size:13.5px} dl.spec dt{color:var(--sub)} dl.spec dd{margin:0}
.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px} figure{margin:0} .thumb img{width:100%;height:auto;border-radius:4px;display:block;background:var(--media)} .thumb figcaption{font-size:13px;margin-top:4px;display:flex;flex-wrap:wrap;gap:2px 8px;align-items:center}
.pres{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px} .pre figcaption{font-size:13px;margin-top:4px}
.tbl-wrap{overflow-x:auto;margin:8px 0} table{border-collapse:collapse;width:100%;font-size:13.5px} th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top} thead th{color:var(--sub);font-weight:500;font-size:12.5px} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.matrix th[scope=row]{font-weight:500;min-width:190px} table.matrix td{padding:6px 6px} .cell{display:inline-block;border-radius:4px;padding:2px 8px;font-size:12px;font-family:"Chakra Petch","Noto Sans KR",sans-serif;font-weight:500;white-space:nowrap}
.cell.ok{background:var(--green-bg);color:var(--green)} .cell.no{background:var(--red-bg);color:var(--red)} .cell.wait{background:var(--amber-bg);color:var(--amber)} .cell.rep{background:var(--blue-bg);color:var(--blue)} .cell.dash{color:var(--sub)}
.ex img{width:100%;height:auto;display:block;border-radius:4px;background:var(--media)} .ex figcaption{font-size:13px;margin:4px 0 12px}
.verdict{border-left:3px solid var(--red);background:var(--panel2);padding:8px 14px;border-radius:0 6px 6px 0;margin:10px 0}
.model-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px} .tile{background:var(--panel2);border-radius:6px;padding:14px 16px} .tile h3{font-size:15px;margin-bottom:6px}
ol.timeline{list-style:none;margin:8px 0 0;padding:0;display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));gap:0;overflow-x:auto;position:relative}
ol.timeline li{position:relative;padding:0 12px 0 0;display:grid;grid-template-rows:auto auto auto;gap:6px} ol.timeline .date{font-size:13px;color:var(--sub)}
ol.timeline .dot{display:block;width:10px;height:10px;border-radius:50%;background:var(--red);position:relative;z-index:1} ol.timeline li::before{content:"";position:absolute;left:5px;right:0;top:27px;border-top:2px solid var(--line)} ol.timeline li:last-child::before{right:auto;width:0}
ol.timeline .lab{font-size:13px;line-height:1.5}
ul.plain{margin:6px 0;padding-left:20px} ul.plain li{margin:0 0 6px}
.decide{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px} .decide .tile h3 span{color:var(--amber);font-family:"Chakra Petch",sans-serif;margin-right:8px}
footer{color:var(--sub);font-size:12.5px;margin-top:22px;line-height:1.7}
[hidden]{display:none!important}
@media (max-width:640px){main{padding:18px 14px 60px} .grid2{grid-template-columns:1fr} dl.spec{grid-template-columns:1fr} nav.tabs button{padding:7px 9px;font-size:13px} section.gen{padding-left:14px}}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
"""
    vv = vera_verdict_nums
    vera_verdict = ""
    if vv:
        h, t = vv["hill_roll_wb"], vv["two_ball_wb"]
        vera_verdict = (f'<div class="verdict"><b>정량 판정 (seed 1–8, 격자점당 n = 8): NO-GO 권고 — 최종 확정은 지훈.</b> 언덕 정확도 {pct(h[0])} [{pct(h[1])}, {pct(h[2])}], 두 공 {pct(t[0])} [{pct(t[1])}, {pct(t[2])}] — 둘 다 등속 기준선 50%를 CI 안에 둔다. '
                        f'언덕은 S=2.0에서만 8/8 통과이고 S ≤ 1.26에서는 0.17–0.57로 비단조(문턱 곡선이 아니라 아주 쉬운 경우만 통과). 두 공 반전률은 S&lt;1 {t[6]:.2f} vs S&gt;1 {t[7]:.2f}로 S 무관, 미결 33%. '
                        f'Cosmos가 못 그리는 반동·반환을 그린다는 정성적 특징은 유지되지만, 공 ≈ 5 px·샘플당 220 s·미결 19–33%라 벤치마크 급 정량 평가에는 부적합.</div>')

    html = f"""<title>SimDROID 씬·모델 총람</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&family=Chakra+Petch:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{css}</style>
<main>
<header class="top">
  <div class="eyebrow">SimDROID · World Model 예측 검증 트랙 · 2026-08-24 → 09-04 · 기록 기준 PROGRESS (35)–(43)</div>
  <h1>씬 5세대와 모델 9종, 무엇을 어디까지 검증했는가</h1>
  <p class="lead">MuJoCo 씬에서 world model에게 <b>미래를 그리게</b> 하고 시뮬레이터 정답과 비교하는 트랙의 현재 상태. 씬은 "밀린 큐브의 물성 반응"(G1)에서 "임계 양쪽의 방향 선택"(G3–G5)으로 옮겨 갔고, 모델은 공개 가중치 8종을 실행했다. 어느 모델도 아직 물리 문턱을 반영하지 못했고, 실패 위치는 모델마다 다르다.</p>
  <div class="chips">{chip("", "씬 22종 + 사전검사 4")}{chip("", "모델 8종 실행 · 2종 실사")}{chip("", "롤아웃 ≈ 4,400")}{chip("no", "문턱 반영 모델 0")}{chip("wait", "결정 대기: VERA 확정 · Phase B 본실험")}</div>
</header>
<nav class="tabs" role="tablist" aria-label="섹션">
  <button role="tab" id="tab-over" aria-controls="p-over" aria-selected="true">한눈에</button>
  <button role="tab" id="tab-scenes" aria-controls="p-scenes" aria-selected="false">씬 G1–G5</button>
  <button role="tab" id="tab-models" aria-controls="p-models" aria-selected="false">모델별 검증</button>
  <button role="tab" id="tab-next" aria-controls="p-next" aria-selected="false">판정 · 다음</button>
  <span class="spacer"></span><button id="motion" type="button" aria-pressed="false">GIF 정지</button>
</nav>

<div id="p-over" role="tabpanel" aria-labelledby="tab-over">
<section class="block">
  <h2>연구 질문과 두 층위</h2>
  <div class="tiles">
    <div class="tile"><h3>질문</h3><p class="small">현재 world model은 실제로 미래를 어디까지 예측하는가, 어디서 무너지는가. 계획 성공이나 표현 품질이 아니라 <b>predictor rollout이 실제 dynamics를 얼마나 따라가는지</b>를 시뮬레이터 GT와 직접 비교한다 (R3-68 → R3-79).</p></div>
    <div class="tile"><h3>표현 층위 (probe)</h3><p class="small">frozen feature <b>안에</b> 물성 정보가 있는가. 박제현의 V-JEPA 2 probe가 여기 해당 — R² 0.91–0.99. 선행연구가 이미 다룬 층위라 차별성 약함.</p></div>
    <div class="tile"><h3>예측 층위 (rollout)</h3><p class="small">predictor가 그 정보를 <b>써서</b> 미래를 물리대로 전개하는가. 이 트랙의 대상. 절대 거리는 모델 간 비교가 안 되므로 짝지은 대조(같은 초기 조건, 변수 하나)로 상대 반응을 읽는다.</p></div>
    <div class="tile"><h3>설계 전환 (09-02)</h3><p class="small">G1의 "물성 변수에 대한 반응"은 문맥 연속과 분리가 안 됐다(Cosmos-2.5 스위프). 그래서 <b>현재 속도의 연장으로는 못 푸는 방향 선택</b>(P1–P4 역학 전이)을 무차원 S로 스윕하고, 등속 외삽을 기준선으로 두는 설계로 바꿨다.</p></div>
  </div>
</section>
<section class="block">
  <h2>시간선</h2>
  {timeline}
</section>
<section class="block">
  <h2>씬 세대 × 모델 — 무엇을 어디에 돌렸는가</h2>
  <p class="small muted">셀 = 그 씬 세대에서 그 모델로 얻은 핵심 결과. 정확도는 비경계 S 방향 선택 정확도(등속 기준선 0.50). 빨강 = 문턱 미반영/게이트 실패/NO-GO, 노랑 = 결정 대기 또는 부분, 파랑 = 표현 층위 결과, — = 미실행.</p>
  {matrix}
</section>
<section class="block">
  <h2>지금 결정 대기</h2>
  <div class="decide">
    <div class="tile"><h3><span>1</span>VERA GO/NO-GO 확정</h3><p class="small">seed 8까지 완료(09-03 23:35 KST). 정량 NO-GO 권고 — 위 매트릭스·모델 탭 근거. 확정되면 HANDOFF §5로 이동.</p></div>
    <div class="tile"><h3><span>2</span>Phase B 씬 피드백 → Cosmos 본실험</h3><p class="small">파일럿 66 롤아웃으로 파이프라인 확인 완료. B1 payload 판독 약함(strict valid 0.30) 처리 방식 결정 후 24 seed(≈ 2 h GPU) 발사.</p></div>
    <div class="tile"><h3><span>3</span>A2World 착수 시점</h3><p class="small">pretrained 변형은 뷰당 조건 프레임 1장이라 수동 문턱 씬에서 속도를 못 읽음. 결정대로 Phase B/C(robot-context 씬) 이후. DW05는 history 1프레임 고정이라 제외 권고.</p></div>
    <div class="tile"><h3><span>4</span>R3-79 코멘트 · Phase C</h3><p class="small">Linear R3-79는 코멘트 0건(지훈 직접 작성 예정). Phase C = 자연 occlusion 짝 3종 + 판독기 가림 창 예외.</p></div>
  </div>
</section>
</div>

<div id="p-scenes" role="tabpanel" aria-labelledby="tab-scenes" hidden>
<section class="block">
  <h2>공통 계약 (G3–G5)</h2>
  <div class="tiles">
    <div class="tile"><h3>조건과 지평</h3><p class="small">832×480 · 16 fps 정확 · 조건 5프레임(0.31 s, 사건 전) → 21프레임(1.31 s) 판독. V-JEPA/DiLA latent 트랙은 256² · 4 fps(문맥 {{o, o+4}}, 예측 4스텝).</p></div>
    <div class="tile"><h3>포화도 S</h3><p class="small">씬마다 결정 변수 하나를 S = 구동/임계로 정규화(S = 1 경계). 격자 11점 {{0.50 … 2.00}} × seed 24. |S−1| ≤ 0.04는 경계로 두고 정확도에서 제외.</p></div>
    <div class="tile"><h3>모델 실행 전 게이트</h3><p class="small">경계 이분법(폐형식 대비 &lt; 격자 한 칸) · 에너지 감사 · timestep 일치 · nuisance(μ·밀도·감쇠) 불변 · 사건이 조건 창 밖 · 지평 내 결정. 통과 전 Cosmos 금지.</p></div>
    <div class="tile"><h3>역학 전이 계층</h3><p class="small">P0 등속 연장(제어) → P1 연속 역학(언덕·진자) → P2 충격(두 공·벽) → P3 접촉 전이(지지 끝) → P4 안정성(시소·탑, Phase 2). 가림 O0–O4는 별도 축(Phase C).</p></div>
  </div>
  <p class="small muted">설계 목표 전문(P0–P4, 세 원칙, 지표 구조 18.1–18.7)은 <a href="https://claude.ai/code/artifact/d972733e-09fb-4e14-a38e-a4777d16e079">씬·설계·모델 UI (09-03)</a> 02 탭에 있다.</p>
</section>

<section class="block gen">
  <div class="gen-head"><span class="gtag">G1</span><h2>5-task 푸셔 씬 — 물성 반응 (rollout-vs-GT)</h2>
    <div class="meta"><span>2026-08-24 박제현 (feat/probe-physics-jepa) · dense 재생성 08-26 지훈</span><span>목적: 같은 초기 프레임·같은 밀기에서 물성 하나가 바뀌면 rollout도 시뮬처럼 바뀌는가</span><span>{chip("no", "트랙 종료 상태 — 문맥 연속과 물성 반응이 분리 안 됨")}</span></div></div>
  <p>rod 푸셔가 1.20 m/s로 큐브/구를 치고(slide·roll·collide), 큐브를 떨어뜨리고(bounce), 경사면에 놓는다(incline). 파라미터는 log-uniform, 매치 쌍은 한 파라미터만 15%/85% 분위로 고정. 팀 데이터(probe용 8프레임)와 같은 seed·index라 물리 파라미터가 같다.</p>
  <div class="grid2">{g1_cards}</div>
  <h4>GT 판별력 상한 (stride 2 = 25 fps, 20프레임, 30쌍 평균 — 어떤 모델도 이 이상 구분할 수 없다)</h4>
  {table(["셀", "#GT MSE @end", "#중심 격차 @mid/@end (px)", "#Δstate end (cm)", "#AdaWorld-LAM cos(lo,hi)"], gt_gap_rows)}
  <p class="small muted">bounce/mass·incline/mass는 자유낙하·마찰 경사면 모두 질량 독립(a = g(sinθ − μcosθ))이라 lo/hi GT가 픽셀 단위로 동일 — 측정 불가. 근거 HANDOFF §3.6, <span class="mono">code/out/rollout_adaworld/pair_gt_gap_s2.json</span>.</p>
</section>

<section class="block gen">
  <div class="gen-head"><span class="gtag">G2</span><h2>초기 임계 씬 9종 — 픽셀만 보고 임계의 어느 쪽인지</h2>
    <div class="meta"><span>2026-08-31 · 09-01 박제현 (feat/probe-physics-vwm) · 판독 지훈 (33)·(34)</span><span>정적 3 + 동적 3 (7점 스윕) + 에너지 보존 3 (8 마진)</span><span>{chip("wait", "P0–P4로 재배치 · 언덕·두 공·진자는 A1·A2·A3의 전신")}</span></div></div>
  <div class="thumbs">{g2_thumbs}</div>
  <h4>지훈 판독에서 나온 설계 이슈 (모델 실행 없이 MuJoCo trace 재실행, 42점 중 41점 margin↔outcome 일치)</h4>
  <ul class="plain small">
    <li><b>collide</b>: 스윕 축(감쇠→반발계수)이 조건 5프레임에서 원리적으로 비가시 — 정답률이 임계 읽기가 아니라 모델의 e prior를 잰다. → rB(크기=질량비) 스윕 제안 → A2 두 공으로 계승.</li>
    <li><b>hill</b>: 공유 비교 시점 +0.32 s에 7점 전부 능선 왼쪽·전진 중 → endpoint 디코딩이 속도만 읽음. 가족별 event_frame+Δ로 지평 이동 제안 → Bundle A의 21프레임 궤적 판정으로 계승.</li>
    <li><b>tower</b>: o1 고정이라 항상 top interface가 binding → "맨 위 큐브 1개" 문제로 축소; 인접점 가시성 5 mm ≈ 3–4 px. <b>domino</b>: 좁은 2점은 첫 전도가 조건 창 안(누설). <b>카메라</b>: hill/collide는 테이블용 side 카메라 재사용(공 절반 잘림).</li>
    <li><b>에너지 묶음</b>: ramp ±0.05는 평가 시점(frame 9)에 결말 미해결, energy_hill |m| ≥ 0.2는 말기에 구슬이 호 밖, pendulum은 무감쇠 회전이라 각도 앨리어싱. 정확-16 fps 반올림 버그(sub=62 → 16.13 fps)는 Bundle A에서 수정, 에너지 씬에도 잠복.</li>
  </ul>
</section>

<section class="block gen">
  <div class="gen-head"><span class="gtag">G3</span><h2>Bundle A toy — 같은 원리를 공유하는 씬 묶음</h2>
    <div class="meta"><span>2026-09-01 지훈 (code_vwm, 스펙 md 2건 + 기계공학 수정안)</span><span>A1 언덕 ↔ A3 진자 = 에너지 짝, A2 두 공 = 운동량, 사전검사 A1-0·A2-0</span><span>{chip("ok", "MuJoCo 게이트 통과 · Cosmos 840 + 대조 288 완료")}</span></div></div>
  <div class="grid2">{g3}</div>
  <h4>사전검사 씬</h4>
  <div class="pres">{g3_pre}</div>
</section>

<section class="block gen">
  <div class="gen-head"><span class="gtag">G4</span><h2>Phase A 작업대 — 외관은 사실적으로, 물리는 그대로</h2>
    <div class="meta"><span>2026-09-02 지훈 (PhysicsGen 재정의 Stage 1)</span><span>P0 등속 제어 신설 + A1·A2 작업대 재스킨 (+ 사전검사 2)</span><span>{chip("ok", "Cosmos 696 · VERA 120 · V-JEPA 2-AC · DiLA 완료")}</span></div></div>
  <p>회색 라미네이트 검사대·뒷벽·정지 Franka Panda·트레이·픽스처·빈·테이프 눈금을 <b>충돌 없는 시각 요소</b>로만 얹었다. 옵션·접촉쌍·충돌 geom·hfield·카메라·S 격자는 toy와 동일(컴파일 수준 서명 identical)이라 toy↔작업대는 순수 외관 대조다. 색은 판독기 마스크 배타성 유지(빨강·파랑·초록만).</p>
  <div class="grid2">{g4}</div>
  <p class="small muted">렌더 함정: GPU가 diffusion에 점유되면 EGL이 검은/손상 프레임을 냈다 → OSMesa로 전환, OSMesa는 상하 반전이라 <span class="mono">Base.render()</span>가 뒤집는다(기준 프레임 비교로 검증). pod 재기동마다 <span class="mono">apt-get install -y libosmesa6</span>.</p>
</section>

<section class="block gen">
  <div class="gen-head"><span class="gtag">G5</span><h2>Phase B 작업대 — 구속 보존 짝 + 접촉 상실</h2>
    <div class="meta"><span>2026-09-03 지훈 (Stage 3)</span><span>P1-B 매달린 payload (A3 재스킨) · P3-A 지지 끝 이탈 (신규)</span><span>{chip("wait", "씬 검증 완료 · Cosmos 파일럿 66 · 본실험은 피드백 후")}</span></div></div>
  <div class="grid2">{g5}</div>
  <p class="small muted">22샘플(480×832·24프레임, OSMesa 이중 렌더) 생성 게이트 전부 통과, GT 자가검증 21/22. 검토 페이지: <a href="https://claude.ai/code/artifact/549af421-d4b2-4dfb-b5fd-60650bd52c2d">Phase B 씬 (09-03)</a>.</p>
</section>
</div>

<div id="p-models" role="tabpanel" aria-labelledby="tab-models" hidden>
<section class="block">
  <h2>한눈에 — 실행한 모델 8종과 실사만 한 2종</h2>
  {table(["모델", "층위 · 종류", "입력 → 출력", "돌린 씬", "#규모", "핵심 결과", "판정"], [
      ("V-JEPA 2 물성 probe (박제현)", "표현 · frozen ViT-g + PCA96/Ridge", "8프레임 클립 → mass/μ/… 회귀", "G1 5-task", N("250+250 / 80×3"), "R² 0.91–0.99, 물리적으로 불가능한 곳에서 정확히 실패; 실제 영상 물리는 못 읽음", chip("rep", "표현 층위 — 예측 검증 아님")),
      ("AdaWorld (ICML'25)", "예측 · latent-action 영상 생성 (SVD UNet + LAM 32-d)", "256² 문맥 1–6 + 이식 액션 → 20프레임", "G1 slide/mu", N("스모크 stride 2·5"), "예측이 푸셔를 지우고 큐브 정지. LAM이 전역 이동에만 민감", chip("no", "게이트 실패 · 액션 추출")),
      ("DreamDojo-Pretrain 2B (NVIDIA)", "예측 · 액션 조건 영상 생성 (Cosmos-2.5 + LAM 700M)", "640×480 첫 프레임 + 13프레임 청크 액션 → 13프레임", "G1 slide/mu (+그리퍼 외형)", N("스모크 + 진단 3"), "우리 씬에서 무응답(in-domain은 정상). 그리퍼는 사람 손으로 환각, 큐브 정지", chip("no", "게이트 실패 · 도메인")),
      ("Cosmos-Predict2.5-2B base", "예측 · 영상 이어 그리기 (액션 없음)", "480×640 문맥 5프레임 → 93프레임", "G1 판별 가능 8셀", N("124 + 확인 17"), "첫 5–8프레임은 freeze 대비 6/8·등속 대비 4/8 셀 개선; 장기 드리프트·seed 의존", chip("wait", "단기만 물리적")),
      ("Cosmos-Predict2-2B Video2World", "예측 · 텍스트+영상 조건 diffusion", "5프레임 0.31 s → 21프레임 1.31 s, 480×832", "G2 (제현) · G3 · G4 · G5", N("535 + 840 + 288 + 696 + 66"), "언덕 전 구간 통과, 두 공 반전 못 그림, 진자만 S자(S50 0.82), 지지 끝은 공중 부양. 텍스트 무효", chip("no", "문턱 미반영 · 생성 한계")),
      ("VERA DROID planner (Wan2.1-I2V-14B)", "예측 · 로봇 영상 생성", "3뷰 576×128 · 29프레임 → 24프레임 × 3청크", "G4 부분집합 15행", N("8 seed = 120"), "반동·반환은 그리지만 S와 무관(무작위), 미결 19–33%, 220 s/샘플", chip("wait", "정량 NO-GO 권고 · 확정 대기")),
      ("V-JEPA 2-AC (ViT-g + AC predictor)", "예측 · latent 자기회귀", "256² 2프레임 latent → 4스텝 latent (액션 0)", "G4 29행 · G3 24행 · G5 11행", N("87 + 134 제어 + 72 + 33"), "인코더 게이트 가까스로 통과, 예측기는 수동 물체 운동을 외삽하지 않음(반전 제어로 확정)", chip("no", "NO-GO · 운동 전개 없음")),
      ("DiLA (ICML'26)", "예측 · 구조/내용 분리 latent + latent action", "2–5프레임 latent + action 정책 → 4–16스텝 → RAE 디코드", "G4 · G3 · G5", N("87 × 10변형 + 72 × 7 + 33 × 3"), "past-only 예측 없음. 어떤 정책·시간축에서도 1–4스텝 안에 장면이 흐려지고 공이 사라짐", chip("no", "NO-GO · 자기회귀 붕괴")),
      ("A2World · DW05-Base", "실사만 (README·코드·HF)", "A2World: 뷰당 조건 1프레임 + 20 action / DW05: 이미지 1장 + 텍스트", "—", N("0"), "둘 다 history에서 속도 식별 불가. A2World는 robot-context 씬에서만 의미", chip("wait", "A2World 연기 · DW05 제외 권고")),
  ])}
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 1 · 표현 층위 · 박제현</span><h2>V-JEPA 2 물성 probe</h2></div>{chip("rep", "표현 층위 — 예측 검증 아님")}</div>
  {dl([("무엇", "frozen V-JEPA 2 (ViT-g, <span class='mono'>vjepa2-ac-vitg.pt</span>) feature에서 물성(mass / μ / 굴림저항 / 접촉 감쇠)을 회귀. linear probe는 250 trajectory로 부족 → PCA 96차원 + Ridge"),
        ("데이터", "G1 5-task, 학습 250 clean + 250 카메라 1 cm 이동, 테스트 80 × 3조건(clean / 카메라 1 cm / 조명 +20%), 8프레임 희소 클립 256²"),
        ("결과 (README)", "sim에서 관측 가능한 물성 R² 0.91–0.99, 물리적으로 불가능한 곳(bounce/mass 등)에서 정확히 실패. raw pixel보다 카메라·조명 변화에 robust. sim-to-real: motion-only feature가 DROID 클립의 분포 간극을 좁히지만 실제 영상 물리는 아직 못 읽음"),
        ("이 트랙에서의 위치", "&quot;feature 안에 물성 정보가 있는가&quot;(표현)에 답한다. 이 트랙의 질문 &quot;predictor가 그 정보를 써서 미래를 전개하는가&quot;(예측)와 층위가 다르다 — 지훈이 R3-69에 남긴 &quot;next-state prediction용 JEPA/WM이 필요&quot;가 그 간극"),
        ("함정 3", "<span class='mono'>body_mass</span> 변경 후 <span class='mono'>mj_setConst()</span> 필수 · EGL 렌더는 조용히 깨짐(2회 렌더 bitwise 일치 요구) · MuJoCo EGL과 PyTorch CUDA를 같은 프로세스에 두지 말 것")])}
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 2 · 예측 · latent-action 영상 생성 · 2026-08-26</span><h2>AdaWorld (ICML 2025)</h2></div>{chip("no", "게이트 실패 · 액션 추출(LAM)")}</div>
  {dl([("계약", "SVD 초기화 VideoUNet + KL-VAE(256) + LAM(32-d, 2프레임 쌍). 256², 문맥 1–6프레임, 1프레임씩 자기회귀, 샘플링 5 step, CFG 1.1. 우리 클립 640×480 → 480² 크롭 → 256², stride 2(25 fps) / 5(10 fps)"),
        ("leakage-safe 프로토콜", "공식 action-transfer 그대로: matched pair에서 latent action을 <b>normal(lo) 영상의 (t, t+1)</b>에서 뽑아 abnormal(hi) 초기 프레임에 이식(또는 역). 두 영상의 푸셔 액션이 같으므로 &quot;같은 액션·다른 물성&quot;과 정확히 맞고 예측 대상의 미래는 모델에 보이지 않음. 진단용 <span class='mono'>oracle_hi</span>(누설)만 예외"),
        ("실행", "slide/mu pair 00, arm lo2hi / hi2lo / oracle_hi, stride 2·5. 환경 함정: xformers 0.0.20에 sm_90 커널 없음 → attention을 SDPA로, VAE를 vanilla로 오버라이드"),
        ("결과", "예측이 <b>푸셔를 지우고 큐브가 정지</b>(stride 2·5, oracle 포함). LAM 진단: 합성 5 px 전역 이동은 2.5로 잡지만 실제 슬라이드(0.8)는 거의 안 담김 — latent action이 물체 이동을 표현하지 못함. LAM cos(lo,hi) 0.85–1.00(G1 표)")])}
  {img(ada, "AdaWorld lo→hi 이식 스모크 (stride 2)")}
  <p class="cap">slide/mu pair 00, lo 영상의 액션을 hi 초기 프레임에 이식(stride 2). 행: 문맥·GT·예측. 예측 프레임에서 푸셔가 사라지고 큐브는 제자리.</p>
  <div class="verdict"><b>실패 위치: 액션 추출.</b> 생성기가 아니라 LAM이 우리 장면의 운동(물체 이동)을 latent action에 담지 않는다. 벤치마크 규칙: &quot;이식된 액션이 우리 장면에서 전개되는가&quot;(action expressivity gate)를 통과해야 rollout-vs-GT 수치를 보고한다 — 게이트 실패 자체가 결과(Track A md §9.2).</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 3 · 예측 · 액션 조건 영상 생성 · 2026-08-26</span><h2>DreamDojo-Pretrain 2B (NVIDIA)</h2></div>{chip("no", "게이트 실패 · 도메인 간극(생성기)")}</div>
  {dl([("계약", "Cosmos-Predict2.5 2B + WAN2.2 tokenizer(시간 압축 4) + LAM 700M(32-d). 640×480, 첫 프레임 조건, 13프레임 청크, 청크당 latent action 12개(action_dim 384 = 12 × 32), 35 step. 학습은 20 fps 연속 13프레임 → 50 Hz에는 stride 3"),
        ("코드에서 확정한 함정", "공개 추론 경로 <span class='mono'>generate_vid2world</span>는 <span class='mono'>lam_video</span>를 넣기만 하고 LAM 적용 블록이 주석 처리 → <b>latent action이 전혀 반영되지 않음</b>(세 arm 예측이 바이트 동일). 어댑터가 <span class='mono'>model.lam</span>을 직접 호출해 action[:, −32:]에 넣도록 수정"),
        ("실행", "slide/mu stride 3·1, guidance 0/3/7, 실제 소스·합성 20 px 팬·null 소스, in-domain GR-1 실클립 배관 검증, 그리퍼 외형 변형(<span class='mono'>pairs_hand</span>: 같은 충돌 캡슐 위에 Panda 그리퍼 메쉬만 시각으로)"),
        ("결과", "우리 씬: 푸셔 미진입·큐브 정지·배경 그리퍼 환각만, guidance·합성 팬에도 무응답. LAM 자체는 표현력 충분(실제 운동 |z − z_null| 5–8, lo/hi cos 0.2–0.6). <b>in-domain GR-1 클립에서는 oracle 액션을 정확히 따름</b>(PSNR 23.0 vs null 19.1) → 배관 정상, 장면–학습분포 간극이 원인. 그리퍼 외형: 행위자가 움직이기 시작하지만 <b>사람 손/팔로 바꿔 그리고</b> 큐브는 정지(stride 3·1 동일)")])}
  <div class="grid2">
    <figure class="ex">{f'<img src="{dd_rod}" alt="DreamDojo 막대 푸셔 스모크" loading="lazy">' if dd_rod else ""}<figcaption>막대 푸셔 (stride 3, lo→hi): 푸셔가 들어오지 않고 큐브 정지.</figcaption></figure>
    <figure class="ex">{f'<img src="{dd_hand}" alt="DreamDojo 그리퍼 외형 스모크" loading="lazy">' if dd_hand else ""}<figcaption>그리퍼 외형 (stride 3, hi→lo): 행위자는 움직이나 사람 손으로 환각, 큐브 정지.</figcaption></figure>
  </div>
  {img(hand_cmp, "막대 푸셔 vs 그리퍼 외형")}
  <p class="cap">진로 (a) 1단계: 같은 물리 위에 외형만 rod → Panda 그리퍼. state 궤적은 float32 정밀도로 동일. 판정 &quot;행위자 외형은 gate의 필요조건일 뿐 충분조건 아님&quot;.</p>
  <div class="verdict"><b>실패 위치: 생성기(액션 → 운동 매핑).</b> AdaWorld와 반대쪽에서 막혔다. 두 공개 모델 모두 무적응으로는 gate를 통과하지 못해 <b>5-task 장면에서 rollout-vs-GT 물리 점수는 산출 불가</b>(08-26 결론). 다음 선택지 (a) 장면을 모델 분포로 / (b) 모델을 장면으로(post-training) / (c) 예측기 부류 변경 — 실제로는 (a)의 1단계만 실행 후 설계 자체를 바꿨다.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 4 · 예측 · 영상 이어 그리기(액션 없음) · 2026-08-26</span><h2>Cosmos-Predict2.5-2B base</h2></div>{chip("wait", "단기만 물리적 · 5-task 트랙 종료")}</div>
  {dl([("계약", "같은 DreamDojo 체크아웃의 base video2world config + HF 캐시 ckpt. 480×640, <span class='mono'>state_t=24</span>(93프레임)로만 정합적 생성 — 17프레임 축약은 노이즈로 붕괴. 문맥 5프레임(<span class='mono'>num_latent_conditional_frames=2</span>), neutral 프롬프트, guidance 3, 생성 48 s/회, VRAM 28 GB"),
        ("스위프 전 확인 4종", "문맥에 <b>접촉과 감속이 보여야</b> 이어 그린다(접촉 전 문맥 44 px &gt; freeze 32) · 720p 무익(3.5배 느림) · guidance 0 열세 · seed 분산 커서 셀당 seed ≥ 3. 최적: ctx_start 3, 첫 5프레임 오차 10.3 ± 4.1 px &lt; freeze 16.8 · constvel 17.5"),
        ("스위프", "판별 가능 8셀 × 4쌍(incline 3) × 2 side × 2 seed = 124회, 08-26 18:01 KST 발사, 2 h 예산. 집계는 이 세션(09-04)에서 처음 수행(<span class='mono'>code/out/sweep_cosmos/SUMMARY.md</span>)")])}
  {img(c25, "Cosmos-2.5 base 이어 그리기 (slide/mu hi)")}
  <p class="cap">slide/mu pair 00 hi(높은 μ) 문맥 이어 그리기, 93프레임. 큐브 확대 시트: 문맥 → 예측 첫 프레임들 → 마지막.</p>
  <h4>셀 스위프 집계 — lo side(큐브가 움직이는 문맥), 예측 첫 5프레임 큐브 궤적 오차 (px)</h4>
  {table(["셀", "#err5", "#freeze5", "#constvel5", "#err8", "#own&lt;src (8f)", "#GT gap8"], [
      ("slide / mu", N("20.8"), N("25.0"), N("19.4"), N("24.0"), N("0.75"), N("59.1")), ("slide / mass", N("15.0"), N("18.3"), N("17.4"), N("21.1"), N("0.88"), N("62.2")),
      ("collide / mu", N("6.8"), N("2.6"), N("8.7"), N("10.2"), N("0.75"), N("16.7")), ("collide / mass2", N("5.8"), N("2.7"), N("7.0"), N("7.0"), N("0.88"), N("18.3")),
      ("roll / roll_fric", N("6.2"), N("14.3"), N("4.0"), N("8.2"), N("1.00"), N("37.2")), ("roll / mass", N("5.4"), N("10.0"), N("3.1"), N("6.5"), N("1.00"), N("30.8")),
      ("bounce / damping", N("32.1"), N("73.0"), N("103.2"), N("35.0"), N("1.00"), N("77.4")), ("incline / mu", N("32.0"), N("78.8"), N("19.8"), N("48.4"), N("0.00"), N("89.7"))])}
  <p class="small muted">own&lt;src = (쌍, seed) 중 예측이 짝의 GT보다 자기 GT에 가까운 비율(첫 8프레임). hi side는 큐브가 이미 정지(freeze ≈ 0)라 표에서 제외. 프롬프트는 전 가족에 slide용 문장 공통(장면 서술 불일치는 해석 시 고려).</p>
  <div class="verdict"><b>판정: 단기 연속만 물리적.</b> 첫 5프레임 오차가 freeze보다 작은 셀 6/8, 등속 외삽보다 작은 셀 4/8(slide/mass·collide 2셀·bounce). own&lt;src 0.75–1.00은 <b>물성 추론이 아니라 문맥 연속의 증거</b> — lo/hi는 문맥 프레임부터 속도·위치가 다르다. incline/mu lo 0.00은 경사면 가속을 못 따라감. &quot;물성 변수에 대한 반응&quot;은 이 설계로는 문맥 차이와 분리되지 않는다 → 문턱 방향 선택(G3) 설계로 전환한 근거.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 5 · 예측 · 텍스트+영상 조건 diffusion · 08-31 → 09-03</span><h2>Cosmos-Predict2-2B Video2World</h2></div>{chip("no", "문턱 미반영 · 관성 외삽 (생성 한계)")}</div>
  {dl([("계약", "Diffusers <span class='mono'>Cosmos2VideoToWorldPipeline</span>, 480p/16 fps 원본 ckpt. 조건 5프레임(= Wan VAE 시간 압축 후 latent 2장) + 중립 프롬프트 → 21프레임, 35 step, guidance 0(G3–G5) / 7(G2), 14 s/롤아웃"),
        ("판독", "GT에만 캘리브레이션 후 동결하는 색 마스크 프록시: 빨강/파랑 중심 궤적 → 방향, (y, v²) 기울기, Σmv 비, validity(면적 0.35–2.8× 게이트). 첫 파일럿에서 manifest kind 충돌로 전 롤아웃이 이미지 조건이 됐던 사고를 REPLAY 검사로 잡아 재발사"),
        ("G2 (박제현, README)", "535 롤아웃(213 base · 213 oracle · 109 image-only). majority 57.1% / base Tower 44.2% / base Hill 46.8% / oracle 프롬프트 45.5 → 54.0% (McNemar p = 0.004) / Hill image-only는 결정 90.9% 뒤집힘(p = 0.34). Tower는 거의 항상 '안정', Hill은 '통과'"),
        ("G3–G5 (지훈)", "Bundle A 24 seed × 35 = 840 + 프롬프트 대조 288 · Phase A 24 × 29 = 696 · Phase B 파일럿 3 × 22 = 66")])}
  <h4>실제 예시 (seed 1)</h4>
  {cosmos_ex}
  <h4>결과 — 비경계 S 방향 선택 정확도 (등속 기준선 0.50)</h4>
  {table(["씬", "#정확도 [Wilson 95%]", "#미결", "#strict valid", "곡선·법칙"], cosmos_rows)}
  <div class="grid2">
    <figure class="ex">{f'<img src="{curves_toy}" alt="Bundle A 곡선" loading="lazy">' if curves_toy else ""}<figcaption>Bundle A toy: P(outcome | S) 24 seed. 언덕 평탄(전부 통과), 진자 S자(S50 0.82), 두 공 평탄(반전 없음).</figcaption></figure>
    <figure class="ex">{f'<img src="{curves_wb}" alt="Phase A 곡선" loading="lazy">' if curves_wb else ""}<figcaption>Phase A 작업대: toy와 CI 안에서 동일 — 외관은 방향 선택을 바꾸지 않는다.</figcaption></figure>
  </div>
  {f'<figure class="ex"><img src="{curves_pb}" alt="Phase B 파일럿 곡선" loading="lazy"><figcaption>Phase B 파일럿(3 seed): 지지 끝은 S ≥ 1.26에서만 1/3 낙하, payload는 등급형이나 판독 약함.</figcaption></figure>' if curves_pb else ""}
  <p class="small muted">P0 등속 제어(24 seed): {p0_line}. 사건 없는 장면에서도 저속은 +25% 과속, 고속은 −23% 감속 = P1/P2 판독의 잡음 바닥. 프롬프트 대조군(정답/오답 서술 288): 세 가족 모두 |Δ| ≤ 0.02, &quot;되튀어 온다&quot;고 써도 반전 13% — 텍스트는 방향 선택을 움직이지 못한다.</p>
  <div class="verdict"><b>판정: 문턱 미반영, 실패는 생성기의 물리에 있다.</b> 언덕은 S 전 구간 96–100% 넘어감(관성 외삽, 사전검사 S=0.35도 20–23/24), 두 공은 S 무관 반전 10–33%(운동량 전달 붕괴, 벽 반동 5–11/24), 진자만 등급형 전이가 있으나 문턱이 S 0.82로 이동, 지지 끝은 플레이트를 지나 <b>공중 직진</b>(지지 일관성 위반 = P3-A가 노린 실패 양식). 외관(toy↔작업대)도 텍스트도 곡선을 바꾸지 못했다.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 6 · 예측 · 로봇 영상 생성 · 09-02 → 09-03</span><h2>VERA DROID planner (Wan2.1-I2V-14B)</h2></div>{chip("wait", "정량 NO-GO 권고 · 최종 확정 대기")}</div>
  {dl([("계약", "DROID 3카메라 192×128 수평 타일(576×128) · 15 fps · 문맥 29프레임 → 24프레임/호출, 텍스트 프롬프트, 40 step. 로드 81 s, VRAM 48 GB, 청크 74 s(단독) / 190–300 s(Cosmos 동시)"),
        ("우리 씬 접합", "문맥 29프레임이 필요한데 사건 전 history가 0.31 s뿐 → <b>100 Hz로 캡처해 시간을 늘림</b>(6.7배 슬로모션), 3청크 자기회귀 = 0.72 s. 3뷰 = 측면 + 3/4 상방 + 근상방. 판독은 측면 타일을 16 fps 등가로 재표본해 동결 판독기 적용(공 ≈ 5 px)"),
        ("실행", "Phase A 부분집합 S ∈ {0.5, 0.79, 0.95, 1.05, 1.26, 2.0} × 2가족 + kin 2 + 벽 1 = 15행 × seed 1–8 = 120 롤아웃(seed 4–8은 09-03 15:47 → 23:34 KST, K700 학습과 CPU 쿼터 경합으로 지연)")])}
  <h4>실제 예시</h4>
  {vera_ex}
  <h4>결과 — seed 1–8 (격자점당 n = 8)</h4>
  {table(["가족", "#n", "#정확도 (미결 = 오답)", "#결정된 것만", "#미결", "#p(pred=1) S&lt;1 vs S&gt;1", "S별 p(pred=1 | 결정) 0.5 / 0.79 / 0.95 / 1.05 / 1.26 / 2.0"], vera_rows)}
  {vera_verdict}
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 7 · 예측 · latent 자기회귀 · 09-03</span><h2>V-JEPA 2-AC (ViT-g + action-conditioned predictor)</h2></div>{chip("no", "NO-GO · 운동 전개 없음")}</div>
  {dl([("계약", "hub <span class='mono'>vjepa2_ac_vit_giant</span>: encoder(256², 패치 16, 튜블릿 2, 프레임당 256토큰 × 1408) + predictor(<span class='mono'>forward(x, actions[B,T,7], states[B,T,7])</span>, 인과 마스크). DROID 4 fps 학습 → 우리 16 fps에서 stride 4. 액션 0·EE state 상수(정지 Franka와 정합) → <b>누설이 구조적으로 없음</b>"),
        ("latent 판독", "같은 시각의 물리 미래 P와 등속 counterfactual K(MuJoCo qpos 재배치 렌더, 재배치 오차 0.002–0.044)와 1 px 지터 J를 같은 인코더로 → 게이트 sep = d(z_P, z_K) &gt; 3 × floor d(z_P, z_J); 변위 방향 dcos = cos(ẑ − z_last, z_P − z_last) − cos(ẑ − z_last, z_K − z_last) (freeze 교란 제거; 오라클 ±0.4–0.5)"),
        ("실행", "Phase A 29행 × 3오프셋 = 87 + 제어(문맥 3프레임 47, 문맥 반전 87) + Bundle A 72 + Phase B 지지 끝 33. 구현·실행 ≈ 55분, GPU ≈ 8분, VRAM 6 GB")])}
  <div class="grid2">
    <figure class="ex">{f'<img src="{vj_prev}" alt="후보 미래 렌더" loading="lazy">' if vj_prev else ""}<figcaption>A1 작업대 S=0.95, 256² 전용 카메라. 위: 문맥 2프레임 + 물리 미래 P · 가운데: 등속 K · 아래: 1 px 지터 J. 예측기는 위 두 프레임의 latent만 받는다.</figcaption></figure>
    <figure class="ex">{f'<img src="{vj_curves}" alt="latent 트랙 곡선" loading="lazy">' if vj_curves else ""}<figcaption>위: Δ/sep — 예측기(빨강)는 freeze(회색)의 축소판. 가운데: sep/floor와 게이트선. 아래: dcos(● 전체, ▲ P≠K 패치)와 오라클 띠 — 어느 S에서도 0 근처.</figcaption></figure>
  </div>
  <h4>결과</h4>
  {table(["씬", "#게이트 통과", "#sep/floor", "#cos_P 스텝 1/2/3/4", "#문맥 반전 시 cos_P", "#dcos (게이트)"], vj_rows)}
  <div class="verdict"><b>판정 NO-GO (모델 원인).</b> 인코더는 두 미래를 1 px 잡음의 2–4.5배로 겨우 구분한다(공 11–17 px). 예측 변위의 정렬(cos ≈ 0.2)은 문맥 순서를 뒤집어도, 3프레임으로 늘려도, 속도를 바꿔도 그대로 — 운동이 아니라 정적 성분에서 오는 정렬. 물리·등속 선호 차 |dcos| ≤ 0.05(오라클 ±0.45), S 무관, EE state 무시(pose 상관 1.00). Phase B 지지 끝에서는 낙하 후 하늘 vs 검사대가 인코더에 뚜렷(sep/floor 3–4.8)한데도 dcos ≈ 0 → <b>낙하와 부양을 구분하지 않는다</b>. DROID 로봇 운동으로 post-train된 AC predictor는 액션 0인 장면의 수동 dynamics를 전개하지 않는다.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 8 · 예측 · 구조/내용 분리 latent 동역학 · 09-03</span><h2>DiLA (Disentangled Latent Action world model, ICML 2026)</h2></div>{chip("no", "NO-GO · past-only 부재 + 자기회귀 붕괴")}</div>
  {dl([("계약", "DINOv2-RAE latent(224² 재표본), 구조 g·내용 c 분리, ST-Transformer 역동역학 → 256-d latent action, Mamba 내용 메모리. <span class='mono'>get_latent_actions</span>는 <b>전체 시퀀스(미래 포함)</b>의 역동역학이라 past-only 예측이 없음 → 미래 action을 우리가 정해야 함: 유지(주) / 0 / 문맥 평균; 누설 참조 = 참 미래 action(진단 전용)"),
        ("실행", "V-JEPA 입력 번들(256², GT/K/J) 재사용, stride 4·2·1(4·8·16 fps), 문맥 2–5프레임, Phase A 87 × 10변형 + Bundle A 72 × 7 + Phase B 33 × 3. 판독 = latent 지표(V-JEPA와 동일) + <b>RAE 디코드 픽셀 판독</b>(빨간 공 존재율·위치). 디코더 상한 정상: GT latent 복원 존재율 0.93–1.0, 중심 오차 ≈ 1 px. VRAM 3 GB")])}
  {img(dila_ex, "DiLA stride 1 예시")}
  <p class="cap">stride 1 · 문맥 5프레임 · action 유지. 샘플 A1·A2 작업대 S=0.95, 4행: GT · 문맥+등속 K · GT latent의 RAE 복원(디코더 상한) · <b>DiLA 예측 디코드</b> — 팔이 유령처럼 번지고 공이 사라진다.</p>
  <h4>결과 — 디코드한 예측에서 빨간 공 존재율, 스텝 1/2/3/4</h4>
  {table(["변형", "#A1 언덕", "#A2 두 공", "#P0 등속 · B2 지지 끝"], dila_rows)}
  <div class="verdict"><b>판정 NO-GO (모델 원인).</b> 미래 action을 유지하든 0이든, 시간축을 4·8·16 fps 어느 것으로 잡든, 구조 롤아웃은 1–4스텝 안에 장면 전체를 흐리고 공을 지운다(스텝 3 이후 존재율 0). latent 거리도 실제 인코더 latent 전부에서 멀어지고(track 3–7 &gt; V-JEPA 1.2–2.3) 물리/등속 선호가 없다(|dcos| ≤ 0.04). <b>참 미래 action을 넣어도 붕괴</b>(누설 참조 A1 0.33/0/0/0) → action 정책 탓이 아니라 학습 분포 밖 장면에서의 자기회귀 롤아웃 자체의 문제. 실패 위치는 V-JEPA(운동 전개 없음)보다 앞: 상태 유지 자체가 안 된다. 단순 외관(toy)에서는 붕괴가 한 스텝 늦다.</div>
</section>

<section class="block">
  <div class="model-head"><div><span class="eyebrow">모델 9 · 실사만 (README · 코드 · HF, 발사 없음) · 09-02 → 09-03</span><h2>A2World · DW05-Base · 그 외 후보</h2></div>{chip("wait", "A2World 연기 · DW05 제외 권고")}</div>
  {table(["모델", "입력 계약 (코드 확인)", "자산", "벤치마크 적합성", "판정"], [
      ("A2World (LogosRoboticsGroup, ECCV'26; Cosmos-Predict2-2B 멀티뷰)", "pretrained: 뷰 1–3개, <b>뷰당 조건 프레임 1장</b>(config max 1), 20 action(14-D = 7 × 2팔, zero-pad 허용) → 21프레임 @10 fps, 256². history 20프레임 + action path는 libero 변형만", "a2world-pretrained.pt 5.05 GB + base Sample-Action-Conditioned 4.05 GB(gated); Cosmos-Predict2 스택 신규 env(로컬 cosmos_v2w와 비호환)", "수동 문턱 씬은 1프레임만 보므로 속도 판독 불가. robot-context 씬(푸시 → 릴리스, action이 운동을 담음)에서만 의미 — 씬 작업 추가 필요(+4–6 h)", chip("wait", "Phase B/C 이후 (지훈 결정 09-03)")),
      ("DW05-Base (dexmal/opendw, Wan2.2 MoT)", "릴리스 런타임 <span class='mono'>create_dw05</span>가 history 1프레임 고정(예외), 단일 이미지 + 프롬프트(+32-D action/proprio), 8 fps", "번들 26 GB", "history에서 속도 식별 불가 → 사용자 Go/No-Go 기준상 NO-GO", chip("no", "제외 권고 (미결)")),
      ("LAWM-in-the-Wild · DINO-WM · LaWM · VLA-JEPA · MinD · π0-WM", "코드/체크포인트 미공개 또는 predictor 재학습 필요", "—", "무학습 추론 조건 불충족", chip("", "제외")),
  ])}
</section>
</div>

<div id="p-next" role="tabpanel" aria-labelledby="tab-next" hidden>
<section class="block">
  <h2>실패 위치 비교 — 같은 "못 한다"가 아니다</h2>
  {table(["모델", "어디서 막히는가", "근거 (재현 경로)"], [
      ("AdaWorld", "<b>액션 추출</b> — LAM이 물체 이동을 latent action에 담지 않음(전역 이동만)", "<span class='mono'>code/out/rollout_adaworld/</span> · PROGRESS (8)–(10)"),
      ("DreamDojo-Pretrain 2B", "<b>생성기의 도메인</b> — in-domain에서는 액션을 따르나 우리 씬에서 무응답; 행위자 외형은 필요조건일 뿐", "<span class='mono'>code/out/rollout_dreamdojo/</span> · PROGRESS (15)–(22)"),
      ("Cosmos-Predict2.5-2B base", "<b>장기</b> — 첫 5–8프레임만 물리적, 이후 드리프트·seed 의존; 물성 반응은 문맥 차이와 분리 불가", "<span class='mono'>code/out/sweep_cosmos/SUMMARY.md</span> · PROGRESS (24)–(32), (43)"),
      ("Cosmos-Predict2-2B V2W", "<b>방향 선택</b> — 관성 외삽: 반동·반환·낙하를 그리지 못함(텍스트·외관 무효), 진자만 부분", "<span class='mono'>code_vwm/artifacts/{bundle_a,phase_a,phase_b}/analysis*/</span> · PROGRESS (37)–(42)"),
      ("VERA DROID planner", "<b>방향의 S 의존 없음</b> — 사건은 그리나 무작위; 해상도(공 5 px)·미결·비용", "<span class='mono'>code_vwm/artifacts/phase_a/analysis_vera_s8/</span> · PROGRESS (39), (43)"),
      ("V-JEPA 2-AC", "<b>운동 전개 없음</b> — 예측 변위가 정적 성분, 물리/등속 선호 없음, EE state 무시", "<span class='mono'>code_vwm/artifacts/vjepa_ac/</span> · PROGRESS (41)"),
      ("DiLA", "<b>상태 유지 실패</b> — past-only 없음 + 자기회귀 붕괴(참 action에도)", "<span class='mono'>code_vwm/artifacts/dila/</span> · PROGRESS (42)"),
  ])}
  <p class="small muted">최종 산출물은 leaderboard가 아니라 <b>Physical Prediction Profile</b>(등속 / 연속 / 충격 / 접촉 전이 / 안정성 별 등급 + 원리 전이·문턱 정밀도·가림 강건성·법칙 일관성). 지금은 P0–P3 열이 Cosmos V2W에 대해서만 채워져 있고 전부 "기준선 수준"이다.</p>
</section>
<section class="block">
  <h2>벤치마크 규칙으로 굳은 것</h2>
  <ul class="plain">
    <li><b>action expressivity gate</b>: 이식된 액션이 우리 장면에서 전개되는가를 먼저 통과해야 rollout-vs-GT 수치를 보고한다. gate 실패는 그 자체로 결과이며 물리 이해 실패와 구분해 표기한다.</li>
    <li><b>leakage 금지</b>: latent action·문맥을 GT 미래에서 뽑지 않는다. 누설 참조(oracle)는 진단 전용으로 표시하고 점수에 넣지 않는다.</li>
    <li><b>MuJoCo 게이트 통과 전 모델 실행 금지</b>: 경계 이분법·에너지 감사·timestep 정수배·nuisance 불변·사건이 조건 창 밖·지평 내 결정.</li>
    <li><b>판독기는 GT에만 캘리브레이션 후 동결</b>, validity(소실·면적·형상)는 물리 판정과 분리. 등속 인코더(항상 "넘어감·직진")가 기준선.</li>
    <li><b>P0 제어를 같은 외관·카메라로 동봉</b>해 생성 실패와 물리 실패를 가른다 (Cosmos: 사건 없는 장면에서도 속도 ±25%).</li>
    <li><b>latent 트랙</b>: 분리도 게이트(sep &gt; 3 × 1 px 지터 floor) → 변위 방향 지표 dcos + 오라클 스케일 + 문맥 반전 제어.</li>
    <li><b>운영</b>: 수집 → 판독은 스크립트 안에서 연쇄(세션 워처 금지, setsid). 렌더는 OSMesa(pod 재기동 시 libosmesa6 재설치). 시각은 KST.</li>
  </ul>
</section>
<section class="block">
  <h2>다음 할 일 (승인·결정 필요 순)</h2>
  {table(["항목", "무엇", "비용", "누가"], [
      ("① VERA 확정", "정량 NO-GO 권고를 확정하거나, 정성 사례(반동 표현)로만 남길지 결정", N("0"), "지훈"),
      ("② Phase B 피드백 → 본실험", "B1 payload 판독(마스크 게이트 vs 씬 조정) 결정 후 Cosmos 24 seed × 22 = 528 롤아웃", N("≈ 2 h GPU"), "지훈 결정 → Claude 실행"),
      ("③ Phase C 자연 가림 짝", "gripper / fixture·clutter / viewpoint 3종 + 판독기 가림 창 예외 + 규칙 1–4 검증", N("6–8 h 구현 · 1.5–3 h GPU"), "Claude (승인 후)"),
      ("④ A2World", "robot-context 씬(푸시 → 릴리스) + MuJoCo → EE action 매핑 후 zero-action/릴리스 구간 평가", N("3–4.5 h env·feasibility + 4–6 h 씬"), "Phase B/C 이후"),
      ("⑤ 기록", "R3-79 Linear 코멘트(gate 실패 결과·설계 전환), HANDOFF §5 결정 반영", N("—"), "지훈 / Claude"),
  ])}
</section>
<section class="block">
  <h2>기록·페이지 포인터</h2>
  <ul class="plain small">
    <li>정본: <span class="mono">SimDROID/HANDOFF_SIMDROID_WMBENCH.md</span> (§3.6 데이터 · §3.7 AdaWorld/DreamDojo/Cosmos-2.5 · §3.8 결과 · §3.9 재정의 이후) · 일지 <span class="mono">PROGRESS.md</span> (1)–(43).</li>
    <li>이전 페이지: <a href="https://claude.ai/code/artifact/d972733e-09fb-4e14-a38e-a4777d16e079">씬·설계·모델 UI (09-03)</a> · <a href="https://claude.ai/code/artifact/5fe29905-bb88-4538-9a36-f477aada7211">Phase A 결과</a> · <a href="https://claude.ai/code/artifact/549af421-d4b2-4dfb-b5fd-60650bd52c2d">Phase B 씬</a> · <a href="https://claude.ai/code/artifact/8d91bcc8-51d7-4be7-be8b-1a29adef6fda">Bundle A 씬·결과</a> · <a href="https://claude.ai/code/artifact/a656ccb7-b67c-449f-b888-fa073f159a15">초기 임계 씬 6종 판독</a> · <a href="https://claude.ai/code/artifact/d36f12b2-eccb-42fb-85af-87e98d23f1db">WM 선정 노트 (08-26)</a>.</li>
    <li>코드: 팀 레포 <span class="mono">code/</span>(브랜치 <span class="mono">mnjihun/r3-79-rollout-vs-gt</span>, 씬 <span class="mono">core/scenes.py</span>, 어댑터 <span class="mono">exp/rollout_{{adaworld,dreamdojo,cosmos}}.py</span>) · <span class="mono">code_vwm/</span>(워크트리 <span class="mono">mnjihun/bundle-a-scenes</span>, 씬 <span class="mono">core/threshold/*</span>, 러너 <span class="mono">run_*_vwm.sh · run_vjepa_ac.sh · run_dila.sh</span>, 판독 <span class="mono">exp/analyze_*.py</span>). push는 지시 없이 하지 않음.</li>
    <li>이 페이지 생성기: <span class="mono">code_vwm/src/exp/build_simdroid_overview.py</span> · 5-task 스트립 <span class="mono">src/gen/render_pusher_strips.py</span>.</li>
  </ul>
</section>
</div>
<footer>SimDROID World Model 예측 검증 트랙 총람 · 2026-09-04 KST · 문지훈 / Claude · 수치는 <span class="mono">artifacts/*/summary.json</span>과 PROGRESS (35)–(43)에서 가져왔고, 박제현의 G1·G2 결과는 README·Linear R3-69 인용.</footer>
</main>
<script>
(function(){{
  var tabs=Array.prototype.slice.call(document.querySelectorAll('nav.tabs [role=tab]'));
  function show(id){{tabs.forEach(function(b){{var on=b.id===id;b.setAttribute('aria-selected',on?'true':'false');document.getElementById(b.getAttribute('aria-controls')).hidden=!on;}});try{{localStorage.setItem('sd_tab',id);}}catch(e){{}}}}
  tabs.forEach(function(b){{b.addEventListener('click',function(){{show(b.id);window.scrollTo({{top:0}});}});b.addEventListener('keydown',function(e){{var i=tabs.indexOf(b);if(e.key==='ArrowRight'){{var n=tabs[(i+1)%tabs.length];n.focus();show(n.id);}}if(e.key==='ArrowLeft'){{var p=tabs[(i+tabs.length-1)%tabs.length];p.focus();show(p.id);}}}});}});
  var saved=null;try{{saved=localStorage.getItem('sd_tab');}}catch(e){{}}
  if(saved&&document.getElementById(saved))show(saved);
  var gifs=Array.prototype.slice.call(document.querySelectorAll('img.gif')),btn=document.getElementById('motion'),still=false;
  function setStill(v){{still=v;gifs.forEach(function(g){{g.src=v?g.getAttribute('data-still'):g.getAttribute('data-anim');}});btn.textContent=v?'GIF 재생':'GIF 정지';btn.setAttribute('aria-pressed',v?'true':'false');}}
  btn.addEventListener('click',function(){{setStill(!still);}});
  if(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches)setStill(true);
}})();
</script>
"""
    Path(dst).write_text(html)
    print(f"OK -> {dst} ({len(html.encode()) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
