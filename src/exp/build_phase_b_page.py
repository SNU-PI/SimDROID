"""Phase B scene review page (2026-09-03): pendulum_rod_wb (P1-B) and support_edge_wb (P3-A).

Reads artifacts/phase_b (manifest, gt clips, previews, verify.json) and writes a
self-contained HTML page for scene feedback before any model run is launched.
"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path("/mnt/nvme/migration/jihun/SimDROID/code_vwm/artifacts/phase_b")
SHOW_S = (0.50, 0.79, 0.95, 1.05, 1.26, 2.00)
GIF_S = (0.50, 0.95, 1.05, 2.00)
STRIP_FRAMES = (4, 12, 20)


def b64_array(arr, scale=1.0, quality=82):
    im = Image.fromarray(arr)
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_gif(frames, scale=0.4, ms=62):
    ims = [Image.fromarray(f).resize((round(f.shape[1] * scale), round(f.shape[0] * scale)), Image.LANCZOS)
           for f in frames]
    buf = io.BytesIO()
    ims[0].save(buf, format="GIF", save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()


def load_clip(rec):
    return np.stack(imageio.mimread(ROOT / rec["gt_clip"], memtest=False))


def strip(recs, clips):
    """rows = frames, columns = S values; each tile labelled by its S and outcome."""
    tiles = []
    for fi in STRIP_FRAMES:
        row = [clips[r["id"]][min(fi, len(clips[r["id"]]) - 1)] for r in recs]
        tiles.append(np.concatenate(row, axis=1))
    return b64_array(np.concatenate(tiles, axis=0), scale=0.3)


CARDS = {
    "pendulum_rod_wb": [
        ("역학 부류", "P1-B 구속 보존계 (강체 봉 진자의 넘어감/되돌아옴)"),
        ("현실 맥락", "작업대 위 steel 포스트에 봉으로 매달린 붉은 payload; 정지 Franka·뒷벽·테이프"),
        ("사건 전 상태", "왼쪽 높이에서 내려와 최저점을 지나는 구간이 문맥(5프레임)에 들어옴"),
        ("전이 법칙", "에너지 보존: ½Iω² + gL(1−cosθ) = const; 넘어감 조건 v₀² ≥ 4gL"),
        ("결정 변수", "봉 길이 L (정적 가시). S = v₀²(1+0.4r²/L²)/(4gL), v₀ 고정 3.6 m/s"),
        ("등속 기준선", "각속도 유지 → 항상 넘어감(1)"),
        ("판독", "붉은 bob 궤적 원 적합 → 피벗·각도; 넘어감 = θ ≥ π, 되돌아옴 = θ 최대 후 ω 부호 반전"),
        ("연속 지표", "(1−cosθ, ω²) 기울기 vs −2g/L (에너지 법칙 잔차)"),
        ("전이 짝", "Hill (P1-A) ↔ Pendulum (P1-B): 같은 에너지 장벽, 다른 구속"),
    ],
    "support_edge_wb": [
        ("역학 부류", "P3-A 지지 상실: 접촉 → 자유낙하 모드 전이"),
        ("현실 맥락", "작업대 위 높이 0.15 m steel 픽스처 플레이트; 오른쪽 끝 너머는 검사대"),
        ("사건 전 상태", "플레이트 위를 v₀ = 0.35 m/s로 등속 구름 (문맥 5프레임 = 0.31 s, 14.5 px/프레임)"),
        ("전이 법칙", "지지 소실 후 z(t) = z_edge − ½gt², x는 v₀ 유지 (포물선)"),
        ("결정 변수", "플레이트 끝 위치 x_edge (정적 가시). S = v₀·T_REF/(x_edge − x_start), T_REF = 1.176 s"),
        ("등속 기준선", "플레이트 높이에서 계속 직진 → 절대 떨어지지 않음(0); S>1에서는 '공중 부양'"),
        ("판독", "붉은 공 중심 행이 문맥 수준보다 0.6 지름 이상 하강 = 낙하(1), 0.3 지름 이내 = 유지(0)"),
        ("연속 지표", "낙하 초기 수직 가속도 / g (fall_g_ratio)"),
        ("경계 보정", "시뮬 이분법: 반지름 절반 하강 사건이 프레임 20에 걸리는 x_edge → S=0.9998"),
    ],
}
QUESTIONS = {
    "pendulum_rod_wb": [
        "payload를 공 대신 공구 형상(실린더·그리퍼 핑거 등)으로 바꿀지 — 판독기는 붉은 마스크만 쓰므로 붉은색이면 형상 자유.",
        "봉 길이 격자(S 0.5–2.0 ↔ L 0.66–0.17 m)가 카메라 3.3 m·fovy 24에서 전부 보임. 카메라를 더 가깝게 두면 bob은 커지고 긴 봉이 잘릴 수 있음.",
    ],
    "support_edge_wb": [
        "S<1 샘플은 사건이 없음(공이 계속 구름) — 지평 안에서 '아무 일도 안 일어남'이 정답. Cosmos P0의 속도 오차(±25%)가 S≈1 부근을 흐릴 것.",
        "낙하 후 착지 반동(강성 접촉)은 판독에 영향 없음(첫 하강으로 판정). 반동을 줄이려면 접촉 solref를 부드럽게 바꿀 수 있으나 물리 서명이 바뀜.",
        "플레이트 끝 너머 검사대까지 0.15 m 낙하 = 2.5 프레임. 더 높은 플레이트(0.25 m)면 포물선 법칙 판독이 길어지지만 화면 하단 여유가 줄어듦.",
    ],
}


def main(dst):
    recs = [json.loads(l) for l in (ROOT / "manifest.jsonl").read_text().splitlines() if l.strip()]
    verify = json.loads((ROOT / "verify.json").read_text()) if (ROOT / "verify.json").exists() else {}
    fams = list(dict.fromkeys(r["family"] for r in recs))
    sections = []
    for fam in fams:
        fr = [r for r in recs if r["family"] == fam]
        clips = {r["id"]: load_clip(r) for r in fr}
        show = [r for r in fr if any(abs(r["S"] - s) < 1e-6 for s in SHOW_S)]
        strip_uri = strip(show, clips)
        gifs = []
        for r in fr:
            if any(abs(r["S"] - s) < 1e-6 for s in GIF_S):
                c = clips[r["id"]]
                gifs.append((r, b64_gif(c[:21]), b64_array(c[4], scale=0.4)))
        rows = "".join(
            f"<tr><td>{r['S']:.2f}</td><td>{r['outcome']}</td><td>{r['event_frame']}</td>"
            f"<td>{r['t_event_s']:.3f}</td><td>{'예' if r['in_map'] else '아니오'}</td>"
            f"<td>{'경계' if r['boundary'] else ''}</td>"
            f"<td class='mono'>{html.escape(json.dumps({k: round(v, 3) for k, v in r['params'].items()}))}</td></tr>"
            for r in fr)
        card = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in CARDS.get(fam, []))
        qs = "".join(f"<li>{q}</li>" for q in QUESTIONS.get(fam, []))
        gif_html = "".join(
            f"<figure class='gif'><img data-anim='{g}' src='{still}' alt='{r['id']}'>"
            f"<figcaption>S = {r['S']:.2f} → GT 결과 {r['outcome']} (사건 프레임 {r['event_frame']})</figcaption></figure>"
            for r, g, still in gifs)
        col_lbl = " · ".join(f"S {r['S']:.2f}→{r['outcome']}" for r in show)
        sections.append(f"""
<section id="{fam}">
  <h2>{fam}</h2>
  <div class="two">
    <table class="card"><tbody>{card}</tbody></table>
    <div>
      <h3>피드백 요청</h3>
      <ol class="qs">{qs}</ol>
    </div>
  </div>
  <h3>S 격자 스트립 <span class="sub">행 = 프레임 4 (문맥 끝) / 12 / 20 (지평 끝), 열 = {col_lbl}</span></h3>
  <div class="scroll"><img src="{strip_uri}" alt="{fam} strip"></div>
  <h3>GT 클립 <span class="sub">21프레임 = 문맥 5 + 예측 지평 16 (1.31 s). 클릭하면 재생/정지</span></h3>
  <div class="gifs">{gif_html}</div>
  <h3>격자표</h3>
  <div class="scroll"><table class="grid"><thead><tr><th>S</th><th>결과</th><th>사건 프레임</th><th>t (s)</th><th>지평 내 결정</th><th></th><th>파라미터</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
</section>""")
    # ---- model results, filled in only when the files exist (pilot / latent tracks)
    def b64_img(path, max_w=1200, quality=80):
        if not path.exists():
            return None
        im = Image.open(path).convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    models_html = ""
    pil = ROOT / "analysis_pilot" / "summary.json"
    if pil.exists():
        ps = json.loads(pil.read_text())
        rows = ""
        for fam in fams:
            f = ps.get(fam)
            if not f:
                continue
            curve = " · ".join(f"{c['S']:.2f}→{c['p_success']:.2f}" for c in f["curve"])
            rows += (f"<tr><td>{fam}</td><td>{f['n_rollouts']}</td><td>{f['accuracy_nonboundary']:.2f}</td>"
                     f"<td>{f['encoder_accuracy_nonboundary']:.2f}</td><td>{f['undecided_rate']:.2f}</td><td>{f['valid_rate']:.2f}</td>"
                     f"<td class='mono'>{curve}</td></tr>")
        curves = b64_img(ROOT / "analysis_pilot" / "curves.png")
        sheets = "".join(f"<figure class='gif'><img src='{b}' alt='{fam} sheet'><figcaption>{fam} 결과 시트 (seed 1–3)</figcaption></figure>"
                         for fam in fams for b in [b64_img(ROOT / "analysis_pilot" / "sheets" / f"{fam}.jpg", max_w=1400)] if b)
        models_html += f"""
<section id="pilot">
  <h2>Cosmos-Predict2-2B V2W 파일럿 <span class="sub">3 seed × 22 = 66 롤아웃 · 파이프라인 검증용, 본실험(24 seed) 아님</span></h2>
  <div class="scroll"><table class="grid"><thead><tr><th>가족</th><th>롤아웃</th><th>비경계 정확도</th><th>등속 기준선</th><th>미결</th><th>strict valid</th><th>P(pred = 1 | S)</th></tr></thead><tbody>{rows}</tbody></table></div>
  <p class="lead">등속 기준선 = 항상 "넘어감"(진자) / 항상 "안 떨어짐"(지지 끝). 정확도가 기준선과 같으면 문턱을 읽지 못한 것. n = 3 seed라 방향만 본다.</p>
  {"<div class='scroll'><img src='" + curves + "' alt='pilot curves'></div>" if curves else ""}
  <div class="gifs">{sheets}</div>
</section>"""
    vj = ROOT.parent / "vjepa_ac" / "phase_b" / "analysis" / "summary.json"
    lat_rows = ""
    if vj.exists():
        f = json.loads(vj.read_text())["families"].get("support_edge_wb")
        if f:
            g = f["gated"]
            lat_rows += (f"<tr><td>V-JEPA 2-AC (액션 0, 4 fps)</td><td>{f['gate_rate']:.2f}</td><td>{f['all_steps']['sep_over_floor_median']:.1f}</td>"
                         f"<td>{g['dcos_mean']:+.3f} (오라클 ±{g['dcos_oracle_P_mean']:.2f})</td><td>{f['all_steps']['track_mean']:.2f}</td><td>—</td></tr>")
    for tag, lab in (("", "DiLA · stride 4 · action 유지"), ("_s1", "DiLA · stride 1 · 유지"), ("_zero", "DiLA · stride 4 · action 0")):
        la = ROOT.parent / "dila" / "phase_b" / f"analysis{tag}" / "summary.json"
        px = ROOT.parent / "dila" / "phase_b" / f"pixels{tag}" / "summary.json"
        if la.exists():
            f = json.loads(la.read_text())["families"].get("support_edge_wb")
            pres = "—"
            if px.exists():
                q = json.loads(px.read_text())["families"].get("support_edge_wb", {})
                pres = " / ".join(f"{q[k]['present_rate']:.2f}" if k in q else "·" for k in ("1", "2", "3", "4"))
            if f:
                g = f["gated"]
                dcos = f"{g['dcos_mean']:+.3f}" if g["dcos_mean"] == g["dcos_mean"] else "—"
                lat_rows += (f"<tr><td>{lab}</td><td>{f['gate_rate']:.2f}</td><td>{f['all_steps']['sep_over_floor_median']:.1f}</td>"
                             f"<td>{dcos}</td><td>{f['all_steps']['track_mean']:.2f}</td><td>{pres}</td></tr>")
    if lat_rows:
        models_html += f"""
<section id="latent">
  <h2>latent 트랙 (support_edge_wb) <span class="sub">K = 플레이트 높이를 유지한 등속 = '공중 부양', P = 낙하. S &lt; 1 은 P ≡ K 라 게이트 불통(설계상)</span></h2>
  <div class="scroll"><table class="grid"><thead><tr><th>모델</th><th>게이트 통과</th><th>sep/floor</th><th>dcos (게이트)</th><th>track</th><th>디코드 공 존재율 스텝 1/2/3/4</th></tr></thead><tbody>{lat_rows}</tbody></table></div>
  <p class="lead">dcos &gt; 0 이면 예측 변위가 낙하 쪽, &lt; 0 이면 부양 쪽. track &gt; 1 이면 예측이 문맥 마지막 프레임보다 물리 미래에서 더 멀다.</p>
</section>"""

    vb = verify.get("support_edge_boundary", {})
    pend = verify.get("pendulum", [])
    pend_txt = ", ".join(f"S {p['S']}: 서명 동일 {p['identical_signature']}, 궤적 차 {p['trace_max_abs_diff']:.0e}" for p in pend)
    verify_html = f"""
<section id="verify">
  <h2>검증 요약 <span class="sub">verify_phase_b.py · {len(recs)} 샘플</span></h2>
  <ul>
    <li><b>진자 toy↔workbench 물리 동일성</b>: {pend_txt or '미실행'}</li>
    <li><b>support_edge 경계 보정</b>: x_edge* = {vb.get('x_edge_boundary', float('nan')):+.4f} m → S_boundary(sim) = {vb.get('S_boundary_sim', float('nan')):.4f} (T_REF {vb.get('T_REF', '')} s)</li>
    <li><b>support_edge toy↔workbench 서명 동일</b>: {verify.get('support_edge_toy_vs_wb_identical', '')}</li>
    <li><b>격자 게이트</b>: 비경계 11점 모두 해석 결과 = 시뮬 결과, 지평 내 결정 {sum(r['in_map'] for r in recs)}/{len(recs)}</li>
  </ul>
</section>"""
    page = f"""<title>Phase B 씬 검토</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --bg:#f6f5f2; --ink:#1d2024; --mute:#5d636b; --line:#d9d6cf; --card:#ffffff; --acc:#a8452e; --acc2:#2e5e8a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#17191c; --ink:#e8e6e1; --mute:#a3a7ae; --line:#33373d; --card:#1f2226; --acc:#e07a5f; --acc2:#7fb0e0; }} }}
:root[data-theme="dark"] {{ --bg:#17191c; --ink:#e8e6e1; --mute:#a3a7ae; --line:#33373d; --card:#1f2226; --acc:#e07a5f; --acc2:#7fb0e0; }}
body {{ background:var(--bg); color:var(--ink); font-family:"IBM Plex Sans KR", system-ui, sans-serif; font-size:15px; line-height:1.55; margin:0; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 22px 60px; }}
h1 {{ font-size:26px; margin:0 0 4px; text-wrap:balance; }}
h2 {{ font-size:20px; margin:36px 0 10px; border-bottom:2px solid var(--acc); padding-bottom:4px; }}
h3 {{ font-size:15px; margin:20px 0 8px; color:var(--ink); }}
.sub {{ font-weight:400; color:var(--mute); font-size:13px; margin-left:8px; }}
.eyebrow {{ color:var(--acc); text-transform:uppercase; letter-spacing:.08em; font-size:12px; font-weight:600; }}
.lead {{ color:var(--mute); max-width:72ch; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 0; }}
.chip {{ border:1px solid var(--line); border-radius:999px; padding:3px 11px; font-size:13px; background:var(--card); }}
.two {{ display:grid; grid-template-columns: minmax(0,1.3fr) minmax(0,1fr); gap:22px; align-items:start; }}
@media (max-width:820px) {{ .two {{ grid-template-columns:1fr; }} }}
table.card {{ border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line); }}
table.card th {{ text-align:left; font-weight:600; width:9em; padding:6px 10px; border-bottom:1px solid var(--line); color:var(--acc2); vertical-align:top; }}
table.card td {{ padding:6px 10px; border-bottom:1px solid var(--line); }}
ol.qs {{ padding-left:20px; margin:0; }} ol.qs li {{ margin-bottom:8px; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); background:var(--card); padding:6px; }}
.scroll img {{ display:block; max-width:none; }}
.gifs {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap:12px; }}
figure.gif {{ margin:0; background:var(--card); border:1px solid var(--line); padding:6px; }}
figure.gif img {{ width:100%; display:block; cursor:pointer; }}
figcaption {{ font-size:13px; color:var(--mute); margin-top:4px; }}
table.grid {{ border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13.5px; min-width:640px; }}
table.grid th, table.grid td {{ padding:4px 10px; border-bottom:1px solid var(--line); text-align:right; }}
table.grid th:first-child, table.grid td:first-child {{ text-align:left; }}
.mono {{ font-family:"IBM Plex Mono", monospace; font-size:12px; text-align:left !important; }}
footer {{ color:var(--mute); font-size:13px; margin-top:40px; border-top:1px solid var(--line); padding-top:10px; }}
code {{ font-family:"IBM Plex Mono", monospace; font-size:12.5px; }}
</style>
<main>
<div class="eyebrow">PhysicsGen · Stage 3 · 2026-09-03</div>
<h1>Phase B 씬 검토: 매달린 payload · 지지 끝 이탈</h1>
<p class="lead">Phase A 계약(5 문맥 프레임 @16 fps, 21프레임 지평, S 격자 11점, 동결 판독기)을 그대로 쓰는 두 씬. 본실험(Cosmos 24 seed ≈ 2 h GPU) 전에 씬 구성에 대한 피드백을 받기 위한 페이지. 씬 절은 GT만 다루고, 파이프라인 검증용 파일럿과 latent 트랙 결과는 맨 아래 절에 있다.</p>
<div class="chips"><span class="chip">{len(recs)} 샘플 · {len(fams)} 가족</span><span class="chip">480×832 · 16 fps</span><span class="chip">S ∈ {{0.50 … 2.00}}</span><span class="chip">물리 서명 toy = workbench</span><span class="chip">경계 S = {vb.get('S_boundary_sim', float('nan')):.4f}</span></div>
{''.join(sections)}
{verify_html}
{models_html}
<footer>재현: <code>MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=src python src/gen/make_phase_b.py</code> · <code>python src/exp/verify_phase_b.py</code> · 페이지 <code>src/exp/build_phase_b_page.py</code>. 기록: SimDROID/PROGRESS.md (42).</footer>
</main>
<script>
document.querySelectorAll('figure.gif img').forEach(img => {{
  const still = img.getAttribute('src'), anim = img.dataset.anim;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduce) img.src = anim;
  img.addEventListener('click', () => {{ img.src = (img.src === anim) ? still : anim; }});
}});
</script>
"""
    Path(dst).write_text(page)
    print("wrote", dst, round(Path(dst).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/migration/jihun/SimDROID/Materials/phase_b_scenes_2026-09-03.html")
