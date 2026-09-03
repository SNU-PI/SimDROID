"""Build the Phase A results artifact page (HTML with inline assets).

Inputs: artifacts/phase_a/{analysis|analysis_pilot}/summary.json + samples.csv
(workbench), artifacts/bundle_a/analysis/summary.json (toy, 24 seeds),
artifacts/phase_a/verify.json, previews/*.png, optional VERA results
(artifacts/phase_a/analysis_vera/summary.json, external/vera_smoke_out/receipt.json).
Usage: build_phase_a_page.py <analysis_dir> <dst.html>
"""
from __future__ import annotations

import base64, csv, io, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/nvme/migration/jihun/SimDROID/code_vwm/artifacts")
EXT = Path("/mnt/nvme/migration/jihun/SimDROID/external")
INK, SUB, LINE = "#21252C", "#626B78", "#E1E4EA"
ACCENT, BLUE, OK, WARN = "#C4402C", "#3E68A8", "#256E46", "#8A5A14"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": LINE, "axes.labelcolor": INK, "xtick.color": SUB,
                     "ytick.color": SUB, "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": "white",
                     "axes.facecolor": "white"})
FAM = {"hill_roll": "A1 구름 언덕", "two_ball": "A2 두 공 충돌"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x, nd=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f}%"


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def b64png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def b64img(path, scale=0.5, quality=80):
    im = Image.open(path).convert("RGB")
    if scale != 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def curve_points(curve):
    S, p, lo, hi, und = [], [], [], [], []
    for c in curve:
        nd = c["n"] - c["n_undecided"]
        k = round(c["p_success"] * nd) if nd and not np.isnan(c["p_success"]) else 0
        S.append(c["S"]); p.append(k / nd if nd else np.nan)
        l, h = wilson(k, nd); lo.append(l); hi.append(h); und.append(c["n_undecided"] / c["n"] if c["n"] else 0)
    return np.array(S), np.array(p), np.array(lo), np.array(hi), np.array(und)


def main(analysis_dir, dst):
    wb = json.loads((Path(analysis_dir) / "summary.json").read_text())
    toy = json.loads((ROOT / "bundle_a/analysis/summary.json").read_text())
    verify = json.loads((ROOT / "phase_a/verify.json").read_text())
    rows = list(csv.DictReader(open(Path(analysis_dir) / "samples.csv")))
    seeds = sorted({int(r["seed"]) for r in rows})
    n_seeds = len(seeds)
    n_roll = len(rows)
    vera = None
    vp = ROOT / "phase_a/analysis_vera_s8/summary.json"          # seeds 1-8 (2026-09-03) when present
    if not vp.exists():
        vp = ROOT / "phase_a/analysis_vera/summary.json"
    if vp.exists():
        vera = json.loads(vp.read_text())
    receipt = json.loads((EXT / "vera_smoke_out/receipt.json").read_text()) if (EXT / "vera_smoke_out/receipt.json").exists() else None

    # ---- figure: toy vs workbench decision curves
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=170, sharey=True)
    for ax, fam in zip(axes, ("hill_roll", "two_ball")):
        ax.plot([0.44, 1, 1, 2.06], [0, 0, 1, 1], color=BLUE, lw=1.6, alpha=.8, label="MuJoCo GT", zorder=2)
        ax.axhline({"hill_roll": 1, "two_ball": 0}[fam], color=SUB, ls=(0, (4, 3)), lw=1.0, label="등속 인코더", zorder=1)
        ax.axvline(1.0, color=LINE, lw=1)
        for name, summ, col, mk, off in (("toy · Bundle A (24 seed)", toy[fam], SUB, "s", -0.012),
                                          (f"작업대 · Phase A ({n_seeds} seed)", wb[fam + "_wb"], ACCENT, "o", 0.012)):
            S, p, lo, hi, und = curve_points(summ["curve"])
            for s, l, h in zip(S, lo, hi):
                ax.plot([s + off] * 2, [l, h], color=col, lw=1.1, alpha=.5, zorder=3)
            ax.plot(S + off, p, marker=mk, ms=4.5, color=col, lw=1.6, label=name, zorder=4)
        ax.set_title(FAM[fam], fontsize=11); ax.set_xlim(0.44, 2.06); ax.set_ylim(-0.05, 1.06)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0]); ax.set_yticks([0, 0.5, 1.0]); ax.set_xlabel("S (saturation)")
    axes[0].set_ylabel("P(통과 방향 예측 | 판정 성립)")
    axes[1].legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout(); img_curves = b64png(fig)

    # ---- P0 control
    ctrl = wb.get("controls", {}).get("kin_roll")
    p0_rows = ""
    if ctrl:
        for s in ctrl["per_sample"]:
            p0_rows += (f'<tr><td class="num">{s["v0"]:.2f}</td><td class="num">{s["n"]}</td><td class="num">{pct(s["p_continue"])}</td>'
                        f'<td class="num">{fmt(s["speed_ratio"])}</td><td class="num">{fmt(s["kin_err_px"],1)}</td>'
                        f'<td class="num">{fmt(s["gt_err_px"],1)}</td><td class="num">{pct(s["valid_rate"])} / {pct(s["valid_frac"])}</td></tr>\n')

    # ---- family table toy vs wb
    def famrow(fam):
        t, w = toy[fam], wb[fam + "_wb"]
        law_t = f'{fmt(t["gen_law_slope_mean"],1)} vs GT {fmt(t["gt_law_slope_mean"],1)}' if t.get("gt_law_slope_mean") is not None else f'Σmv {fmt(t["momentum_ratio_mean"])}'
        law_w = f'{fmt(w["gen_law_slope_mean"],1)} vs GT {fmt(w["gt_law_slope_mean"],1)}' if w.get("gt_law_slope_mean") is not None else f'Σmv {fmt(w["momentum_ratio_mean"])}'
        return (f'<tr><td>{FAM[fam]}</td><td class="num">{pct(t["accuracy_nonboundary"])} / {pct(w["accuracy_nonboundary"])}</td>'
                f'<td class="num">{pct(t["encoder_accuracy_nonboundary"])}</td><td class="num">{pct(t["undecided_rate"])} / {pct(w["undecided_rate"])}</td>'
                f'<td class="num">{pct(t["valid_rate"])} / {pct(w["valid_rate"])}</td><td class="num">{law_t} → {law_w}</td></tr>\n')
    fam_rows = famrow("hill_roll") + famrow("two_ball")

    # ---- prechecks
    pre = wb.get("prechecks", {})
    hill_pre = [v for k, v in pre.items() if k.startswith("hill_roll_wb_pre")]
    wall_pre = [v for k, v in pre.items() if k.startswith("wall_bounce_wb_pre")]
    hill_pre_pass = sum(v["prediction"] == 1 for v in hill_pre)
    wall_ok = sum(v["prediction"] == v["gt"] for v in wall_pre)
    toy_pre = toy.get("prechecks", {})
    toy_wall_ok = sum(v["prediction"] == v["gt"] for k, v in toy_pre.items() if k.startswith("wall_bounce_pre"))
    toy_wall_n = sum(1 for k in toy_pre if k.startswith("wall_bounce_pre"))

    # ---- verification
    vrows = ""
    for k, lab in (("hill_roll", "A1"), ("two_ball", "A2"), ("wall_bounce", "A2-0")):
        v = verify[k]
        vrows += f'<tr><td>{lab}</td><td>{"동일" if v["identical"] else "차이: " + ", ".join(v["diff_keys"])}</td><td class="num">{v["n_pairs"]}</td><td class="mono small">{", ".join(v["colliding_geoms"])}</td></tr>\n'
    kin_v = verify["kin_roll"]
    kin_line = " · ".join(f'v0={r["v0"]:.2f}: 속도비 {r["speed_ratio"]:.3f}, slip {r["slip_max"]:.4f}' for r in kin_v)

    # ---- previews
    prev = {}
    for fam in ("kin_roll", "hill_roll_wb", "two_ball_wb"):
        p = ROOT / "phase_a/previews" / f"{fam}.png"
        if p.exists():
            prev[fam] = b64img(p, scale=0.6, quality=78)

    # ---- VERA strips (GT vs generated side view), a few representative samples
    def vera_strip(sid, seed=1):
        gp, rp = ROOT / "phase_a/vera" / f"{sid}.npz", ROOT / f"phase_a/vera_out/seed_{seed:02d}" / sid / "rollout.npz"
        if not (gp.exists() and rp.exists()):
            return None
        gt = np.load(gp)["side_view"]; gen = np.load(rp)["side_view"]
        idx = [28, 40, 52, 64, 76, 88, 100]
        row_gt = np.concatenate([gt[min(i, len(gt) - 1)] for i in idx], axis=1)
        row_gen = np.concatenate([gen[min(i, len(gen) - 1)] for i in idx], axis=1)
        strip = np.concatenate([row_gt, row_gen], axis=0)
        im = Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    strips_html = ""
    for sid, lab in (("hill_roll_wb_02", "A1 S=0.79 (GT: 반환)"), ("hill_roll_wb_06", "A1 S=1.05 (GT: 통과)"),
                     ("two_ball_wb_02", "A2 S=0.79 (GT: 직진)"), ("two_ball_wb_08", "A2 S=1.26 (GT: 반전)"),
                     ("kin_roll_03", "P0 v0=0.75"), ("wall_bounce_wb_pre", "A2-0 벽 반동")):
        b = vera_strip(sid)
        if b:
            strips_html += (f'<div class="media scroll"><img src="data:image/jpeg;base64,{b}" alt="{lab} GT vs VERA"></div>'
                            f'<p class="cap">{lab} — 위 GT, 아래 VERA(seed 1). 열 = 프레임 28(문맥 끝)·40·52·64·76·88·100 (100 Hz 캡처 = 0.28–1.0 s).</p>\n')

    # ---- VERA block
    if vera:
        vrows_vera = ""
        for fam, v in vera.items():
            def _lab(smp):
                if smp["S"] is not None:
                    return f"S={smp['S']:.2f}"
                return f"v0={smp['v0']:.2f}" if smp.get("v0") is not None else "pre"
            cells = " ".join(f"{_lab(smp)}: {smp['gt']}→<b>{smp['pred']}</b>" for smp in v["per_sample"])
            vrows_vera += f'<tr><td>{fam}</td><td class="num">{v["n"]}</td><td class="num">{pct(v["accuracy"])}</td><td class="num">{pct(v["undecided"])}</td><td class="num">{pct(v["valid_frac"])}</td><td class="small">{cells}</td></tr>\n'
        vera_block = f"""<div class="tbl-wrap"><table>
    <tr><th>가족</th><th class="num">n</th><th class="num">정확도</th><th class="num">미결</th><th class="num">valid_frac</th><th>샘플별 GT→예측 (1=통과/반전/계속)</th></tr>
{vrows_vera}  </table></div>"""
    else:
        vera_block = '<p class="small muted">VERA 롤아웃 판독은 아직 없음 — 추론 완료 후 갱신.</p>'
    rc = ""
    if receipt:
        g = receipt["generations"]
        rc = (f'로드 {receipt["load_s"]:.0f} s · VRAM {receipt["vram_after_load_gb"]:.0f} GB · 문맥 {receipt["ctx_frames"]}프레임 → 청크 {receipt["chunk_frames"]}프레임 · '
              f'생성 {g[0]["gen_s"]:.0f} s (GPU 단독) / {g[1]["gen_s"]:.0f} s (Cosmos 동시 실행)')

    # ---- V-JEPA 2-AC latent track (2026-09-03)
    VJ = ROOT / "vjepa_ac"
    def _load(pth):
        return json.loads(pth.read_text())["families"] if pth.exists() else None
    vj = _load(VJ / "phase_a/analysis/summary.json")
    vj_c3 = _load(VJ / "phase_a/analysis_c3/summary.json")
    vj_rev = _load(VJ / "phase_a/analysis_rev/summary.json")
    vj_toy = _load(VJ / "bundle_a/analysis/summary.json")
    VJ_FAM = {"kin_roll": "P0 등속 제어", "hill_roll_wb": "A1 구름 언덕", "two_ball_wb": "A2 두 공 충돌", "wall_bounce_wb_pre": "A2-0 벽 반동"}
    vj_rows, vj_var_rows, vj_block = "", "", ""
    if vj:
        for fam in ("kin_roll", "hill_roll_wb", "two_ball_wb", "wall_bounce_wb_pre"):
            if fam not in vj:
                continue
            a, g, bs = vj[fam]["all_steps"], vj[fam]["gated"], vj[fam]["by_step"]
            cosP = " / ".join(fmt(bs[k]["cos_P_mean"]) for k in ("1", "2", "3", "4"))
            cosK = " / ".join(fmt(bs[k]["cos_K_mean"]) for k in ("1", "2", "3", "4"))
            trk = f'{fmt(bs["1"]["track_mean"])} → {fmt(bs["4"]["track_mean"])}'
            if fam == "kin_roll":
                dc, orc, pp = "P≡K", "—", "—"
            else:
                dc, orc, pp = f'{g["dcos_mean"]:+.3f}', f'±{fmt(g["dcos_oracle_P_mean"])}', pct(g["p_dcos_pos"])
            vj_rows += (f'<tr><td>{VJ_FAM[fam]}</td><td class="num">{a["n"]}</td><td class="num">{pct(vj[fam]["gate_rate"])}</td>'
                        f'<td class="num">{fmt(a["sep_over_floor_median"],1)}</td><td class="num">{trk}</td>'
                        f'<td class="num small">{cosP}<br>{cosK}</td><td class="num">{dc}</td><td class="num">{pp}</td><td class="num">{orc}</td></tr>\n')
        def _kin(src):
            return " / ".join(fmt(src["kin_roll"]["by_step"][k]["cos_P_mean"]) for k in ("2", "3", "4")) if src and "kin_roll" in src else "—"
        def _dc(src, fam):
            return f'{src[fam]["gated"]["dcos_mean"]:+.3f}' if src and fam in src else "—"
        for lab, src, h, tb in (("기본 (문맥 2프레임 · 작업대)", vj, "hill_roll_wb", "two_ball_wb"),
                                 ("문맥 3프레임 (0, 4, 8) · 사건 전인 hill·P0만", vj_c3, "hill_roll_wb", "two_ball_wb"),
                                 ("문맥 순서 반전 (운동 단서 뒤집기)", vj_rev, "hill_roll_wb", "two_ball_wb"),
                                 ("toy 외관 (Bundle A)", vj_toy, "hill_roll", "two_ball")):
            vj_var_rows += f'<tr><td>{lab}</td><td class="num">{_kin(src)}</td><td class="num">{_dc(src, h)}</td><td class="num">{_dc(src, tb)}</td></tr>\n'
        vj_curves = b64img(VJ / "phase_a/analysis/curves.png", scale=0.9, quality=82) if (VJ / "phase_a/analysis/curves.png").exists() else None
        vj_strips = ""
        for sid, lab in (("hill_roll_wb_04", "A1 S=0.95 (GT: 반환)"), ("two_ball_wb_04", "A2 S=0.95 (GT: 직진, 파란 공이 앞서감)")):
            pp_ = VJ / "phase_a/previews" / f"{sid}.png"
            if pp_.exists():
                vj_strips += (f'<div class="media"><img src="data:image/jpeg;base64,{b64img(pp_, scale=0.5, quality=80)}" alt="{lab} 후보 미래"></div>'
                              f'<p class="cap">{lab} — 256² 전용 카메라. 위: 문맥 2프레임(t=0, 0.25 s) + 물리 미래 P(0.5/0.75/1.0/1.25 s) · 가운데: 등속 counterfactual K(문맥 속도 유지, MuJoCo qpos 재배치 렌더) · 아래: 지터 J(물리 위치 + 빨간 공 1 px). 예측기는 위 두 프레임의 latent만 받는다.</p>\n')
        vj_block = f"""<section>
  <div class="sec-head"><span class="n">06</span><h2>V-JEPA 2-AC latent 트랙 — 세 번째 모델</h2></div>
  <p>픽셀을 생성하지 않는 예측기를 같은 씬에 넣었다. V-JEPA 2-AC(ViT-g 인코더 + action-conditioned predictor, DROID 4 fps로 post-train)에 <b>로봇 액션 0</b>·상수 EE state를 주고, 사건 전 두 프레임(0.25 s 간격)의 latent에서 4스텝(0.5–1.25 s)을 자기회귀로 예측했다.
  판독은 latent 공간에서만 한다: 같은 시각의 <b>물리 미래 P</b>와 <b>등속 counterfactual K</b>를 MuJoCo로 렌더해 인코딩한 뒤, 예측이 어느 쪽에 가까운지 본다. 누출 없음(액션 0, 문맥은 사건 전만). 오프셋 0/1/2가 seed를 대신한다(결정적 모델).</p>
  <div class="kv">
    <dt>게이트</dt><dd>인코더가 두 미래를 구분하는가: sep = d(z_P, z_K)를 <b>빨간 공 1 px 이동</b>의 latent 거리(floor)와 비교, sep &gt; 3·floor인 (샘플, 스텝)만 판독. P0 제어(P≡K)는 sep ≈ 0.1·floor로 자기검증.</dd>
    <dt>지표</dt><dd>변위 방향 dcos = cos(ẑ−z_last, z_P−z_last) − cos(ẑ−z_last, z_K−z_last). 양수 = 물리 쪽. 물리 오라클(ẑ=z_P)은 +(1−cos_PK) ≈ +0.45, 등속 오라클은 그 음수. freeze(ẑ=z_last)는 변위 0이라 이 지표에서 중립 — 거리 기반 Δ에서 생기는 "반환하는 물리 미래는 문맥에 가깝다" 교란을 제거한다.</dd>
    <dt>규모</dt><dd>Phase A 29행 × 3오프셋 = 87 롤아웃(+ toy 24행 × 3, 문맥 3프레임·반전 제어) · 생성 ≈ 5 s/샘플(3오프셋) · VRAM 6 GB · 자산 ckpt 11.8 GB.</dd>
  </div>
{vj_strips}  <div class="tbl-wrap"><table>
    <tr><th>씬</th><th class="num">스텝 행</th><th class="num">게이트 통과</th><th class="num">sep/floor</th><th class="num">track 스텝1→4</th><th class="num">cos_P / cos_K (스텝 1–4)</th><th class="num">dcos (게이트)</th><th class="num">P(dcos&gt;0)</th><th class="num">오라클</th></tr>
{vj_rows}  </table></div>
  <p class="small muted">track = d(ẑ, z_P)/d(z_last, z_P): 1보다 크면 예측 latent가 마지막 문맥 프레임보다 물리 미래에서 멀다. cos_P·cos_K = 예측 변위와 각 후보 변위의 코사인(스텝 1/2/3/4). 오라클 = 물리 오라클의 dcos(등속 오라클은 그 음수).</p>
  {"<div class='media'><img src='data:image/jpeg;base64," + vj_curves + "' alt='V-JEPA latent track curves'></div><p class='cap'>위: 마지막 스텝 거리차 Δ/sep — 예측기(빨강)는 freeze 기준선(회색)의 축소판. 가운데: 분리도 sep/floor·motion/floor와 게이트선. 아래: 변위 방향 dcos(전체 토큰 ●, P≠K 패치 ▲)와 오라클 범위(회색 띠). 어느 S에서도 dcos는 0 근처이고 S=1에서 부호가 바뀌지 않는다.</p>" if vj_curves else ""}
  <div class="tbl-wrap"><table>
    <tr><th>변형</th><th class="num">P0 cos_P (스텝 2/3/4)</th><th class="num">A1 dcos</th><th class="num">A2 dcos</th></tr>
{vj_var_rows}  </table></div>
  <div class="ask"><p><b>판정: NO-GO (모델 원인).</b> (1) 인코더 게이트는 가까스로 통과한다(sep/floor 2–4.5, 통과율 A1 47%·A2 82%) — 256²에서 공 11–17 px가 한계다. (2) 예측기는 수동 물체의 운동을 외삽하지 않는다: 예측 변위의 정렬(cos ≈ 0.2)이 문맥 순서를 뒤집어도, 문맥을 3프레임으로 늘려도, 공 속도를 0.3→0.9 m/s로 바꿔도 그대로다 — 운동 단서가 아니라 정적 성분에서 오는 정렬이다. EE state를 바꿔도 예측은 동일(상관 1.00). (3) 물리 미래와 등속 미래를 구분하지 않는다(|dcos| ≤ 0.05, 오라클 ±0.45, S 무관, toy·작업대 동일). DROID 로봇 운동으로 post-train된 AC predictor는 액션이 0인 장면에서 dynamics를 전개하지 않는다 — Cosmos(관성 외삽)·VERA(무작위 반전)와는 다른 실패 위치("운동 전개 자체 없음").</p></div>
</section>
"""

    html = f"""<title>Phase A 작업대 씬</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F2F3F6;--panel:#FFFFFF;--ink:#21252C;--sub:#626B78;--line:#E1E4EA;--accent:#C4402C;--accent-ink:#A33422;--blue:#3E68A8;--ok:#256E46;--ok-bg:#E4F1E9;--warn:#8A5A14;--warn-bg:#F6EDDA;--chip-bg:#EEF0F4;--media:#23262C;--code-bg:#F0F1F4}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#141518;--panel:#1D1F24;--ink:#E8EAEE;--sub:#9AA2AE;--line:#2E323A;--accent:#E2604A;--accent-ink:#E9836F;--blue:#85A9DC;--ok:#6CC392;--ok-bg:#1C2E24;--warn:#D8A959;--warn-bg:#2E2718;--chip-bg:#26292F;--media:#101114;--code-bg:#23262C}}}}
:root[data-theme="dark"]{{--bg:#141518;--panel:#1D1F24;--ink:#E8EAEE;--sub:#9AA2AE;--line:#2E323A;--accent:#E2604A;--accent-ink:#E9836F;--blue:#85A9DC;--ok:#6CC392;--ok-bg:#1C2E24;--warn:#D8A959;--warn-bg:#2E2718;--chip-bg:#26292F;--media:#101114;--code-bg:#23262C}}
*{{box-sizing:border-box}} body{{background:var(--bg);color:var(--ink);margin:0;font-family:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:15px;line-height:1.68;word-break:keep-all}}
.mono{{font-family:"IBM Plex Mono",Consolas,monospace}} main{{max-width:980px;margin:0 auto;padding:40px 22px 90px}} a{{color:var(--blue)}}
h1{{font-size:27px;line-height:1.3;margin:8px 0 10px;font-weight:700;text-wrap:balance}} h2{{font-size:19px;margin:0 0 4px;font-weight:700;text-wrap:balance}} h3{{font-size:15.5px;margin:22px 0 8px;font-weight:700}} p{{margin:9px 0}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--sub)}}
.lead{{font-size:16.5px;color:var(--ink)}} .muted{{color:var(--sub)}} .small{{font-size:13px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 6px}} .chip{{background:var(--chip-bg);border-radius:999px;padding:4px 12px;font-size:12.5px}} .chip.ok{{background:var(--ok-bg);color:var(--ok)}} .chip.warn{{background:var(--warn-bg);color:var(--warn)}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin:18px 0}}
.sec-head{{display:flex;align-items:baseline;gap:12px;margin-bottom:8px}} .sec-head .n{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--sub)}}
.media{{background:var(--media);border-radius:8px;padding:8px;margin:12px 0 4px}} .media img{{display:block;width:100%;height:auto;border-radius:4px}} .media.scroll{{overflow-x:auto}} .media.scroll img{{width:auto;max-width:none;height:260px}}
.cap{{font-size:12.5px;color:var(--sub);margin:4px 0 12px}}
.tbl-wrap{{overflow-x:auto;margin:10px 0}} table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}} th{{color:var(--sub);font-weight:500;font-size:12.5px}} td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
.kv{{display:grid;grid-template-columns:120px 1fr;gap:6px 14px;margin:10px 0}} .kv dt{{color:var(--sub);font-size:13px}} .kv dd{{margin:0}}
.ask{{border-left:3px solid var(--accent);padding:6px 14px;background:var(--code-bg);border-radius:0 6px 6px 0}}
footer{{color:var(--sub);font-size:12.5px;margin-top:26px;line-height:1.7}}
</style>
<main>
<div class="eyebrow">PhysicsGen · Phase A · 2026-09-02 – 09-03</div>
<h1>Phase A 작업대 씬 — 외관 도메인 대조와 P0 제어</h1>
<p class="lead">Bundle A(toy 외관)의 P1·P2 임계 씬을 <b>물리는 그대로 두고 외관만</b> 로봇 작업대(회색 검사대, 정지한 Franka, 픽스처, 테이프 눈금)로 바꾸고,
결정 변수가 없는 <b>P0 등속 제어 씬</b>을 더했다. 질문은 둘이다: (1) toy 외관이 Cosmos에 불리했는가 — 사실적 외관에서 결정 곡선이 달라지는가,
(2) 생성 실패(객체 소실·정체)와 물리 실패를 P0로 분리할 수 있는가.</p>
<div class="chips">
  <span class="chip ok">Phase A 평가 — {n_seeds} seed · {n_roll} 롤아웃</span>
  <span class="chip">물리 서명 toy↔작업대 동일 (3/3)</span>
  <span class="chip warn">A1 정확도 {pct(toy["hill_roll"]["accuracy_nonboundary"])} → {pct(wb["hill_roll_wb"]["accuracy_nonboundary"])}</span>
  <span class="chip warn">A2 정확도 {pct(toy["two_ball"]["accuracy_nonboundary"])} → {pct(wb["two_ball_wb"]["accuracy_nonboundary"])}</span>
  {"<span class='chip'>P0 계속 " + pct(ctrl["p_continue"]) + "</span>" if ctrl else ""}
  {"<span class='chip warn'>V-JEPA 2-AC latent: 운동 전개 없음</span>" if vj else ""}
</div>

<section>
  <div class="sec-head"><span class="n">01</span><h2>무엇을 바꿨고 무엇을 지켰나</h2></div>
  <div class="kv">
    <dt>외관</dt><dd>기존 바닥 평면을 회색 라미네이트 검사대 재질로, 뒷벽, 팀 레포의 정적 Franka Panda(관절 없음·충돌 없음), 트레이·픽스처·빈, 테이프 스트립. 눈 언덕 → <b>초록 케이블커버 hump</b>(판독기 마스크 배타성 유지: 빨강·파랑·초록 외 색 없음).</dd>
    <dt>물리</dt><dd>변경 없음. 옵션(timestep·적분기·솔버)·접촉쌍·충돌 geom(형상·크기·위치·질량)·hfield를 컴파일된 모델에서 비교해 세 쌍 모두 동일함을 확인(§02).</dd>
    <dt>P0 제어</dt><dd>같은 접촉 모델·카메라·외관에서 평탄한 검사대 위 등속 구름(v0 5점). 사건 없음. 판독: 생성 클립 자신의 조건 프레임으로 등속 외삽한 궤적과의 오차(kin_err), GT 대비 오차(gt_err), 변위비(speed_ratio), validity.</dd>
    <dt>규모</dt><dd>29행(P0 5 + A1 11 + A2 11 + 사전검사 2) × {n_seeds} seed = {n_roll} 롤아웃 · Cosmos-Predict2-2B V2W, 5프레임 조건, 21프레임 롤아웃, Bundle A와 동일 설정.</dd>
  </div>
</section>

<section>
  <div class="sec-head"><span class="n">02</span><h2>씬과 검증</h2></div>
  {"".join(f'<div class="media scroll"><img src="data:image/jpeg;base64,{prev[f]}" alt="{f} 프리뷰"></div><p class="cap">{f} — 열 = 격자점, 행 = t=0 / 조건 끝 / +0.31 s / 사건 / t=1.25 s.</p>' for f in ("kin_roll","hill_roll_wb","two_ball_wb") if f in prev)}
  <div class="tbl-wrap"><table>
    <tr><th>씬</th><th>toy ↔ 작업대 물리 서명</th><th class="num">접촉쌍</th><th>충돌 geom</th></tr>
{vrows}  </table></div>
  <p class="small muted">P0 등속 감사(MuJoCo, 렌더 없음): {kin_line}. 격자 게이트(outcome·누설·map): A1·A2 전부 통과, 사건 프레임은 Bundle A와 동일.</p>
</section>

<section>
  <div class="sec-head"><span class="n">03</span><h2>P0 등속 제어 — 생성이 운동을 이어 그리는가</h2></div>
  {"<div class='tbl-wrap'><table><tr><th class='num'>v0 (m/s)</th><th class='num'>n</th><th class='num'>계속 굴러감</th><th class='num'>변위비 gen/GT</th><th class='num'>등속 외삽 오차 px</th><th class='num'>GT 오차 px</th><th class='num'>validity strict / frac</th></tr>" + p0_rows + "</table></div>" if ctrl else "<p class='small muted'>제어 결과 없음.</p>"}
  <p class="small muted">변위비 = (생성 프레임 20 − 5 변위)/(GT 동일 구간 변위). 판정 1 = 변위비 ≥ 0.5, 0 = &lt; 0.2(정지·반전), 그 사이 미결. 공 지름 ≈ 50 px.</p>
</section>

<section>
  <div class="sec-head"><span class="n">04</span><h2>P1·P2 결정 곡선 — toy 대 작업대</h2></div>
  <div class="media"><img src="data:image/png;base64,{img_curves}" alt="toy(Bundle A) 대 작업대(Phase A) 결정 곡선"></div>
  <p class="cap">회색 ■ Bundle A toy 외관(24 seed), 빨강 ● Phase A 작업대 외관({n_seeds} seed). 수직선 Wilson 95% CI. 파랑 = GT 계단, 점선 = 등속 인코더 기준선.</p>
  <div class="tbl-wrap"><table>
    <tr><th>씬</th><th class="num">비경계 정확도 toy / 작업대</th><th class="num">인코더</th><th class="num">미결 toy / 작업대</th><th class="num">strict valid toy / 작업대</th><th class="num">법칙 지표 toy → 작업대</th></tr>
{fam_rows}  </table></div>
  <p class="small muted">사전검사(작업대): hill S=0.35에서 통과 예측 {hill_pre_pass}/{len(hill_pre)} seed, 벽 반동 정판정 {wall_ok}/{len(wall_pre)} seed (toy: {toy_wall_ok}/{toy_wall_n}).</p>
</section>

<section>
  <div class="sec-head"><span class="n">05</span><h2>VERA DROID planner — 두 번째 모델</h2></div>
  <p>Wan2.1-I2V-14B 기반 DROID 비디오 planner(공개 체크포인트)를 추가 학습 없이 붙였다. {rc}</p>
  <p class="small muted">입력 규약: 3카메라 192×128 타일(576×128)·15 fps·문맥 29프레임. 우리 씬은 사건 전 history가 0.31 s뿐이라 <b>100 Hz 캡처로 시간을 늘려</b>(15 fps 재생 시 6.7배 슬로모션) 29프레임 = 0.29 s 사건 전 문맥을 만들고, 24프레임 청크 3개를 autoregressive로 이어 0.72 s를 덮었다. 3뷰 = 씬 측면 카메라 + 3/4 상방 + 근상방. 판독은 측면 타일을 16 fps 등가로 재표본한 뒤 동결 판독기 적용.</p>
  {vera_block}
{strips_html}</section>

{vj_block}<section>
  <div class="sec-head"><span class="n">07</span><h2>파이프라인 교훈과 다음 단계</h2></div>
  <ul>
    <li><b>렌더 백엔드.</b> GPU가 diffusion 작업에 점유된 동안 MuJoCo EGL 렌더는 간헐적으로 검거나 부분 손상된 프레임을 냈고, 재시도 중 프로세스가 조용히 종료되기도 했다. 소프트웨어 렌더(OSMesa)로 전환하되 OSMesa 프레임은 EGL과 상하 반전이므로 기준 프레임 비교로 검증한 뒤 뒤집어 쓴다. 생성 시 두 번 렌더해 완전 일치해야 통과.</li>
    <li><b>세션 소멸 대비.</b> 수집→판독은 세션 워처가 아니라 스크립트 안에서 연쇄한다.</li>
    <li><b>latent 트랙 방법론(재사용 가능).</b> 등속 counterfactual 렌더러(물리 궤적 qpos 재배치로 자가검증, 재배치 오차 &lt; 0.05 px), 1 px 지터 floor와 분리도 게이트, freeze 교란을 제거한 변위 방향 지표와 오라클 스케일, 문맥 반전 제어. 다음 latent 후보(DiLA 등)에 그대로 적용한다.</li>
    <li><b>다음.</b> 결정 대기: VERA GO/NO-GO(반동은 그리나 S 무관, 미결 22–33%, 220 s/샘플), DiLA 착수(past-only 예측 미제공 → 미래 latent action 정책 필요). Phase B(매달린 payload 진자·stopper·support-edge 이탈), Phase C(자연 occlusion 짝) 이후 A2World(robot-context 씬).</li>
  </ul>
</section>
<footer>생성: <span class="mono">SimDROID/code_vwm</span> 브랜치 <span class="mono">mnjihun/bundle-a-scenes</span> · 씬 <span class="mono">core/threshold/{{workbench,kin_roll}}.py</span> · 생성 <span class="mono">gen/make_phase_a.py</span> · 검증 <span class="mono">exp/verify_phase_a.py</span> · 판독 <span class="mono">exp/analyze_bundle_a.py</span> · VERA <span class="mono">external/vera_infer.py</span> · V-JEPA 2-AC <span class="mono">gen/make_vjepa_inputs.py · exp/rollout_vjepa_ac.py · exp/analyze_vjepa_ac.py</span> · 2026-09-02/03 · 문지훈/Claude</footer>
</main>
"""
    Path(dst).write_text(html)
    print(f"OK -> {dst} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
