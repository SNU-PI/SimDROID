"""Splice the prompt-contrast (oracle) results into the Bundle A artifact as section 09.

Reads analysis/oracle_samples.csv (prompt arms) and analysis/samples.csv (neutral arm)
and compares, at the boundary flanks S=0.95/1.05, P(pass) under three prompts:
neutral, "return"-worded, "pass"-worded.  Idempotent: an existing section 09 is
replaced.  Usage: build_oracle_section.py <src.html> <dst.html>
"""
import base64, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/nvme/migration/jihun/SimDROID/code_vwm/artifacts/bundle_a")
ASSETS = ROOT / "analysis" / "artifact_assets"
FAMS = (("hill_roll", "A1 rolling hill"), ("two_ball", "A2 two-ball"), ("pendulum_rod", "A3 rod pendulum"))
FLANKS = (0.95, 1.05)
INK, SUB, LINE = "#21252C", "#626B78", "#E1E4EA"
ACCENT, BLUE, OK = "#C4402C", "#3E68A8", "#256E46"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def arm_stats(rows):
    dec = [r for r in rows if r["prediction"] in (0, 1)]
    k = sum(r["prediction"] for r in dec)
    p = k / len(dec) if dec else np.nan
    lo, hi = wilson(k, len(dec))
    vf = float(np.mean([r["valid_frac"] for r in rows])) if rows else np.nan
    return {"n": len(rows), "n_dec": len(dec), "k": k, "p": p, "lo": lo, "hi": hi, "valid_frac": vf}


def pct(x, nd=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f}%"


def load():
    orc = list(csv.DictReader(open(ROOT / "analysis/oracle_samples.csv")))
    for r in orc:
        r["S"] = float(r["S"]); r["outcome"] = int(r["outcome"]); r["prediction"] = int(r["prediction"])
        r["seed"] = int(r["seed"]); r["valid_frac"] = float(r["valid_frac"])
        r["prompt_dir"] = r["outcome"] if r["oracle"] == "correct" else 1 - r["outcome"]
    neu = []
    for r in csv.DictReader(open(ROOT / "analysis/samples.csv")):
        if r["role"] != "grid" or float(r["S"]) not in FLANKS:
            continue
        neu.append({"family": r["family"], "S": float(r["S"]), "outcome": int(r["outcome"]),
                    "prediction": int(r["prediction"]), "seed": int(r["seed"]), "valid_frac": float(r["valid_frac"])})
    return orc, neu


def main(src, dst):
    orc, neu = load()
    seeds = sorted({r["seed"] for r in orc})
    n_seeds = len(seeds)
    table = {}   # (fam, S) -> {arm: stats}
    effects = {}
    for fam, _ in FAMS:
        for S in FLANKS:
            table[(fam, S)] = {
                "neutral": arm_stats([r for r in neu if r["family"] == fam and r["S"] == S]),
                "return": arm_stats([r for r in orc if r["family"] == fam and r["S"] == S and r["prompt_dir"] == 0]),
                "pass": arm_stats([r for r in orc if r["family"] == fam and r["S"] == S and r["prompt_dir"] == 1]),
            }
        fr = [r for r in orc if r["family"] == fam]
        # pooled text effect: P(pass | pass-prompt) - P(pass | return-prompt), decided rollouts, both flanks
        ps = arm_stats([r for r in fr if r["prompt_dir"] == 1]); rs = arm_stats([r for r in fr if r["prompt_dir"] == 0])
        hi = arm_stats([r for r in fr if r["S"] == 1.05]); lo = arm_stats([r for r in fr if r["S"] == 0.95])
        # paired flips: same seed & S, correct vs wrong prediction
        pairs = defaultdict(dict)
        for r in fr:
            pairs[(r["seed"], r["S"])][r["oracle"]] = r["prediction"]
        both = [v for v in pairs.values() if len(v) == 2 and v["correct"] in (0, 1) and v["wrong"] in (0, 1)]
        flips = sum(v["correct"] != v["wrong"] for v in both)
        # accuracy at flanks (undecided counted wrong), per arm
        acc = {}
        for name, rows in (("neutral", [r for r in neu if r["family"] == fam]),
                           ("correct", [r for r in fr if r["oracle"] == "correct"]),
                           ("wrong", [r for r in fr if r["oracle"] == "wrong"])):
            acc[name] = float(np.mean([r["prediction"] == r["outcome"] for r in rows])) if rows else np.nan
        effects[fam] = {"d_text": ps["p"] - rs["p"], "d_S": hi["p"] - lo["p"], "p_passprompt": ps["p"], "p_returnprompt": rs["p"],
                        "pairs": len(both), "flips": flips, "acc": acc,
                        "undecided_rate": float(np.mean([r["prediction"] == -1 for r in fr])) if fr else np.nan}

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), dpi=170, sharey=True)
    arms = (("neutral", "neutral prompt", SUB, "o", -0.03), ("return", "prompt: returns", BLUE, "v", 0.0),
            ("pass", "prompt: passes", ACCENT, "^", 0.03))
    for ax, (fam, label) in zip(axes, FAMS):
        ax.plot([0.44, 1, 1, 1.56], [0, 0, 1, 1], color=BLUE, lw=1.2, alpha=.45, zorder=1)
        ax.axvline(1.0, color=LINE, lw=1)
        for arm, aname, col, mk, off in arms:
            xs, ys = [], []
            for S in FLANKS:
                st = table[(fam, S)][arm]
                x = S + off
                if st["n_dec"]:
                    ax.plot([x, x], [st["lo"], st["hi"]], color=col, lw=1.2, alpha=.6, zorder=2)
                    xs.append(x); ys.append(st["p"])
                    ax.annotate(f"n={st['n_dec']}", (x, -0.16), ha="center", fontsize=6.5, color=SUB)
            ax.plot(xs, ys, marker=mk, ms=6.5, color=col, lw=1.1, ls="--", label=aname, zorder=3)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0.84, 1.16); ax.set_ylim(-0.22, 1.06)
        ax.set_xticks([0.95, 1.05]); ax.set_xticklabels(["S=0.95\n(GT: returns)", "S=1.05\n(GT: passes)"])
        ax.set_yticks([0, 0.5, 1.0])
    axes[0].set_ylabel("P(predict pass | decided)")
    axes[2].legend(loc="center right", fontsize=8.2, frameon=False)
    fig.tight_layout(); fig.savefig(ASSETS / "oracle3.png", bbox_inches="tight"); plt.close(fig)
    img = base64.b64encode((ASSETS / "oracle3.png").read_bytes()).decode()

    # ---- readings
    def read_text(d):
        if np.isnan(d): return "판정 불가"
        a = abs(d)
        return "텍스트가 방향을 움직인다" if a >= 0.30 else ("텍스트 효과 약함" if a >= 0.15 else "텍스트에 둔감")
    def read_S(d):
        if np.isnan(d): return "판정 불가"
        a = abs(d)
        return "S에 반응" if a >= 0.30 else ("S 효과 약함" if a >= 0.15 else "S에 둔감")

    rows_html = ""
    for fam, label in FAMS:
        for S in FLANKS:
            t = table[(fam, S)]
            gt = "반환" if S < 1 else "통과"
            rows_html += (f'<tr><td>{label}</td><td class="num">{S:.2f} ({gt})</td>'
                          + "".join(f'<td class="num">{pct(t[a]["p"])} <span class="small muted">({t[a]["k"]}/{t[a]["n_dec"]} · v {t[a]["valid_frac"]:.2f})</span></td>'
                                    for a in ("neutral", "return", "pass"))
                          + "</tr>\n")
    eff_html = ""
    for fam, label in FAMS:
        e = effects[fam]
        eff_html += (f'<tr><td>{label}</td><td class="num">{e["d_text"]:+.2f}</td><td>{read_text(e["d_text"])}</td>'
                     f'<td class="num">{e["d_S"]:+.2f}</td><td>{read_S(e["d_S"])}</td>'
                     f'<td class="num">{e["flips"]}/{e["pairs"]}</td>'
                     f'<td class="num">{pct(e["acc"]["neutral"])} → {pct(e["acc"]["correct"])} / {pct(e["acc"]["wrong"])}</td></tr>\n')

    e1, e2, e3 = effects["hill_roll"], effects["two_ball"], effects["pendulum_rod"]
    two_pass = e2["p_passprompt"]
    a2_line = (f'A2에서 "빨간 공이 되튀어 온다"고 써 줘도 반전 비율은 {pct(two_pass)} — '
               + ("반동 자체가 생성되지 않는다: 질량비 판독 이전의 <b>생성 한계</b>로 확정된다."
                  if not np.isnan(two_pass) and two_pass < 0.35 else
                  "텍스트로는 반동이 나온다: 중립 조건의 실패는 생성 한계가 아니라 <b>픽셀 판독 실패</b> 쪽이다."))
    a1_line = (f'A1에서 "되돌아온다"고 써 주면 통과 예측이 {pct(table[("hill_roll",0.95)]["neutral"]["p"])}(중립)에서 '
               f'{pct(table[("hill_roll",0.95)]["return"]["p"])}(반환 프롬프트, S=0.95)로 '
               + ("움직인다 — 되돌아오는 운동을 그릴 수는 있으나 장면에서 그 이유를 읽지 못한다."
                  if not np.isnan(e1["d_text"]) and abs(e1["d_text"]) >= 0.30 else
                  "거의 움직이지 않는다 — 관성 외삽이 텍스트 지시보다 강하다."))
    a3_line = (f'A3는 텍스트 효과 {e3["d_text"]:+.2f}, S 효과 {e3["d_S"]:+.2f} — '
               + ("텍스트와 화면 신호가 함께 작동한다." if (not np.isnan(e3["d_text"]) and abs(e3["d_text"]) >= 0.15 and abs(e3["d_S"]) >= 0.15)
                  else "화면 신호(각감속)가 주도하고 텍스트는 보조적이다." if (not np.isnan(e3["d_S"]) and abs(e3["d_S"]) >= 0.15)
                  else "경계 양측에서 텍스트도 S도 결과를 거의 바꾸지 못한다."))

    section = f"""<section>
  <div class="sec-head"><span class="n">09</span><h2>프롬프트 대조 — 텍스트가 판독을 대신할 수 있는가</h2></div>
  <p class="lead">경계 양측 S=0.95(GT: 반환)·S=1.05(GT: 통과)에서 <b>같은 조건 영상·같은 seed</b>에 정답 서술과 오답 서술을 각각 붙여
  {n_seeds} seed × 12행 = {n_seeds*12} 롤아웃을 추가 수집했다. 중립 프롬프트 결과(§07)와 나란히 놓으면
  "모델이 그 운동을 <i>그릴 수 있는가</i>"와 "장면에서 그 이유를 <i>읽는가</i>"가 분리된다.</p>
  <div class="media"><img src="data:image/png;base64,{img}" alt="프롬프트 대조: 가족별 S=0.95/1.05에서 중립·반환·통과 프롬프트의 통과 예측 확률"></div>
  <p class="cap">P(통과 예측 | 판정 성립) — 회색 ○ 중립, 파랑 ▽ "되돌아온다/멈춘다" 서술, 빨강 △ "넘어간다/되튄다/돈다" 서술. 수직선 Wilson 95% CI, 아래 n = 판정 성립 수. 연한 파랑 계단 = GT.</p>
  <div class="tbl-wrap"><table>
    <tr><th>씬</th><th class="num">S (GT)</th><th class="num">중립</th><th class="num">"반환" 서술</th><th class="num">"통과" 서술</th></tr>
    <tr><td colspan="5" class="small muted">칸 표기: P(통과) (통과 수/판정 성립 수 · v = valid_frac 평균, 객체 마스크가 정상 크기로 보인 프레임 비율)</td></tr>
{rows_html}  </table></div>
  <div class="tbl-wrap"><table>
    <tr><th>씬</th><th class="num">텍스트 효과 Δ<sub>text</sub></th><th>읽기</th><th class="num">물리 효과 Δ<sub>S</sub></th><th>읽기</th><th class="num">짝 뒤집힘</th><th class="num">경계 정확도 중립 → 정답 / 오답</th></tr>
{eff_html}  </table></div>
  <p class="small muted">Δ<sub>text</sub> = P(통과 | "통과" 서술) − P(통과 | "반환" 서술), 양측 S 합산·판정 성립 롤아웃 기준. Δ<sub>S</sub> = P(통과 | S=1.05) − P(통과 | S=0.95), 두 서술 합산.
  짝 뒤집힘 = 같은 seed·같은 S에서 정답/오답 서술의 판정이 서로 다른 쌍 수 / 양쪽 모두 판정 성립한 쌍 수. 경계 정확도는 미결을 오답으로 센다. 미결 비율(대조군 전체): A1 {pct(e1["undecided_rate"])} · A2 {pct(e2["undecided_rate"])} · A3 {pct(e3["undecided_rate"])}.</p>
  <h3>판정</h3>
  <ul>
    <li>{a2_line}</li>
    <li>{a1_line}</li>
    <li>{a3_line}</li>
  </ul>
  <p class="small muted">프롬프트 원문 — A1: "The red ball lacks energy: it rolls partway up the white hill, slows, and returns to the left." / "…has enough energy: it rolls over the white hill's crest and continues to the right."
  A2: "The blue ball is smaller and lighter: after impact the red ball keeps moving right…" / "…larger and heavier: after impact the red ball bounces back to the left while the blue ball moves right."
  A3: "The rod is too long: the pendulum rises, stops before the top, and swings back." / "The rod is short enough: the pendulum swings over the top and completes the rotation."</p>
</section>
"""
    html = Path(src).read_text()
    html, _ = re.subn(r'<section>\s*<div class="sec-head"><span class="n">09</span>.*?</section>\s*', "", html, flags=re.S)
    html, n = re.subn(r"(</section>\s*)(<footer>)", lambda m: m.group(1) + section + "\n" + m.group(2), html, count=1, flags=re.S)
    assert n == 1, "footer anchor"
    html, n = re.subn(r"단일 체크포인트\(2B, 480p/16fps\)·중립 프롬프트 결과다\. 스펙의 프롬프트 대조\(정답/오답 서술\)와\s*denoising-loss 판단 채점\(VoE\)은 후속 — 텍스트가 물리 신호를 보완하는지가 다음 질문\.",
                      "단일 체크포인트(2B, 480p/16fps) 결과다. 프롬프트 대조(정답/오답 서술)는 §09에 반영했고, denoising-loss 판단 채점(VoE)은 후속.", html)
    Path(dst).write_text(html)
    json.dump({"seeds": seeds, "effects": effects,
               "table": {f"{k[0]}_S{k[1]}": v for k, v in table.items()}},
              open(ASSETS / "oracle_stats.json", "w"), indent=2, default=float)
    print(f"OK -> {dst} ({len(html)/1e6:.2f} MB); limits bullet replaced: {n}")
    for fam, _ in FAMS:
        e = effects[fam]
        print(f"{fam}: d_text={e['d_text']:+.2f} d_S={e['d_S']:+.2f} p_pass(pass-prompt)={e['p_passprompt']:.2f} p_pass(return-prompt)={e['p_returnprompt']:.2f} flips={e['flips']}/{e['pairs']} acc n/c/w={e['acc']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
