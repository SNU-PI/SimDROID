"""Build the PVR PoC report page (self-contained HTML, Korean) from the artifacts:
scene construction and purpose, validation method and purpose, results, implications.

Inputs (per scene under artifacts/pvr/<scene>/): manifest.jsonl, summary.json, previews/*.png,
analysis/{summary.json,curves.png,sheets/<scene>.jpg}, attrib/{summary.json,region_fractions.png,
heatmaps_core.png}, plus artifacts/pvr/verify.json and an optional narrative file
artifacts/pvr/report_text.json (section -> HTML fragments written after reading the numbers).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

SCENE_NAMES = {"hill": "Predictive Hill", "collide": "Predictive Collide", "edge": "Predictive Edge / Fall"}
FAMILY = {"hill": "hill_roll_pvr", "collide": "two_ball_pvr", "edge": "support_edge_pvr"}
EVENT = {"hill": "통과(1) / 복귀(0)", "collide": "반전(1) / 계속(0)", "edge": "낙하(1) / 유지(0)"}
SCORE = {"hill": "넘김 정도 (x_max − x_crest)/d", "collide": "v_post / v_pre (GT (1−S)/(1+S))",
         "edge": "낙하 시각(첫 하강 프레임, 검열 = 21)"}

CSS = """
:root{--ground:#F3F3F0;--surface:#FFFFFF;--surface-2:#EAEBE7;--ink:#22262B;--ink-2:#4B5058;--muted:#6E737B;--line:#D6D8D3;
--green:#2E8B4A;--green-soft:#DCEFE1;--red:#C2432F;--red-soft:#F6E1DC;--blue:#3A6BB5;--sage:#8FA48F;--sage-soft:#E6ECE5;--tape:#D9CFA6;--tape-soft:#F1ECDA;--focus:#1F6FB2}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#1E2125;--surface:#272B30;--surface-2:#2F343A;--ink:#E8E9EA;--ink-2:#C3C7CC;--muted:#9CA2AA;--line:#3D434A;
--green:#4CB56B;--green-soft:#22402C;--red:#E0654F;--red-soft:#43292A;--blue:#6C9BE0;--sage:#7E9680;--sage-soft:#2B352C;--tape:#B9AE86;--tape-soft:#3A3729;--focus:#7DB6EA}}
:root[data-theme="dark"]{--ground:#1E2125;--surface:#272B30;--surface-2:#2F343A;--ink:#E8E9EA;--ink-2:#C3C7CC;--muted:#9CA2AA;--line:#3D434A;
--green:#4CB56B;--green-soft:#22402C;--red:#E0654F;--red-soft:#43292A;--blue:#6C9BE0;--sage:#7E9680;--sage-soft:#2B352C;--tape:#B9AE86;--tape-soft:#3A3729;--focus:#7DB6EA}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans KR","Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;font-size:15px;line-height:1.7}
a{color:var(--focus)}h1,h2,h3{font-family:"Noto Serif KR","Nanum Myeongjo",serif;font-weight:600;line-height:1.3;margin:0;text-wrap:balance}
h1{font-size:2.1rem}h2{font-size:1.35rem}h3{font-size:1.05rem}p{margin:0}.mono,code{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.92em}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.page{max-width:76rem;margin:0 auto;padding:2.5rem 1.5rem 5rem}.prose{max-width:48em}.prose p+p{margin-top:.8em}ul,ol{margin:0;padding-left:1.3em}li+li{margin-top:.3em}
header.top{display:grid;gap:.9rem;padding-bottom:1.6rem;border-bottom:1px solid var(--line)}.lede{font-size:1.05rem;color:var(--ink-2);max-width:48em}
.chips{display:flex;flex-wrap:wrap;gap:.45rem}.chip{font-family:"IBM Plex Mono",monospace;font-size:.76rem;padding:.18rem .6rem;border-radius:999px;border:1px solid var(--line);color:var(--ink-2);background:var(--surface)}
.chip.on{border-color:var(--green);color:var(--green);background:var(--green-soft)}.chip.warn{border-color:var(--red);color:var(--red);background:var(--red-soft)}
section{padding:2.2rem 0 0}section>h2{display:flex;align-items:baseline;gap:.7rem;margin-bottom:.9rem}section>h2 .num{font-family:"IBM Plex Mono",monospace;font-size:.8rem;color:var(--muted);letter-spacing:.06em}
.grid-2{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:1.6rem;align-items:start}.grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}
@media (max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}
.tablewrap{overflow-x:auto;border:1px solid var(--line);background:var(--surface)}table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;vertical-align:top;padding:.42rem .6rem;border-bottom:1px solid var(--line)}th{font-weight:600;font-size:.78rem;color:var(--ink-2);background:var(--surface-2);white-space:nowrap}
tr:last-child td{border-bottom:none}td.id{font-family:"IBM Plex Mono",monospace;white-space:nowrap}td.num{text-align:right;font-family:"IBM Plex Mono",monospace;white-space:nowrap}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.72rem;padding:.04rem .4rem;border-radius:3px}.tag.rel{background:var(--green-soft);color:var(--green)}.tag.dec{background:var(--sage-soft);color:var(--sage)}.tag.ext{background:var(--tape-soft);color:var(--ink-2)}
figure{margin:0}figure img{width:100%;height:auto;display:block;border:1px solid var(--line);background:var(--surface)}figcaption{font-size:.84rem;color:var(--muted);margin-top:.45rem}
.callout{border:1px solid var(--line);background:var(--surface);padding:.9rem 1.1rem;font-size:.92rem}.callout.key{border-color:var(--green)}.callout.warn{border-color:var(--tape);background:var(--tape-soft)}
.card{border:1px solid var(--line);background:var(--surface);padding:1rem 1.1rem;display:grid;gap:.5rem}.card h3{font-family:"IBM Plex Sans KR",sans-serif;font-weight:600}
.small{font-size:.85rem;color:var(--muted)}footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.84rem;color:var(--muted)}
"""


def img_b64(path, max_w=1500, quality=82):
    if not path or not Path(path).exists():
        return None
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(path, caption, max_w=1500):
    src = img_b64(path, max_w)
    if src is None:
        return f'<div class="callout warn">그림 없음: {Path(path).name}</div>'
    return f'<figure><img src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def f2(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "–"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def scene_block(scene, root, text):
    ana = load_json(root / "analysis" / "summary.json")
    att = load_json(root / "attrib" / "summary.json")
    html = [f'<section id="{scene}"><h2><span class="num">{SCENE_NAMES[scene]}</span>결과</h2>']
    if ana is None:
        html.append('<div class="callout warn">판독 결과 없음(수집 미완료)</div></section>')
        return "\n".join(html)
    ov = ana["overall"]
    html.append(f'<div class="chips"><span class="chip">롤아웃 {ana["rollouts"]}</span><span class="chip">seed {len(ana["seeds"])}</span>'
                f'<span class="chip">GT 자가검증 {ana["gt_selftest"]}</span><span class="chip">비경계 정확도 {f2(ov["accuracy_nonboundary"])}</span>'
                f'<span class="chip">유효율 {f2(ov["valid_rate"])}</span><span class="chip">미결 {f2(ov["undecided_rate"])}</span></div>')
    if text.get(f"{scene}_result"):
        html.append(f'<div class="callout key" style="margin-top:.9rem">{text[f"{scene}_result"]}</div>')
    rows = []
    for c in ana["conditions"]:
        rows.append(f'<tr><td class="id">{c["cond"]}</td><td>{c["group"]}</td><td class="num">{f2(c["x"], 3)}</td>'
                    f'<td class="num">{c["outcome"]}</td><td class="num">{f2(c["p_event"])}</td><td class="num">{f2(c["accuracy"])}</td>'
                    f'<td class="num">{f2(c["s_e_mean"])} ± {f2(c["s_e_sd"])}</td><td class="num">{f2(c["s_e_gt"])}</td>'
                    f'<td class="num">{f2(c["event_gen_mean"], 1)} ({c["event_gen_n"]})</td><td class="num">{c["event_gt"] if c["event_gt"] is not None else "–"}</td>'
                    f'<td class="num">{f2(c["undecided"])}</td><td class="num">{f2(c["valid_rate"])}</td>'
                    + (f'<td class="num">{f2(c.get("hover_rate"))}</td>' if scene == "edge" else "") + "</tr>")
    xlab = {"hill": "h [m]", "collide": "S", "edge": "t_edge [s]"}[scene]
    html.append(f'<h3 style="margin-top:1.2rem">조건별 판독 (사건 {EVENT[scene]}, s_e = {SCORE[scene]})</h3><div class="tablewrap"><table><thead><tr>'
                f'<th>id</th><th>군</th><th>{xlab}</th><th>GT</th><th>P(사건)</th><th>정확도</th><th>s_e 모델</th><th>s_e GT</th><th>사건 프레임 모델(n)</th><th>GT</th><th>미결</th><th>유효</th>'
                + ("<th>부양</th>" if scene == "edge" else "") + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    erows = []
    for e in ana["effects"]:
        tag = {"relevant": "rel", "decoy": "dec", "decoy_presence": "ext"}[e["kind"]]
        ci_excl = (e["ci_lo"] > 0 or e["ci_hi"] < 0) if not np.isnan(e["ci_lo"]) else False
        erows.append(f'<tr><td><span class="tag {tag}">{e["kind"]}</span> {e["label"]}</td><td class="num">{e["n_pairs"]}</td>'
                     f'<td class="num">{f2(e["delta_mean"])} [{f2(e["ci_lo"])}, {f2(e["ci_hi"])}]{" *" if ci_excl else ""}</td>'
                     f'<td class="num">{f2(e["delta_gt"])}</td><td class="num">{f2(e["normalised"])}</td>'
                     f'<td>{"" if e["direction_agrees"] is None else ("예" if e["direction_agrees"] else "아니오")}</td>'
                     f'<td class="num">{f2(e["flip_rate"])}</td><td class="num">{f2(e["p_event_from"])} → {f2(e["p_event_to"])}</td><td class="num">{f2(e["cohen_d"])}</td></tr>')
    ri = ana["relevance_index"]
    html.append('<h3 style="margin-top:1.2rem">짝 편집 효과 (같은 seed, 부트스트랩 95% CI; * = CI가 0을 제외)</h3><div class="tablewrap"><table><thead><tr>'
                '<th>편집</th><th>n</th><th>Δs_e 모델 [CI]</th><th>Δs_e GT</th><th>정규화 Δ/Δ_GT</th><th>방향 일치</th><th>뒤집힘률</th><th>P(사건) 전→후</th><th>d</th></tr></thead><tbody>'
                + "".join(erows) + "</tbody></table></div>")
    timing = ""
    if scene == "edge" and ana.get("timing"):
        t = ana["timing"]
        timing = (f'<div class="callout"><b>낙하 시각 정확도</b>(낙하 셀, 같은 검출기): n={t["n"]}, 평균 오차 {f2(t["mean_err_frames"], 1)} 프레임, MAE {f2(t["mae_frames"], 1)} 프레임, '
                  f'검열(낙하 미검출) {t["censored_fall_cells"]}건, 유지 셀 거짓 낙하율 {f2(t["false_fall_rate_hold"])}, 낙하 셀 부양률 {f2(t["hover_rate_fall_cells"])}</div>')
    html.append('<div class="grid-2" style="margin-top:.8rem"><div class="callout"><b>Relevance Index</b> = |Δ관련| − |Δ미끼| (seed별 짝, 부트스트랩 CI)<br>'
                + "<br>".join(f'{r["relevant"]} vs {r["decoy"]}: <span class="mono">{f2(r["ri_mean"])} [{f2(r["ci_lo"])}, {f2(r["ci_hi"])}]</span> (n={r["n"]})' for r in ri)
                + '</div>' + (timing or "<div></div>") + "</div>")
    html.append('<div class="grid-2" style="margin-top:1rem">' + fig(root / "analysis" / "curves.png", f"{SCENE_NAMES[scene]}: 결정 변수에 대한 P(사건)와 s_e(모델 vs GT).")
                + fig(root / "analysis" / "sheets" / f"{scene}.jpg", "결과 시트: 열 = 조건, 행 = GT와 seed별 롤아웃(GT 사건 프레임에서). 기호 1/0/? = 판독.", max_w=1500) + "</div>")
    if att:
        trows = []
        for t in att["table"]:
            g, gd, ad = t["grad"], t["grad_density"], t["attention_ball_late_density"]
            trows.append(f'<tr><td class="id">{t["cond"]}</td><td class="num">{t["n_seeds"]}</td>'
                         + "".join(f'<td class="num">{f2(g[r][0])}</td>' for r in ("ball", "structure", "decoy", "support", "background"))
                         + "".join(f'<td class="num">{f2(gd[r], 1)}</td>' for r in ("ball", "structure", "decoy", "support", "background"))
                         + "".join(f'<td class="num">{f2(ad[r], 1)}</td>' for r in ("ball", "structure", "decoy", "support", "background"))
                         + f'<td class="num">{f2(t["latent_frame_share"][0])}</td></tr>')
        html.append(f'<h3 style="margin-top:1.4rem">③ 위치 층 — 조건 프레임 5장 × 영역 (그래디언트 질량 분율 / 면적 대비 밀도 / 어텐션 밀도, seed {att["table"][0]["n_seeds"]}·step 3 평균)</h3>'
                    '<div class="tablewrap"><table><thead><tr><th>id</th><th>n</th>'
                    '<th colspan="5">|∂s/∂x| 질량 분율: 공·구조·미끼·지지·배경</th><th colspan="5">밀도(질량/면적, 1 = 균일): 공·구조·미끼·지지·배경</th>'
                    '<th colspan="5">어텐션 밀도(예측 공 토큰 → 조건 토큰, 후기 블록): 공·구조·미끼·지지·배경</th><th>잠재 프레임0 몫</th></tr></thead><tbody>'
                    + "".join(trows) + "</tbody></table></div>")
        rnd = att.get("randomization") or []
        rtxt = "; ".join(", ".join(f"k={k}: r={f2(v['corr_with_original'])}" for k, v in r.items()) for r in rnd[:2]) or "–"
        html.append(f'<div class="callout" style="margin-top:.8rem"><b>정합 검사.</b> 무관 스칼라(잠재 평균) 지도와의 상관 평균 {f2(att.get("wrong_target_corr_mean"))} '
                    f'(1에 가까울수록 원시 그래디언트가 점수 무관 성분에 지배됨) · 마지막 k 블록 무작위화 후 지도 상관: {rtxt} · '
                    f'seed 간 구조 질량 변동계수: ' + ", ".join(f'{c} {f2(v)}' for c, v in list(att["seed_cv_structure"].items())[:6]) + "</div>")
        html.append('<div class="grid-2" style="margin-top:1rem">' + fig(root / "attrib" / "region_fractions.png", "조건별 영역 질량 분율: 그래디언트 / 어텐션(공 토큰 질의, 후기 블록) / Sobel 가장자리 기준선.")
                    + fig(root / "attrib" / "heatmaps_core.png", "핵심 조건의 |∂s/∂x| 열지도(마지막 조건 프레임, step·seed 평균).", max_w=1100) + "</div>")
        if text.get(f"{scene}_attrib"):
            html.append(f'<div class="callout" style="margin-top:.8rem">{text[f"{scene}_attrib"]}</div>')
    else:
        html.append('<div class="callout warn" style="margin-top:1rem">귀속(③ 위치 층) 결과 없음(미실행 또는 진행 중)</div>')
    html.append("</section>")
    return "\n".join(html)


PURPOSE = {
    "hill": ("진행 경로 위 경사면의 높이가 통과/복귀를 결정한다. 사건 전에 <b>경로 위 지형</b>을 쓰는지 본다.",
             "관련 편집 = 주 언덕 높이 h(S=2.0 ↔ 0.5), 미끼 = 공 뒤 같은 프로파일의 언덕 h_d, 확장 = 20·28 cm 언덕·수직 벽·평지."),
    "collide": ("아직 닿지 않은 표적의 크기(질량비)가 반전/계속을 결정한다. 사건 전에 <b>충돌 상대</b>를 쓰는지 본다.",
                "관련 편집 = 표적 반지름 r₂(S=2.0 ↔ 0.5), 미끼 = 공 뒤 정지한 파란 공 r_d, 확장 = 벽·초경량 표적·빗나감(y +0.12)·표적 없음."),
    "edge": ("지지 끝까지의 거리와 현재 속도가 <b>언제</b> 떨어지는지를 결정한다(t = d/v₀). 사건 전에 지지 경계와 속도를 함께 쓰는지 본다.",
             "관련 편집 = 오른쪽 끝 t_edge(0.6 ↔ 1.05 s)·속도 v₀(0.25/0.35/0.50, 등시간 격자), 미끼 = 플레이트 왼쪽 끝 x_left, 확장 = 이음선·단차·3° 기울기, 유지 대조 t=1.4."),
}

METHOD_DEFAULT = """
<div class="grid-2"><div class="prose">
<p><b>모델 계약.</b> Cosmos-Predict2-2B Video2World(480p·16 fps 체크포인트)에 조건 5프레임과 씬당 하나의 중립 프롬프트를 주고 35 step, guidance 7.0으로 21프레임(조건 5 + 예측 16)을 생성한다. 씬당 조건 15·15·18 × seed 12(모든 조건에 같은 seed = common random numbers).</p>
<p><b>① 정확도 층.</b> 동결 판독기(색 마스크 중심 궤적)로 사건을 읽고, 연속 점수 s_e(넘김 정도 / 사후·사전 속도비 / 낙하 시각)를 GT와 같은 판독기로 비교한다. 결정 변수 사다리에서 응답 함수 s_model vs s_GT, Edge에서는 낙하 시각 오차·검열율·부양률.</p>
<p><b>② 의존 층.</b> 짝 편집 효과 Δs = mean_seed[s_e(편집 후) − s_e(편집 전)]와 95% 부트스트랩 CI, GT 효과 Δs_GT, 정규화 Δ/Δ_GT, 방향 일치, 뒤집힘률. Relevance Index = |Δ관련| − |Δ미끼|(seed별 짝). 관련 편집은 실제 미래를 바꾸고, 미끼 편집은 바꾸지 않는다.</p>
<p><b>③ 위치 층.</b> 파이프라인의 샘플링을 같은 seed로 재현하고 step 20/24/28(σ ≈ 1.09/0.30/0.06)에서 x_i를 고정한 채 한 번의 x₀ 예측을 잠재 볼 probe 점수(16채널 로지스틱 → 뾰족한 softmax 중심; GT 잠재에서 오차 중앙값 ≈ 1.2 셀)로 미분해 조건 5프레임(fp32 VAE 인코더 경유)과 조건 잠재 2프레임에 대한 |∂s/∂x|를 얻는다. 어텐션 라우팅은 예측 잠재 프레임 5의 질의(전체·생성된 공 토큰 가중)에서 조건 토큰 3,120개로 가는 softmax(QKᵀ)를 28블록에서 재계산한다. 둘 다 세그멘테이션 마스크 영역(공·관련 구조·미끼·지지면·로봇팔·잡동사니·배경)으로 집계하고, 면적 대비 밀도(1 = 균일)를 병기한다.</p>
</div><div>
<div class="callout"><b>정합 검사.</b> (i) 무관 스칼라(예측 잠재 평균)의 지도와의 상관 — 높으면 원시 그래디언트가 점수 무관 성분에 지배됨; (ii) 마지막 k 블록(4/8/16/28) 무작위화 후 지도 상관 — 떨어져야 가중치 의존; (iii) seed 4개 간 변동계수; (iv) Sobel 가장자리 기준선.</div>
<div class="callout" style="margin-top:.8rem"><b>사전 등록 가설.</b> H1 핵심 조건에서 관련 구조 질량(밀도) > 미끼; H2 구조가 없는 셀(평지·표적 없음·긴 플레이트)에서 그 자리의 질량 감소; H3 영역 순서가 ② 효과 순서와 일치. 하나라도 깨지면 열지도는 "선별 신호"로 강등.</div>
<div class="callout" style="margin-top:.8rem"><b>목적.</b> 기존 결과("언덕 무반응", "반전 무작위", "부양")를 관측 실패와 동역학 실패로 분류하고, 반응이 시작되는 구조를 찾는다. 세 씬을 같은 표에 놓아 필요한 정보의 종류(기하·상대·경계+속도)별로 실패 위치가 같은지 본다.</div>
</div></div>"""


def build(root, out, text_path):
    text = (load_json(text_path) if text_path else None) or {}
    verify = load_json(root / "verify.json") or {}
    scenes = [s for s in ("hill", "collide", "edge") if (root / s / "manifest.jsonl").exists()]
    parts = ['<title>PVR PoC 결과</title>',
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">',
             f"<style>{CSS}</style>", '<div class="page">']
    parts.append('<header class="top"><div class="eyebrow">SimDROID · PhysicsGen 트랙 · PVR PoC 결과 · 2026-09-07 KST</div><h1>PVR PoC 결과</h1>'
                 f'<p class="lede">{text.get("lede", "세 씬(Predictive Hill · Collide · Edge/Fall) × Cosmos-Predict2-2B Video2World에서, 세계 모델이 사건 전에 결정 구조를 정확하게 사용하는지를 세 층(정확도·의존·위치)으로 잰 첫 실험의 결과.")}</p>'
                 + '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in text.get("chips", [])) + '</div></header>')
    if text.get("summary"):
        parts.append(f'<section><h2><span class="num">00</span>한눈에</h2><div class="callout key">{text["summary"]}</div></section>')
    parts.append('<section id="scenes"><h2><span class="num">01</span>씬 구성과 목적</h2>')
    parts.append(f'<div class="prose">{text.get("scenes_intro", "")}</div>')
    cards = []
    for s in scenes:
        summ = load_json(root / s / "summary.json") or {}
        cards.append(f'<div class="card"><h3>{SCENE_NAMES[s]} <span class="mono small">{FAMILY[s]}</span></h3><p>{PURPOSE[s][0]}</p><p class="small">{PURPOSE[s][1]}</p>'
                     f'<p class="small">조건 {summ.get("samples", "?")}, 480×832·16 fps, 조건 5프레임(0–0.25 s), 지평 21프레임. 물리는 Phase A/B 클래스 그대로(카메라 fovy 26·미끼·확장만 추가).</p></div>')
    parts.append('<div class="grid-3">' + "".join(cards) + "</div>")
    for s in scenes:
        parts.append(fig(root / s / "previews" / f"{FAMILY[s]}.png", f"{SCENE_NAMES[s]} 조건 미리보기: 열 = 조건(S 또는 v0, GT 결과 y), 행 = t=0 / 조건 종료 0.25 s / +0.31 s / 사건 / t=1.25 s."))
    vt = []
    if verify:
        idn, gates, masks = verify.get("identity", {}), verify.get("gates", {}), verify.get("masks", {})
        for s in scenes:
            ident = idn.get(s, [])
            if s == "hill":
                idtxt = ("정렬 후 x 차 ≤ " + f2(max(r["aligned_x_max_abs_diff"] for r in ident), 3) + " m, 결과 동일") if ident else "–"
            else:
                idtxt = ("궤적 차 ≤ " + f"{max(r['trace_max_abs_diff'] for r in ident):.1e}") if ident else "–"
            g = gates.get(s, [])
            m = masks.get(s, {})
            vt.append(f'<tr><td>{SCENE_NAMES[s]}</td><td>{idtxt}</td><td>{sum(1 for r in g if not r["problems"])}/{len(g)}</td>'
                      f'<td>{f2(m.get("iou_min"), 3)} / {f2(m.get("iou_mean"), 3)}</td><td>{(load_json(root / s / "analysis" / "summary.json") or {}).get("gt_selftest", "–")}</td></tr>')
    parts.append('<h3 style="margin-top:1.2rem">씬 검증</h3><div class="tablewrap"><table><thead><tr><th>씬</th><th>물리 동일성(기존 클래스 대비)</th><th>물리 게이트</th><th>마스크 IoU 최소/평균</th><th>판독기 GT 자가검증</th></tr></thead><tbody>'
                 + "".join(vt) + "</tbody></table></div>")
    parts.append("</section>")
    parts.append('<section id="method"><h2><span class="num">02</span>모델 검증 방식과 목적</h2>' + text.get("method", METHOD_DEFAULT) + "</section>")
    for s in scenes:
        parts.append(scene_block(s, root / s, text))
    parts.append('<section id="impl"><h2><span class="num">04</span>시사점</h2>' + text.get("implications", '<div class="callout warn">판독 후 작성</div>') + "</section>")
    parts.append('<footer>Claude(지훈 세션) 작성 · 시각 KST · 근거: artifacts/pvr/*/analysis/summary.json, attrib/summary.json, verify.json · 설계: DESIGN_PVR_POC_2026-09-06.md · 기록: PROGRESS (47)</footer></div>')
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/pvr"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pvr/report.html"))
    parser.add_argument("--text", type=Path, default=Path("artifacts/pvr/report_text.json"))
    args = parser.parse_args()
    build(args.root.resolve(), args.out, args.text if args.text.exists() else None)


if __name__ == "__main__":
    main()
