"""Splice Cosmos evaluation results into the Bundle A artifact HTML."""
import base64, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/nvme/migration/jihun/SimDROID/code_vwm/artifacts/bundle_a")
ASSETS = ROOT / "analysis" / "artifact_assets"

def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()

def pct(x, nd=0):
    return f"{100*x:.{nd}f}%" if x is not None and not (isinstance(x,float) and np.isnan(x)) else "—"

def main(src, dst, n_seeds):
    stats = json.loads((ASSETS/"stats.json").read_text())
    summary = json.loads((ROOT/"analysis/summary.json").read_text())
    html = Path(src).read_text()

    H, T, P = stats["hill_roll"], stats["two_ball"], stats["pendulum_rod"]
    fitP = stats.get("pendulum_rod_fit")
    fitT = stats.get("two_ball_fit")
    n_total = H["n"] + T["n"] + P["n"] + 2*n_seeds

    # pendulum curve quotes
    pc = {c["S"]: c for c in summary["pendulum_rod"]["curve"]}
    def pq(S):
        c = pc[S]; nd = c["n"]-c["n_undecided"]
        return f'{c["p_success"]:.2f}' if nd else "—"

    # precheck aggregates
    pre = summary["prechecks"]
    hill_pre = [v for k,v in pre.items() if k.startswith("hill_roll_pre")]
    wall_pre = [v for k,v in pre.items() if k.startswith("wall_bounce_pre")]
    hill_pre_pass = sum(v["prediction"]==1 for v in hill_pre)
    wall_pre_ok = sum(v["prediction"]==v["gt"] for v in wall_pre)

    # ---------- hero patches ----------
    html, n = re.subn(r"씬 리뷰 — Cosmos 실행 전", "평가 결과 보고 · 2026-09-02", html); assert n==1, "eyebrow"
    html, n = re.subn(
        r"MuJoCo 사전 검사\(§2\.5\)는 전부 통과했고,\s*\n\s*Cosmos 발사는 이 페이지의 피드백을 받은 뒤 진행한다\.",
        f"MuJoCo 사전 검사(§2.5) 전부 통과 후 Cosmos 평가까지 실행했다 — 중립 프롬프트, "
        f"{n_seeds} seed × 35행 = {n_total} 롤아웃. 이 페이지가 그 결과 보고서다.", html); assert n==1, "lead"
    chips = (
        f'<span class="chip ok">평가 완료 — {n_seeds} seed · {n_total} 롤아웃</span>\n'
        f'    <span class="chip warn">A1 문턱 불인지 — 전 구간 "넘어감" {pct(H["always_pass_frac"])}</span>\n'
        f'    <span class="chip warn">A2 Σmv 붕괴 — 평균 {T["momentum_mean"]:+.2f} (GT 0.98–1.00)</span>\n'
        f'    <span class="chip warn">A3 문턱 변위 — PSE≈{fitP["PSE"]:.2f} (GT 1.00)</span>'
        if fitP else f'<span class="chip ok">평가 완료 — {n_seeds} seed</span>')
    html, n = re.subn(r'<span class="chip hot">Cosmos 실행 대기 — 피드백 요청</span>', chips, html); assert n==1, "chips"
    html, n = re.subn(r"(<h2>스펙과 달리한 결정) — 피드백 요청(</h2>)",
                      r"\1 — 7건 전부 승인됨 (지훈, 2026-09-01)\2", html); assert n==1, "sec05 title"

    # ---------- new sections ----------
    img_curves = b64(ASSETS/"curves3.png"); img_laws = b64(ASSETS/"laws3.png")
    sheets_html = ""
    for fam, label in (("hill_roll","A1"),("two_ball","A2"),("pendulum_rod","A3")):
        sp = ASSETS / f"sheet_{fam}.jpg"
        if sp.exists():
            sheets_html += (f'<div class="media scroll"><img src="data:image/jpeg;base64,{b64(sp)}" '
                            f'alt="{label} 결과 시트: 행=S 격자, 열=GT+seed별 생성, 사건 프레임"></div>\n'
                            f'<p class="cap">{label} — 각 행이 S 격자 한 점(위→아래로 S 증가), 첫 열이 GT, 이후 열이 seed별 생성 결과 (사건 프레임 t_event 시점). 칸 위 표기 1=통과·0=반환·?=미결.</p>\n')

    fitP_txt = f'PSE≈{fitP["PSE"]:.2f}, 기울기 k≈{fitP["slope"]:.0f}' if fitP else "적합 불가"
    fitT_txt = ("평탄 — 문턱 없음" if (not fitT or fitT["slope"] < 1.0 or not (0.45 < fitT["PSE"] < 2.0))
                else f'PSE≈{fitT["PSE"]:.2f}, k≈{fitT["slope"]:.1f}')

    new = f"""<section>
  <div class="sec-head"><span class="n">06</span><h2>Cosmos 평가 실행 — 무엇을 어떻게 돌렸나</h2></div>
  <div class="kv">
    <dt>모델</dt><dd>nvidia/Cosmos-Predict2-2B-Video2World · model-480p-16fps.pt · diffusers 0.35.1 (Cosmos2VideoToWorldPipeline)</dd>
    <dt>샘플링</dt><dd>21프레임(1.3125 s) · 35 steps · guidance 7.0 · fps 16 · 중립 프롬프트 + 표준 negative</dd>
    <dt>조건</dt><dd>프레임 0–4 (5프레임 비디오 컨디셔닝) — 롤아웃 초반이 조건 구간과 픽셀 일치함을 전 씬에서 확인</dd>
    <dt>규모</dt><dd>35행(33 격자 + 2 사전검사) × {n_seeds} seed = {n_total} 롤아웃 · 실측 ~14 s/롤아웃 (H200 1장, 총 ~3.3 h; 수집 2026-09-01, 판독 09-02)</dd>
    <dt>판정</dt><dd>21프레임 마스크 궤적 기반(마지막 프레임 단독 판정 아님) · GT 자가검증 34/35 후 동결 (유일 미결정 = S=1.00 나이프에지) · 등속 인코더 기준선 병기</dd>
  </div>
  <p class="small muted">실행 노트: 발사 직전 점검에서 러너의 컨디셔닝 계약(manifest <span class="mono">kind=="dynamic"</span>일 때만
  5프레임 비디오 조건)과 제 manifest 필드가 충돌해 프레임 0 단독 조건으로 돌아가는 문제를 발견,
  수정 후 조건 구간 픽셀 일치·motion 상관(0.09→0.75) 재검증을 거쳐 전량 재수집했다. 여기 실린 결과는 전부 수정 후 데이터다.</p>
</section>

<section>
  <div class="sec-head"><span class="n">07</span><h2>결과 — 방향 선택 곡선과 법칙 지표</h2></div>
  <p class="lead">한 줄 요약: <b>세 씬 모두 임계 구조(S=1 문턱)를 재현하지 못했다.</b>
  A1은 전 구간 "넘어감"(인코더 기준선과 동일), A2는 접촉까지만 맞고 운동량 전달이 붕괴,
  A3만 문턱이 존재하되 S≈{fitP["PSE"]:.2f}로 낮게 변위 — 셋 다 방향은 다르지만 원인은 하나,
  <b>운동 지속 사전이 보존법칙 제약을 이긴다</b>.</p>

  <div class="media"><img src="data:image/png;base64,{img_curves}" alt="P(통과 예측|S) 3패널 곡선: GT 계단, 인코더 기준선, Cosmos 곡선"></div>
  <p class="cap">결정 곡선 P(통과 방향 예측 | 판정 성립) vs S. 파랑 = GT 계단(S=1에서 0→1), 점선 = 등속 인코더 기준선(항상 통과=1),
  빨강 = Cosmos({n_seeds} seed 평균, 수직선 = Wilson 95% CI, 하단 회색 막대 = 미결 비율).</p>

  <div class="card">
    <div class="card-head"><span class="scene-id">A1</span><h2>구름 언덕 — 문턱을 전혀 읽지 않는다</h2><span class="law">정확도 {pct(H["acc_nonboundary"])} = 인코더 {pct(H["encoder_acc"])}</span></div>
    <p>격자 전 구간에서 통과 예측 {pct(H["always_pass_frac"])} — S=0.50(운동에너지가 필요량의 절반)에서도,
    감속 사전검사 S=0.35에서도({hill_pre_pass}/{len(hill_pre)} seed가 통과 예측) 공은 언덕을 그냥 넘는다.
    로지스틱 적합 자체가 퇴화(전-통과)라 PSE가 정의되지 않는다. 정지 프레임에 항상 보이는 결정 변수(언덕 높이)가
    예측에 개입한 증거가 없다.</p>
  </div>
  <div class="card">
    <div class="card-head"><span class="scene-id">A2</span><h2>두 공 충돌 — 접촉은 읽고 응답은 버린다</h2><span class="law">Σmv 평균 {T["momentum_mean"]:+.2f} · 중앙값 {stats["two_ball_momentum"]["median"]:+.2f}</span></div>
    <p>접촉 시점 국재화는 정상(파란 공이 GT와 같은 프레임대에 움직이기 시작)이나, 그 직후 운동량 전달이 붕괴한다 —
    GT는 Σmv 비 0.98–1.00, 생성은 평균 {T["momentum_mean"]:+.2f}. 전형 패턴은 충돌 후 두 공이 유착·정체.
    방향 판정 정확도 {pct(T["acc_nonboundary"])}는 기준선 {pct(T["encoder_acc"])}보다도 낮다(질량비와 무관한 잡음성 반응).
    결정 곡선은 {fitT_txt}. 질량비 이전의 문제다: 판단 요소가 없는 <b>A2-0 탄성벽 사전검사에서도 {len(wall_pre)} seed 중 {wall_pre_ok}건만 반동</b>했다 —
    접촉 후 반동·운동량 전달 자체를 거의 그리지 않는다.</p>
  </div>
  <div class="card">
    <div class="card-head"><span class="scene-id">A3</span><h2>강체 진자 — 문턱은 있으나 자리가 틀렸다</h2><span class="law">{fitP_txt} (GT 1.00)</span></div>
    <p>유일하게 실질 신호가 있는 씬. S≤0.79에선 대체로 되돌아오고(P={pq(0.5)}~{pq(0.79)}), S≥1.05에선 돈다(P={pq(1.05)}~{pq(2.0)}) —
    정확도 {pct(P["acc_nonboundary"])} &gt; 기준선 {pct(P["encoder_acc"])}. 그러나 문턱이 S≈{fitP["PSE"]:.2f}에 있어
    GT(1.00)보다 낮다: 경계 바로 아래 S=0.89·0.95(GT: 못 넘음)에서 통과 예측 P={pq(0.89)}·{pq(0.95)} — 정점 감속을 읽되
    <b>과소평가</b>해서 돌려보내야 할 스윙을 넘겨버린다.</p>
  </div>

  <div class="media"><img src="data:image/png;base64,{img_laws}" alt="법칙 지표: A1 에너지 기울기 분포, A2 운동량 보존비, A3 기울기 분포"></div>
  <p class="cap">법칙 지표. 왼쪽: A1 생성 롤아웃의 (y, v²) 회귀 기울기 분포(중앙값 {stats["hill_gen_slopes"]["median"]:+.1f}) vs
  구름 이론 −14.0 · 미끄럼 −19.6 · GT 실측 {H["gt_slope"]:+.1f}. 가운데: A2 Σmv(후)/Σmv(전) — GT 0.98–1.00(초록), 생성은 0 주변 산포.
  오른쪽: A3 기울기 분포(GT 실측 {P["gt_slope"]:+.1f}).</p>

  <h3>지도(map) — 채워진 결과</h3>
  <div class="tbl-wrap"><table>
    <tr><th>씬</th><th class="num">비경계 정확도</th><th class="num">인코더 기준선</th><th class="num">PSE / k</th><th class="num">미결</th><th class="num">validity (strict / frac)</th><th class="num">법칙 지표</th></tr>
    <tr><td>A1 hill</td><td class="num">{pct(H["acc_nonboundary"])}</td><td class="num">{pct(H["encoder_acc"])}</td><td class="num">— (전-통과)</td><td class="num">{pct(H["undecided_rate"])}</td><td class="num">{pct(H["valid_rate_strict"])} / {pct(H["valid_frac_mean"])}</td><td class="num">기울기 중앙값 {stats["hill_gen_slopes"]["median"]:+.1f} (GT {H["gt_slope"]:+.1f})</td></tr>
    <tr><td>A2 two-ball</td><td class="num">{pct(T["acc_nonboundary"])}</td><td class="num">{pct(T["encoder_acc"])}</td><td class="num">{fitT_txt}</td><td class="num">{pct(T["undecided_rate"])}</td><td class="num">{pct(T["valid_rate_strict"])} / {pct(T["valid_frac_mean"])}</td><td class="num">Σmv {T["momentum_mean"]:+.2f} (GT 0.98–1.00)</td></tr>
    <tr><td>A3 pendulum</td><td class="num">{pct(P["acc_nonboundary"])}</td><td class="num">{pct(P["encoder_acc"])}</td><td class="num">{fitP_txt}</td><td class="num">{pct(P["undecided_rate"])}</td><td class="num">{pct(P["valid_rate_strict"])} / {pct(P["valid_frac_mean"])}</td><td class="num">기울기 중앙값 {stats["pend_gen_slopes"]["median"]:+.1f} (GT {P["gt_slope"]:+.1f})</td></tr>
  </table></div>
  <p class="small muted">validity strict = 21프레임 전부에서 마스크 면적이 기준의 0.35–2.8×; frac = 프레임 비율 평균.
  strict 하락은 대부분 생성 프레임의 일시적 마스크 소실(1–8프레임)이고 방향 판정은 보간 궤적으로 성립.
  사전검사: wall_bounce {wall_pre_ok}/{len(wall_pre)} seed 정판정, hill 감속(S=0.35) {hill_pre_pass}/{len(hill_pre)} seed가 오판(통과 예측).</p>

  <h3>대표 증거 — 궤적 수치 (판독기 오독 아님을 확인한 사례)</h3>
  <div class="tbl-wrap"><table>
    <tr><th>사례</th><th>GT (MuJoCo)</th><th>Cosmos 생성</th></tr>
    <tr><td>A1 S=0.50 · seed1</td><td>x 최대 354px(정점 416 못 미침) 후 반환, 끝 9px</td><td>53→746px 등속 직진 — 정점을 감속 없이 통과</td></tr>
    <tr><td>A3 S=0.89 · seed2</td><td>정점 140°에서 반환</td><td>프레임 9까지 GT 추종(131°) 후 293°까지 과회전</td></tr>
    <tr><td>A2 S=2.00 · seed3</td><td>파란 공 826px로 이탈, 빨간 공 반전</td><td>접촉 후 정체 — red 486px·blue 502px에 유착</td></tr>
  </table></div>
  <p class="small muted">세 사례 모두 조건 구간(프레임 0–4)은 GT와 픽셀 일치 — 발산은 롤아웃 구간에서만 시작된다.</p>
{sheets_html}</section>

<section>
  <div class="sec-head"><span class="n">08</span><h2>기계공학적 해석 — 가설 사다리 판정</h2></div>
  <div class="tbl-wrap"><table>
    <tr><th>가설</th><th>판정</th><th>근거</th></tr>
    <tr><td>H0 · prior 지배</td><td><span class="chip ok">기각</span></td><td>조건 구간 운동을 정확히 이어받음(초반 픽셀 일치, motion 상관 0.75) — 텍스트/장면 prior가 초기 조건을 덮지 않는다</td></tr>
    <tr><td>H1 · 고체성만</td><td><span class="chip ok">초과</span></td><td>객체 지속·접촉 국재화 정상(A2에서 파란 공이 올바른 프레임대에 반응 시작)</td></tr>
    <tr><td>H2 · 장면 통계(관성 외삽)</td><td><span class="chip warn">A1의 현 위치</span></td><td>언덕 높이라는 정적 결정 변수가 예측에 미개입 — 등속 인코더와 구별 불가</td></tr>
    <tr><td>H3 · 휴리스틱</td><td><span class="chip warn">A3 부분 도달</span></td><td>각감속 신호는 읽힘(문턱 존재, 기준선 초과)하나 크기를 과소평가 → PSE {fitP["PSE"]:.2f}. A2의 "충돌=정지" 반응도 휴리스틱 수준</td></tr>
    <tr><td>H4 · 원리 보존</td><td><span class="chip warn">기각</span></td><td>법칙 지표 전부 실패: A1 에너지 기울기 얕음({stats["hill_gen_slopes"]["median"]:+.1f} vs −14.0), A2 Σmv 붕괴({T["momentum_mean"]:+.2f}), A3 문턱 변위</td></tr>
  </table></div>

  <h3>한 문장으로</h3>
  <p class="ask"><b>Cosmos-Predict2-2B는 "무엇이 움직이고 있는가"는 안다 — "무엇이 그 운동을 멈출 것인가"를 모른다.</b>
  운동 지속 사전(motion-continuation prior)이 위치에너지 장벽(A1)·운동량 수지(A2)·임계 감속(A3)이라는
  보존법칙 제약을 일관되게 이긴다.</p>

  <h3>기계공학 관점의 세부 함의</h3>
  <ul>
    <li><b>에너지 장부가 없다 (A1).</b> v²–y 기울기 분포가 이론값(−14.0)에 모이지 않고 얕은 쪽·양수까지 퍼진다 —
    운동에너지가 고도와 교환된다는 제약 자체가 롤아웃에 없다. 정지 프레임에서 항상 보이는 언덕 높이는
    "장면 소품"으로만 취급된다.</li>
    <li><b>접촉역학은 침투 회피까지만 (A2).</b> 접촉 검출(시점)은 학습돼 있으나 응답(충격량 교환)은 "부드러운 정지"라는
    시각적으로 무난한 사전으로 대체된다. Σmv가 0 근처로 무너지는 것은 렌더 노이즈가 아니라 계 전체 운동량의 소멸이다.</li>
    <li><b>감속 휴리스틱은 있다, 적분이 없다 (A3).</b> 진자에서만 문턱이 생긴 것은 각속도 감소라는 <i>화면 내</i> 신호가
    조건 구간에 직접 보이기 때문으로 해석된다(언덕의 감속은 롤아웃 구간에서야 시작 — 조건 구간에서 안 보임).
    그러나 감속률의 적분(에너지 수지)이 아니라 순간 패턴에 반응하므로 문턱이 낮게 밀린다.</li>
    <li><b>세 실패가 한 방향.</b> A1 전-통과, A2 반응 부족, A3 과회전 — 전부 "지금의 운동을 이어간다" 쪽 오류다.
    무작위 오차가 아니라 편향이며, 따라서 데이터·스케일로 지워질지 별도 검증이 필요한 구조적 결함이다.</li>
  </ul>

  <h3>한계와 다음 단계</h3>
  <ul>
    <li>단일 체크포인트(2B, 480p/16fps)·중립 프롬프트 결과다. 스펙의 프롬프트 대조(정답/오답 서술)와
    denoising-loss 판단 채점(VoE)은 후속 — 텍스트가 물리 신호를 보완하는지가 다음 질문.</li>
    <li>판정은 픽셀 마스크 기반: 생성 프레임의 일시적 객체 소실이 strict validity를 낮춘다(frac 병기로 보정).
    방향 판정 자체는 GT 자가검증 34/35로 동결된 규칙을 그대로 적용했다.</li>
    <li>지평 21프레임(1.31 s) — t_event가 지평 안에 오도록 설계된 격자라 지평 부족의 영향은 경계점에 국한된다.</li>
  </ul>
</section>
"""
    html, n = re.subn(r'<section>\s*<div class="sec-head"><span class="n">06</span>.*?</section>',
                      new.strip(), html, flags=re.S); assert n==1, "sec06 splice"
    html, n = re.subn(r"· 2026-09-01 · 문지훈/Claude",
                      f"· 판독 <span class=\"mono\">artifacts/bundle_a/analysis/</span> · 2026-09-01 씬 구현·Cosmos 수집 · 09-02 전체 판독 · 문지훈/Claude", html); assert n==1, "footer"
    Path(dst).write_text(html)
    print(f"OK -> {dst}  ({len(html)/1e6:.2f} MB)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
