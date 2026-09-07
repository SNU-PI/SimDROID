"""Explainer page for the PVR PoC: scenes & purpose, validation method & purpose, results & implications.

Reads artifacts/pvr/*/analysis/summary.json, edge speed diagnostics, variance_decomp.json and
attrib/summary.json, draws every chart as inline SVG from the numbers, and embeds small real
frames (condition -> GT event vs model prediction) as base64 JPEG.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from exp.pvr_heatmap_figs import panels as heat_panels

# ------------------------------------------------------------------ data helpers

def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def conds(summary):
    return {c["cond"]: c for c in summary["conditions"]}


def effects(summary):
    return {e["label"]: e for e in summary["effects"]}


def jpeg_b64(img: Image.Image, width=416, quality=80):
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def frames_for(root, scene, cond, seed):
    """condition frame 4, GT event & last, model event & last -> dict of data URIs + event frame."""
    manifest = [json.loads(l) for l in (root / scene / "manifest.jsonl").read_text().splitlines() if l.strip()]
    rec = next(r for r in manifest if r["cond"] == cond)
    z = np.load(root / scene / rec["input_npz"])
    ev = int(rec["event_frame"]) if rec["event_frame"] is not None else 12
    ev = max(6, min(ev, rec["rollout_frames"] - 1))
    gt = imageio.mimread(root / scene / rec["gt_clip"], memtest=False)[: rec["rollout_frames"]]
    clip = root / scene / "cosmos_v2w" / f"seed_{seed:02d}" / rec["id"] / "rollout.mp4"
    model = imageio.mimread(clip, memtest=False)[: rec["rollout_frames"]]
    last = min(len(gt), len(model)) - 1
    ev2 = min(ev + 4, last)
    out = {"cond": jpeg_b64(Image.fromarray(z["condition_primary"][4])),
           "gt_ev": jpeg_b64(Image.fromarray(gt[ev])), "model_ev": jpeg_b64(Image.fromarray(model[ev])),
           "gt_ev2": jpeg_b64(Image.fromarray(gt[ev2])), "model_ev2": jpeg_b64(Image.fromarray(model[ev2])),
           "gt_last": jpeg_b64(Image.fromarray(gt[last])), "model_last": jpeg_b64(Image.fromarray(model[last])),
           "ev": ev, "ev2": ev2, "last": last, "id": rec["id"], "seed": seed}
    return out


# ------------------------------------------------------------------ SVG chart primitives

def fmt(v, d=2):
    return f"{v:.{d}f}"


class Chart:
    """Minimal SVG scatter/line chart with linear or log x."""

    def __init__(self, w=640, h=360, ml=58, mr=24, mt=28, mb=54, xlim=(0, 1), ylim=(0, 1), xlog=False):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.xlim, self.ylim, self.xlog = xlim, ylim, xlog
        self.parts = []

    def X(self, x):
        a, b = self.xlim
        if self.xlog:
            x, a, b = math.log(x), math.log(a), math.log(b)
        return self.ml + (x - a) / (b - a) * (self.w - self.ml - self.mr)

    def Y(self, y):
        a, b = self.ylim
        return self.h - self.mb - (y - a) / (b - a) * (self.h - self.mt - self.mb)

    def axes(self, xticks, yticks, xlabel, ylabel, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}"):
        p = self.parts
        x0, x1, y0, y1 = self.ml, self.w - self.mr, self.h - self.mb, self.mt
        for yt in yticks:
            p.append(f'<line x1="{x0}" y1="{self.Y(yt):.1f}" x2="{x1}" y2="{self.Y(yt):.1f}" class="grid"/>')
            p.append(f'<text x="{x0 - 8}" y="{self.Y(yt) + 4:.1f}" class="tick" text-anchor="end">{yfmt(yt)}</text>')
        for xt in xticks:
            p.append(f'<text x="{self.X(xt):.1f}" y="{y0 + 18}" class="tick" text-anchor="middle">{xfmt(xt)}</text>')
        p.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')
        p.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{self.h - 12}" class="label" text-anchor="middle">{xlabel}</text>')
        p.append(f'<text transform="translate(16 {(y0 + y1) / 2:.1f}) rotate(-90)" class="label" text-anchor="middle">{ylabel}</text>')

    def hline(self, y, cls="ref", label=None):
        self.parts.append(f'<line x1="{self.ml}" y1="{self.Y(y):.1f}" x2="{self.w - self.mr}" y2="{self.Y(y):.1f}" class="{cls}"/>')
        if label:
            self.parts.append(f'<text x="{self.w - self.mr - 4}" y="{self.Y(y) - 5:.1f}" class="note" text-anchor="end">{label}</text>')

    def line(self, pts, cls):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        self.parts.append(f'<path d="{d}" class="{cls}" fill="none"/>')

    def points(self, pts, cls, r=5, err=None, labels=None):
        for i, (x, y) in enumerate(pts):
            if err is not None and err[i]:
                self.parts.append(f'<line x1="{self.X(x):.1f}" y1="{self.Y(y - err[i]):.1f}" x2="{self.X(x):.1f}" y2="{self.Y(y + err[i]):.1f}" class="{cls} err"/>')
            self.parts.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" class="{cls} dot"/>')
            if labels and labels[i]:
                self.parts.append(f'<text x="{self.X(x) + 8:.1f}" y="{self.Y(y) - 8:.1f}" class="note">{labels[i]}</text>')

    def text(self, x, y, s, cls="note", anchor="start"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>')

    def legend(self, items, x=None, y=None):
        x = self.ml + 10 if x is None else x
        y = self.mt + 6 if y is None else y
        for i, (cls, name) in enumerate(items):
            yy = y + i * 18
            self.parts.append(f'<circle cx="{x + 5}" cy="{yy}" r="5" class="{cls} dot"/><text x="{x + 16}" y="{yy + 4}" class="legend">{name}</text>')

    def svg(self, title=""):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" class="chart" role="img" aria-label="{title}">' + "".join(self.parts) + "</svg>")


def forest(rows, xlim, xlabel, w=640, unit_note=""):
    """rows: list of (name, kind, mean, lo, hi, gt) -> horizontal CI plot."""
    rh, mt, mb, ml, mr = 34, 22, 46, 150, 24
    h = mt + mb + rh * len(rows)
    c = Chart(w, h, ml, mr, mt, mb, xlim, (0, len(rows)))
    X = c.X
    p = c.parts
    p.append(f'<line x1="{X(0):.1f}" y1="{mt}" x2="{X(0):.1f}" y2="{h - mb}" class="zero"/>')
    for i, (name, kind, m, lo, hi, gt) in enumerate(rows):
        y = mt + rh * i + rh / 2
        p.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" class="rowname {kind}" text-anchor="end">{name}</text>')
        lo_c, hi_c = max(lo, xlim[0]), min(hi, xlim[1])
        p.append(f'<line x1="{X(lo_c):.1f}" y1="{y:.1f}" x2="{X(hi_c):.1f}" y2="{y:.1f}" class="ci {kind}"/>')
        p.append(f'<circle cx="{X(min(max(m, xlim[0]), xlim[1])):.1f}" cy="{y:.1f}" r="5.5" class="{kind} dot"/>')
        if gt is not None and xlim[0] <= gt <= xlim[1]:
            p.append(f'<path d="M{X(gt):.1f},{y - 7:.1f} l6,7 l-6,7 l-6,-7 z" class="gtmark"/>')
        elif gt is not None:
            side = xlim[0] if gt < xlim[0] else xlim[1]
            p.append(f'<text x="{X(side) + (-4 if gt < xlim[0] else 4):.1f}" y="{y + 4:.1f}" class="note" text-anchor="{"end" if gt < xlim[0] else "start"}">GT {gt:+.2g} ↗</text>')
    ticks = np.linspace(xlim[0], xlim[1], 5)
    for t in ticks:
        p.append(f'<text x="{X(t):.1f}" y="{h - mb + 18}" class="tick" text-anchor="middle">{t:+.2g}</text>')
    p.append(f'<line x1="{ml}" y1="{h - mb}" x2="{w - mr}" y2="{h - mb}" class="axis"/>')
    p.append(f'<text x="{(ml + w - mr) / 2:.1f}" y="{h - 10}" class="label" text-anchor="middle">{xlabel}{unit_note}</text>')
    return c.svg("paired effects")


def stacked(rows, w=640):
    """rows: (name, cond, seed, resid) fractions."""
    rh, ml, mr, mt = 40, 90, 24, 8
    h = mt + rh * len(rows) + 30
    p = []
    span = w - ml - mr
    for i, (name, a, b, r) in enumerate(rows):
        y = mt + rh * i
        x = ml
        for frac, cls, lab in ((a, "vcond", "조건"), (b, "vseed", "seed"), (r, "vres", "잔차")):
            ww = frac * span
            p.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{ww:.1f}" height="{rh - 14}" class="{cls}"/>')
            if ww > 46:
                p.append(f'<text x="{x + ww / 2:.1f}" y="{y + rh / 2 + 3:.1f}" class="barlabel" text-anchor="middle">{lab} {frac * 100:.0f}%</text>')
            elif ww > 16:
                p.append(f'<text x="{x + ww / 2:.1f}" y="{y + rh / 2 + 3:.1f}" class="barlabel" text-anchor="middle">{frac * 100:.0f}%</text>')
            x += ww
        p.append(f'<text x="{ml - 10}" y="{y + rh / 2 + 4:.1f}" class="rowname" text-anchor="end">{name}</text>')
    y = mt + rh * len(rows) + 10
    for j, (cls, lab) in enumerate((("vcond", "물리 조건 주효과"), ("vseed", "잡음 seed 주효과"), ("vres", "잔차(조건×seed)"))):
        x = ml + j * 190
        p.append(f'<rect x="{x}" y="{y}" width="14" height="14" class="{cls}"/><text x="{x + 20}" y="{y + 12}" class="legend">{lab}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="variance decomposition">' + "".join(p) + "</svg>"


def heat_row(root, scene, cond, title, reading):
    hp = heat_panels(root, scene, cond)
    def fig(key, cap):
        return f'<figure><img src="{jpeg_b64(hp[key], 416, 82)}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
    return ('<div class="scene"><div class="head"><h3>' + title + '</h3><span class="fam">' + hp["id"] + f' · seed {len(hp["seeds"])}개 평균 · step {hp["step"]} · 블록 {hp["block"]}</span></div>'
            '<div class="frames">'
            + fig("frame", "① 조건 마지막 프레임과 영역 윤곽 — 노랑 = 관련 구조, 보라 = 미끼, 흰색 = 공 (지형의 초록은 씬 자체의 색)")
            + fig("grad", "② 그래디언트 |∂s/∂x| — s = 예측된 공 위치 점수. 밝을수록 그 픽셀이 점수를 많이 움직임")
            + fig("wrong", f"③ 같은 방식, 점수 대신 무관한 스칼라(잠재 평균). ②와의 상관 {hp['corr_grad_wrong']:.2f}")
            + fig("att_ball", "④ 어텐션 — 예측된 공의 토큰이 조건 프레임 1–4 토큰을 읽는 양(후기 블록)")
            + fig("att_all", "⑤ 어텐션 — 예측 프레임의 모든 토큰 평균")
            + '</div><div class="reading">' + reading + '</div></div>')


def heatmap_primer():
    """Plain-language explanation of how the two heatmaps are built, with one pipeline diagram."""
    d = []
    def box(x, y, w, h, title, sub, cls="lbox"):
        d.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" class="{cls}"/>')
        d.append(f'<text x="{x + w / 2}" y="{y + 24}" class="ltitle" text-anchor="middle" style="font-size:13px">{title}</text>')
        d.append(f'<text x="{x + w / 2}" y="{y + 42}" class="ldesc" text-anchor="middle">{sub}</text>')
    box(10, 20, 120, 56, "조건 5프레임", "사건 0.25 s 전까지")
    box(160, 20, 110, 56, "VAE 인코더", "픽셀 → 잠재 2프레임")
    box(300, 20, 150, 56, "DiT (35 step 중 1 step)", "잡음 상태 고정")
    box(480, 20, 110, 56, "x̂₀ 예측", "깨끗한 미래 추정")
    box(620, 20, 110, 56, "탐지기 s", "마지막 프레임의 공 위치")
    for x0, x1 in ((130, 160), (270, 300), (450, 480), (590, 620)):
        d.append(f'<path d="M{x0},48 h{x1 - x0}" class="edge arr"/>')
    d.append('<path d="M640,90 q-300,60 -560,0" class="edge rel arr" style="stroke-dasharray:6 4"/>')
    d.append('<text x="360" y="128" class="stag rel" text-anchor="middle">그래디언트 열지도 = 각 픽셀을 살짝 바꿨을 때 s가 얼마나 변하는가 (역전파로 한 번에 계산)</text>')
    d.append('<rect x="300" y="150" width="150" height="46" rx="7" class="lbox l2"/><text x="375" y="170" class="ltitle" text-anchor="middle" style="font-size:13px">DiT 블록 안</text><text x="375" y="187" class="ldesc" text-anchor="middle">예측 공 토큰 Q · 조건 토큰 K</text>')
    d.append('<path d="M300,173 q-90,0 -180,-100" class="edge dec arr" style="stroke-dasharray:6 4"/>')
    d.append('<text x="120" y="215" class="stag dec" text-anchor="start">어텐션 열지도 = 예측된 공의 토큰이 조건 프레임의 어느 토큰을 얼마나 참고했는가 (계산된 비율을 그대로 읽음)</text>')
    svg = '<svg viewBox="0 0 750 230" class="schem" role="img" aria-label="heatmap pipeline" style="max-width:760px">' + "".join(d) + "</svg>"
    return (
        '<div class="scene"><div class="head"><h3>열지도는 어떻게 만들었나 — 쉬운 설명</h3></div>'
        '<div class="prose"><p><b>알고 싶은 것.</b> 모델이 미래를 그릴 때 입력 화면의 <i>어느 부분을 얼마나 썼는가</i>를 화면 위에 색으로 칠하는 것이 열지도다. 우리는 서로 다른 원리의 지도 두 가지를 만들었다.</p></div>'
        '<div style="margin:.8rem 0">' + svg + '</div>'
        '<div class="grid3" style="grid-template-columns:repeat(2,minmax(0,1fr))">'
        '<div class="step"><div class="n">지도 1 — 그래디언트(민감도) 지도</div>'
        '<p><b>비유.</b> 입력 사진의 픽셀 하나를 아주 조금 밝게 해 보고, 그때 모델이 예측한 공의 위치가 얼마나 움직이는지를 잰다. 이것을 모든 픽셀에 대해 반복하면 "이 픽셀이 예측을 얼마나 흔드는가"의 지도가 된다. 실제로는 하나씩 시험하지 않고 미분(역전파)으로 한 번에 계산한다.</p>'
        '<p><b>필요한 준비 둘.</b> (1) 출력이 "숫자 하나"여야 미분할 수 있다. 그래서 마지막 예측 프레임의 잠재에서 공의 가로 위치를 읽는 작은 탐지기(16채널 로지스틱 회귀 → 뾰족한 softmax 중심)를 미리 학습해 붙였다. 실제 클립에서 오차 중앙값 약 1셀. (2) 확산 모델은 35단계에 걸쳐 그림을 다듬으므로 전체를 미분하면 비용이 너무 크다. 그래서 한 단계(20·24·28)에서 잡음 상태를 고정하고, 그 단계가 내놓는 "깨끗한 그림 추정" x̂₀까지만 미분했다.</p>'
        '<p><b>믿어도 되는지 확인하는 법.</b> 공 위치 점수 대신 아무 의미 없는 숫자(잠재의 평균)로 같은 미분을 해 본다. 두 지도가 닮으면 그 지도는 "공의 미래를 정하는 곳"이 아니라 "모델이 입력 전반에 민감한 곳"을 그린 것이다. 이번 결과가 그랬다(상관 0.64–0.85).</p></div>'
        '<div class="step"><div class="n">지도 2 — 어텐션(참고) 지도</div>'
        '<p><b>비유.</b> 트랜스포머는 화면을 16×16 픽셀 조각(토큰)으로 나누고, 매 층에서 각 조각이 다른 조각들을 "얼마나 참고하는가"의 비율을 계산해 정보를 섞는다. 이 비율은 모델이 실제로 계산하는 값이라 미분 없이 그대로 꺼내 읽을 수 있다.</p>'
        '<p><b>무엇을 그렸나.</b> 예측 프레임에서 공이 있는 조각들이 조건 프레임(모델 입력)의 어느 조각을 얼마나 참고했는지를, 28개 블록 중 후기 블록에서 읽어 조건 프레임 위에 칠했다. 공 조각의 위치는 생성된 영상에서 빨간 공을 찾아 정했다.</p>'
        '<p><b>한계.</b> "참고했다"와 "그 정보로 결과를 정했다"는 다르다. 어텐션은 정보가 어디서 어디로 흘렀는지의 배관도이지, 그 정보가 결과에 쓰였다는 증명이 아니다. 그래서 어텐션의 "관련 &gt; 미끼" 방향성은 반사실 결과와 맞을 때만 해석했다.</p></div></div>'
        '<div class="callout" style="margin-top:1rem"><b>영역별로 더하는 법.</b> 시뮬레이터가 각 픽셀이 공·관련 구조·미끼·지지면·로봇팔·잡동사니·배경 중 무엇인지 알려 주므로(세그멘테이션), 지도의 밝기를 영역별로 합산한다. 넓은 영역은 합이 커지므로 면적으로 나눈 <b>밀도</b>(1 = 화면 전체에 고르게 퍼진 경우)로 비교한다. 실행 전에 정한 통과 기준은 "관련 구조의 밀도 &gt; 미끼의 밀도"였고, 세 씬 모두 통과하지 못했다.</div></div>')


# ------------------------------------------------------------------ scene schematics (SVG)

def schematic(scene):
    """Side-view cartoon: ball moving right, relevant structure ahead, same-type decoy behind."""
    w, h = 520, 170
    g = 128
    ball = '<circle cx="{x}" cy="{y}" r="13" class="sball"/>'
    arrow = '<path d="M{x},{y} h34 l-8,-7 m8,7 l-8,7" class="sarrow"/>'
    parts = [f'<line x1="20" y1="{g}" x2="{w - 20}" y2="{g}" class="sground"/>']
    if scene == "hill":
        parts.append(f'<path d="M300,{g} q45,-70 90,0" class="srel"/>')
        parts.append(f'<path d="M60,{g} q35,-40 70,0" class="sdec"/>')
        parts.append(ball.format(x=200, y=g - 13) + arrow.format(x=218, y=g - 13))
        parts.append(f'<text x="345" y="{g - 78}" class="stag rel" text-anchor="middle">관련: 경로 위 언덕 높이 h</text>')
        parts.append(f'<text x="95" y="{g - 48}" class="stag dec" text-anchor="middle">미끼: 공 뒤 언덕 h_d</text>')
        parts.append(f'<text x="200" y="{g + 26}" class="snote" text-anchor="middle">v₀ = 1.0 m/s → 넘는가? 되돌아오는가?</text>')
    elif scene == "collide":
        parts.append(f'<circle cx="360" cy="{g - 19}" r="19" class="srel ball2"/>')
        parts.append(f'<circle cx="80" cy="{g - 15}" r="15" class="sdec ball2"/>')
        parts.append(ball.format(x=200, y=g - 13) + arrow.format(x=218, y=g - 13))
        parts.append(f'<text x="360" y="{g - 52}" class="stag rel" text-anchor="middle">관련: 표적 공 크기 (질량비 S)</text>')
        parts.append(f'<text x="80" y="{g - 42}" class="stag dec" text-anchor="middle">미끼: 공 뒤의 공</text>')
        parts.append(f'<text x="220" y="{g + 26}" class="snote" text-anchor="middle">충돌 후 계속 가는가? 되튀는가? 얼마나 느려지는가?</text>')
    else:
        parts = [f'<rect x="60" y="{g}" width="300" height="12" class="splate"/>',
                 f'<line x1="360" y1="{g}" x2="360" y2="{g + 12}" class="srel thick"/>',
                 f'<line x1="60" y1="{g}" x2="60" y2="{g + 12}" class="sdec thick"/>',
                 f'<line x1="20" y1="{g + 50}" x2="{w - 20}" y2="{g + 50}" class="sground"/>',
                 ball.format(x=180, y=g - 13) + arrow.format(x=198, y=g - 13),
                 f'<text x="360" y="{g - 30}" class="stag rel" text-anchor="middle">관련: 모서리 위치 (t_edge = 거리 ÷ v₀)</text>',
                 f'<text x="60" y="{g - 30}" class="stag dec" text-anchor="middle">미끼: 판의 왼쪽 끝</text>',
                 f'<path d="M372,{g + 8} q30,10 30,40" class="sfall"/>',
                 f'<text x="230" y="{g + 40}" class="snote" text-anchor="middle">언제 모서리를 넘어 떨어지는가?</text>']
        w, h = 520, 200
    return f'<svg viewBox="0 0 {w} {h}" class="schem" role="img" aria-label="{scene} schematic">' + "".join(parts) + "</svg>"


def design_grid():
    """2x2 A/B/C/D + ladder + extension cartoon."""
    p = []
    cells = {"A": (60, 40), "B": (60, 130), "C": (250, 40), "D": (250, 130)}
    names = {"A": "A  관련 작음 · 미끼 작음", "B": "B  관련 큼 · 미끼 작음", "C": "C  관련 작음 · 미끼 큼", "D": "D  관련 큼 · 미끼 큼"}
    for k, (x, y) in cells.items():
        p.append(f'<rect x="{x}" y="{y}" width="160" height="60" rx="6" class="cell"/>')
        p.append(f'<text x="{x + 80}" y="{y + 27}" class="cellname" text-anchor="middle">{k}</text>')
        p.append(f'<text x="{x + 80}" y="{y + 47}" class="celldesc" text-anchor="middle">{names[k][3:]}</text>')
    p.append('<path d="M140,100 v30" class="edge rel arr"/><text x="128" y="119" class="stag rel" text-anchor="end">관련 편집</text>')
    p.append('<path d="M330,100 v30" class="edge rel arr"/><text x="342" y="119" class="stag rel">관련 편집</text>')
    p.append('<path d="M220,70 h30" class="edge dec arr"/><text x="235" y="60" class="stag dec" text-anchor="middle">미끼 편집</text>')
    p.append('<path d="M220,160 h30" class="edge dec arr"/><text x="235" y="182" class="stag dec" text-anchor="middle">미끼 편집</text>')
    p.append('<rect x="440" y="40" width="150" height="60" rx="6" class="cell nod"/><text x="515" y="66" class="cellname" text-anchor="middle">A0 · B0</text><text x="515" y="86" class="celldesc" text-anchor="middle">미끼 없음</text>')
    p.append('<rect x="440" y="130" width="150" height="60" rx="6" class="cell ext"/><text x="515" y="156" class="cellname" text-anchor="middle">L · T · X</text><text x="515" y="176" class="celldesc" text-anchor="middle">사다리 · 등시각 · 문턱 확장</text>')
    p.append('<text x="60" y="225" class="snote">관련 편집 = 실제 미래를 바꾼다(정답이 바뀜) · 미끼 편집 = 같은 종류의 물체를 공 뒤에 두거나 키움(정답이 안 바뀜)</text>')
    return '<svg viewBox="0 0 620 240" class="schem" role="img" aria-label="2x2 design">' + "".join(p) + "</svg>"


def layers_diagram():
    p = []
    boxes = [("① 정확도", "사건을 맞히는가, 얼마나?", "동결 판독기 → 연속 사건 점수 s_e를 GT와 비교, 결정 변수 사다리에서 응답 함수 s_model(h) vs s_GT(h)"),
             ("② 의존", "무엇에 반응해서 그러는가?", "짝 편집 효과 Δ = s_e(편집 후) − s_e(편집 전), 같은 seed로 짝지어 95% CI. 관련 편집엔 반응, 미끼 편집엔 무반응이어야 함"),
             ("③ 위치", "입력의 어디를 쓰는가?", "단일 step x₀ 그래디언트 + 어텐션 라우팅을 영역(공·관련 구조·미끼·지지·배경)으로 집계, 정합 검사 4종, 사전 등록 H1–H3")]
    for i, (t, q, d) in enumerate(boxes):
        x = 10 + i * 205
        p.append(f'<rect x="{x}" y="10" width="195" height="150" rx="8" class="lbox l{i}"/>')
        p.append(f'<text x="{x + 14}" y="38" class="ltitle">{t}</text>')
        p.append(f'<text x="{x + 14}" y="60" class="lq">{q}</text>')
        # wrap description crudely
        words, lines, cur = d.split(" "), [], ""
        for wd in words:
            if len(cur) + len(wd) > 17:
                lines.append(cur); cur = wd
            else:
                cur = (cur + " " + wd).strip()
        lines.append(cur)
        for j, ln in enumerate(lines[:5]):
            p.append(f'<text x="{x + 14}" y="{82 + j * 16}" class="ldesc">{ln}</text>')
        if i < 2:
            p.append(f'<path d="M{x + 195},85 h10" class="edge arr"/>')
    return '<svg viewBox="0 0 630 170" class="schem" role="img" aria-label="three layers">' + "".join(p) + "</svg>"


# ------------------------------------------------------------------ page

CSS = """
:root{--ground:#F4F6F3;--surface:#FFFFFF;--surface2:#E9EDE9;--ink:#1E2A2B;--ink2:#41504F;--muted:#65736F;--line:#D3DAD5;
--model:#D9603B;--gt:#1F6E78;--dec:#8A7BAE;--rel:#2F8F5B;--warn:#B8860B;--warn-soft:#F6EFD6;--good-soft:#E3F1E8;--bad-soft:#F8E4DD;--focus:#1F6E78;
--vcond:#2F8F5B;--vseed:#D9603B;--vres:#C9D2CD}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#171C1D;--surface:#1F2628;--surface2:#283133;--ink:#E6ECEA;--ink2:#C2CCC9;--muted:#98A6A2;--line:#37423F;
--model:#F0876A;--gt:#5FB7C1;--dec:#B3A6D6;--rel:#67C08E;--warn:#E0B24A;--warn-soft:#3A3320;--good-soft:#1F3A2A;--bad-soft:#402823;--focus:#5FB7C1;--vcond:#67C08E;--vseed:#F0876A;--vres:#3E4A47}}
:root[data-theme="dark"]{--ground:#171C1D;--surface:#1F2628;--surface2:#283133;--ink:#E6ECEA;--ink2:#C2CCC9;--muted:#98A6A2;--line:#37423F;
--model:#F0876A;--gt:#5FB7C1;--dec:#B3A6D6;--rel:#67C08E;--warn:#E0B24A;--warn-soft:#3A3320;--good-soft:#1F3A2A;--bad-soft:#402823;--focus:#5FB7C1;--vcond:#67C08E;--vseed:#F0876A;--vres:#3E4A47}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans KR","Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;font-size:15.5px;line-height:1.75}
h1,h2,h3,h4{font-family:"Gowun Batang","Noto Serif KR","Nanum Myeongjo",serif;font-weight:700;line-height:1.3;margin:0;text-wrap:balance}
h1{font-size:2.3rem}h2{font-size:1.5rem}h3{font-size:1.15rem}h4{font-size:1rem;font-family:"IBM Plex Sans KR",sans-serif;font-weight:600}
p{margin:0}.prose p+p{margin-top:.8em}ul,ol{margin:0;padding-left:1.25em}li+li{margin-top:.35em}a{color:var(--focus)}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.9em}
.page{max-width:66rem;margin:0 auto;padding:2.2rem 1.4rem 5rem}.prose{max-width:46em}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header.top{display:grid;gap:.8rem;padding-bottom:1.4rem;border-bottom:1px solid var(--line)}.lede{font-size:1.08rem;color:var(--ink2);max-width:46em}
nav.toc{display:flex;flex-wrap:wrap;gap:.5rem .9rem;font-size:.86rem;margin-top:.4rem}nav.toc a{text-decoration:none;color:var(--ink2);border-bottom:1px dotted var(--line)}
section{padding-top:2.6rem}section>h2{display:flex;align-items:baseline;gap:.7rem;margin-bottom:1rem}section>h2 .num{font-family:"IBM Plex Mono",monospace;font-size:.8rem;color:var(--muted)}
.verdicts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1rem}
.verdict{border:1px solid var(--line);background:var(--surface);padding:1rem 1.1rem;display:grid;gap:.4rem;border-top:4px solid var(--model)}
.verdict .name{font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--muted)}.verdict .big{font-family:"Gowun Batang",serif;font-size:1.2rem;font-weight:700;line-height:1.35}
.verdict .sub{font-size:.9rem;color:var(--ink2)}
.answer{border:1px solid var(--line);border-left:5px solid var(--gt);background:var(--surface);padding:1.1rem 1.3rem;font-size:1.05rem;margin-top:1rem}
.scene{border:1px solid var(--line);background:var(--surface);padding:1.3rem 1.4rem;margin-top:1.4rem;display:grid;gap:1rem}
.scene .head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.6rem 1rem}.scene .head .fam{font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--muted)}
.two{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:1.4rem;align-items:start}@media(max-width:860px){.two,.verdicts{grid-template-columns:1fr}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.35rem .9rem;font-size:.92rem}.kv dt{color:var(--muted);white-space:nowrap}.kv dd{margin:0}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.72rem;padding:.05rem .45rem;border-radius:3px;border:1px solid var(--line);color:var(--ink2)}
.tag.rel{border-color:var(--rel);color:var(--rel)}.tag.dec{border-color:var(--dec);color:var(--dec)}.tag.model{border-color:var(--model);color:var(--model)}.tag.gt{border-color:var(--gt);color:var(--gt)}
.frames{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.5rem;align-items:start}@media(max-width:860px){.frames{grid-template-columns:repeat(3,minmax(0,1fr))}}
.frames figure{margin:0}.frames img{width:100%;height:auto;display:block;border:1px solid var(--line)}.frames figcaption{font-size:.78rem;color:var(--muted);margin-top:.3rem;line-height:1.4}
.frames figure.gt figcaption b{color:var(--gt)}.frames figure.model figcaption b{color:var(--model)}
.reading{border-left:3px solid var(--line);padding:.2rem 0 .2rem 1rem;font-size:.95rem}.reading b.v{color:var(--model)}
.callout{border:1px solid var(--line);background:var(--surface);padding:.9rem 1.1rem;font-size:.94rem}.callout.warn{border-color:var(--warn);background:var(--warn-soft)}.callout.key{border-left:5px solid var(--gt)}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}@media(max-width:860px){.grid3{grid-template-columns:1fr}}
.step{border:1px solid var(--line);background:var(--surface);padding:.9rem 1rem;display:grid;gap:.4rem}.step .n{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--muted)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);background:var(--surface)}table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;vertical-align:top;padding:.45rem .65rem;border-bottom:1px solid var(--line)}th{font-size:.78rem;color:var(--ink2);background:var(--surface2);white-space:nowrap}tr:last-child td{border-bottom:none}
td.num{text-align:right;font-family:"IBM Plex Mono",monospace;white-space:nowrap}td.ok{background:var(--good-soft)}td.bad{background:var(--bad-soft)}
figure.fig{margin:0;border:1px solid var(--line);background:var(--surface);padding:.6rem .6rem .4rem}figure.fig figcaption{font-size:.84rem;color:var(--muted);padding:.3rem .3rem 0;line-height:1.5}
svg.chart{width:100%;height:auto;display:block;font-family:"IBM Plex Sans KR",sans-serif}svg.schem{width:100%;height:auto;display:block;max-width:640px}
.chart .grid{stroke:var(--line);stroke-width:1}.chart .axis{stroke:var(--ink2);stroke-width:1}.chart .tick{fill:var(--muted);font-size:11px}.chart .label{fill:var(--ink2);font-size:12px}
.chart .note{fill:var(--muted);font-size:11px}.chart .legend{fill:var(--ink2);font-size:12px}.chart .ref{stroke:var(--muted);stroke-dasharray:4 4}.chart .zero{stroke:var(--ink2);stroke-dasharray:3 3}
.chart .model.dot{fill:var(--model)}.chart .model{stroke:var(--model);stroke-width:2.2}.chart .model.err{stroke-width:1.4;opacity:.7}
.chart .gt.dot{fill:var(--gt)}.chart .gt{stroke:var(--gt);stroke-width:2.2}.chart .dec.dot{fill:var(--dec)}.chart .dec{stroke:var(--dec);stroke-width:2.2}
.chart .rel.dot{fill:var(--rel)}.chart .rel{stroke:var(--rel);stroke-width:2.2}.chart .ci{stroke-width:3;stroke-linecap:round}.chart .rowname{fill:var(--ink);font-size:12px}.chart .rowname.rel{fill:var(--rel)}.chart .rowname.dec{fill:var(--dec)}
.chart .gtmark{fill:var(--gt)}.chart .vcond{fill:var(--vcond)}.chart .vseed{fill:var(--vseed)}.chart .vres{fill:var(--vres)}.chart .barlabel{fill:#fff;font-size:11.5px;font-weight:600}
:root[data-theme="dark"] .chart .barlabel,:root:not([data-theme="light"]) .chart .barlabel{fill:#101414}
.schem .sground{stroke:var(--ink2);stroke-width:2}.schem .sball{fill:var(--model)}.schem .sarrow{stroke:var(--ink);stroke-width:2;fill:none}
.schem .srel{stroke:var(--rel);stroke-width:3;fill:var(--good-soft)}.schem .sdec{stroke:var(--dec);stroke-width:3;fill:none;stroke-dasharray:5 4}
.schem .ball2.srel{fill:var(--rel);stroke:none}.schem .ball2.sdec{fill:none;stroke:var(--dec);stroke-width:2.5;stroke-dasharray:4 3}
.schem .splate{fill:var(--surface2);stroke:var(--ink2)}.schem .thick{stroke-width:6}.schem .sfall{stroke:var(--gt);stroke-width:2;fill:none;stroke-dasharray:4 3}
.schem .stag{font-size:12px;fill:var(--ink2)}.schem .stag.rel{fill:var(--rel)}.schem .stag.dec{fill:var(--dec)}.schem .snote{font-size:12px;fill:var(--muted)}
.schem .cell{fill:var(--surface);stroke:var(--line)}.schem .cell.nod{stroke-dasharray:4 3}.schem .cell.ext{fill:var(--surface2)}.schem .cellname{font-size:15px;font-weight:700;fill:var(--ink)}.schem .celldesc{font-size:11.5px;fill:var(--muted)}
.schem .edge{fill:none;stroke:var(--ink2);stroke-width:2}.schem .edge.rel{stroke:var(--rel)}.schem .edge.dec{stroke:var(--dec)}.schem .arr{marker-end:url(#arrow)}
.schem .lbox{fill:var(--surface);stroke:var(--line)}.schem .lbox.l0{stroke:var(--gt)}.schem .lbox.l1{stroke:var(--rel)}.schem .lbox.l2{stroke:var(--dec)}
.schem .ltitle{font-size:15px;font-weight:700;fill:var(--ink)}.schem .lq{font-size:12.5px;fill:var(--ink2);font-weight:600}.schem .ldesc{font-size:11px;fill:var(--muted)}
.small{font-size:.86rem;color:var(--muted)}footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.84rem;color:var(--muted)}
.pill{display:inline-block;padding:.1rem .6rem;border-radius:999px;font-size:.8rem;font-weight:600}.pill.partial{background:var(--warn-soft);color:var(--warn)}.pill.none{background:var(--bad-soft);color:var(--model)}.pill.yes{background:var(--good-soft);color:var(--rel)}
@media (prefers-reduced-motion: no-preference){a:focus-visible,button:focus-visible{outline:2px solid var(--focus);outline-offset:2px}}
"""

DEFS = '<svg width="0" height="0" style="position:absolute"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs></svg>'


def frame_row(fr, scene_label, gt_words, model_words):
    return (
        '<div class="frames">'
        f'<figure><img src="{fr["cond"]}" alt="조건 마지막 프레임"><figcaption><b>모델 입력</b> · 조건 5프레임의 마지막(t = 0.25 s). 사건 전.</figcaption></figure>'
        f'<figure class="gt"><img src="{fr["gt_ev"]}" alt="GT 사건 프레임"><figcaption><b>실제(MuJoCo)</b> 프레임 {fr["ev"]}: {gt_words[0]}</figcaption></figure>'
        f'<figure class="model"><img src="{fr["model_ev"]}" alt="모델 사건 프레임"><figcaption><b>Cosmos 예측</b> 프레임 {fr["ev"]} (seed {fr["seed"]}): {model_words[0]}</figcaption></figure>'
        f'<figure class="gt"><img src="{fr["gt_last"]}" alt="GT 마지막 프레임"><figcaption><b>실제</b> 프레임 {fr["last"]}: {gt_words[1]}</figcaption></figure>'
        f'<figure class="model"><img src="{fr["model_last"]}" alt="모델 마지막 프레임"><figcaption><b>Cosmos 예측</b> 프레임 {fr["last"]}: {model_words[1]}</figcaption></figure>'
        "</div>")


def build(root: Path, out: Path):
    hill, col, edge = (load(root / s / "analysis" / "summary.json") for s in ("hill", "collide", "edge"))
    H, C, E = conds(hill), conds(col), conds(edge)
    eh, ec = effects(hill), effects(col)
    spd = load(root / "edge" / "analysis" / "speed_diag.json")
    spe = load(root / "edge" / "analysis" / "speed_effects.json")
    vd = load(root / "variance_decomp.json")
    att = {s: load(root / s / "attrib" / "summary.json") for s in ("hill", "collide", "edge")}

    # ---------------- charts
    # Hill: s_e vs h (cm)
    ch = Chart(640, 340, xlim=(0, 30), ylim=(-4, 11))
    ch.axes([0, 5, 10, 15, 20, 25, 30], [-2, 0, 2, 4, 6, 8, 10], "경로 위 언덕 높이 h (cm)", "넘김 정도 s_e (공 지름 단위)")
    ch.hline(0, "ref", "0 = 마루에서 멈춤 · 위 = 넘음 · 아래 = 되돌아옴")
    order = ["XF", "A", "L5", "L4", "L3", "L2", "L1", "B", "X20", "X28"]
    pts_gt = [(H[k]["x"] * 100, H[k]["s_e_gt"]) for k in order]
    pts_m = [(H[k]["x"] * 100, H[k]["s_e_mean"]) for k in order]
    se = [H[k]["s_e_sd"] / math.sqrt(12) for k in order]
    ch.line(pts_gt, "gt"); ch.points(pts_gt, "gt", r=4)
    ch.line(pts_m, "model"); ch.points(pts_m, "model", r=5, err=se)
    ch.text(ch.X(30) - 2, ch.Y(H["XW"]["s_e_mean"]) - 4, f"수직 벽(XW): 모델 {H['XW']['s_e_mean']:.1f} / 실제 {H['XW']['s_e_gt']:.1f}", anchor="end")
    ch.legend([("gt", "실제(MuJoCo)"), ("model", "Cosmos 평균 ± SE (seed 12)")], x=ch.X(13), y=ch.Y(10.2))
    hill_chart = ch.svg("hill response")

    # Collide: s_e vs S (log)
    cc = Chart(640, 340, xlim=(0.1, 2.6), ylim=(-1.1, 1.15), xlog=True)
    cc.axes([0.125, 0.25, 0.5, 1, 2], [-1, -0.5, 0, 0.5, 1], "질량비 S = (r_표적 / r_공)³  (로그 눈금)", "접촉 후 / 접촉 전 속도비")
    cc.hline(0, "ref", "0 = 멈춤 · 아래 = 되튐 · 위 = 계속")
    Ss = np.exp(np.linspace(math.log(0.1), math.log(2.6), 60))
    cc.line([(s, (1 - s) / (1 + s)) for s in Ss], "gt")
    ordc = ["XL", "A", "L1", "L2", "L3", "L4", "L5", "B"]
    pm = [(C[k]["S"], C[k]["s_e_mean"]) for k in ordc]
    cc.points(pm, "model", r=5, err=[C[k]["s_e_sd"] / math.sqrt(12) for k in ordc])
    cc.line(pm, "model")
    cc.text(cc.X(2.6) - 2, cc.Y(-0.98) + 14, f"수직 벽(XW): 모델 {C['XW']['s_e_mean']:.2f} / 실제 {C['XW']['s_e_gt']:.2f}", anchor="end")
    cc.legend([("gt", "실제: (1−S)/(1+S)"), ("model", "Cosmos 평균 ± SE (seed 12)")], x=cc.X(0.7), y=cc.Y(1.05))
    col_chart = cc.svg("collide response")

    # Edge: approach speed ratio vs distance
    ce = Chart(640, 340, xlim=(0, 10.5), ylim=(0, 1.2))
    ce.axes([0, 2, 4, 6, 8, 10], [0, 0.25, 0.5, 0.75, 1.0], "조건 종료 시점의 공–모서리 거리 (공 지름 단위)", "접근 속도비 (모델 / 실제, 프레임 5–10)")
    ce.hline(1.0, "ref", "1.0 = 실제와 같은 속도")
    rows = []
    for k, d in spd.items():
        rows.append((d["dist_edge_dia"], float(np.mean([r["v_ratio"] for r in d["rows"]])), k))
    rows.sort()
    ce.points([(x, y) for x, y, _ in rows], "model", r=5, labels=[(k if k in ("A", "B", "N1", "N2", "XJ", "T1", "T6", "XS") else "") for _, _, k in rows])
    ce.text(ce.X(0.3), ce.Y(0.08), f"조건 수준 Spearman ρ = {spe['spearman']['cond'][0]:.2f} (n = 18) · 롤아웃 수준 {spe['spearman']['rollout'][0]:.2f} (n = 216)")
    ce.legend([("model", "조건별 Cosmos 평균 (seed 12)")], x=ce.X(5.5), y=ce.Y(0.3))
    edge_chart = ce.svg("edge approach speed")

    # Edge fall-rate bars by t_edge group
    groups = [("0.6 s", ["A", "C", "A0", "T1", "T4"]), ("0.75 s", ["T2", "T5", "T7"]), ("1.05 s", ["B", "D", "B0", "T3", "T6"]), ("유지(1.4 s)", ["N1", "N2"]), ("이음매 XJ", ["XJ"])]
    cb = Chart(640, 250, ml=58, xlim=(0, len(groups)), ylim=(0, 1.05))
    cb.axes([], [0, 0.25, 0.5, 0.75, 1.0], "모서리 도달 시각 t_edge (실제) — 왼쪽 세 묶음은 실제로 반드시 떨어짐, 오른쪽 둘은 안 떨어짐", "낙하 확률 (21 프레임 안)")
    for i, (name, ks) in enumerate(groups):
        m = float(np.mean([E[k]["p_event"] for k in ks]))
        gt = 1.0 if i < 3 else 0.0
        x0, x1 = cb.X(i + 0.18), cb.X(i + 0.5)
        cb.parts.append(f'<rect x="{x0:.1f}" y="{cb.Y(gt):.1f}" width="{x1 - x0:.1f}" height="{cb.Y(0) - cb.Y(gt):.1f}" class="gt" style="fill:var(--gt);opacity:.35;stroke:none"/>')
        x0, x1 = cb.X(i + 0.5), cb.X(i + 0.82)
        cb.parts.append(f'<rect x="{x0:.1f}" y="{cb.Y(m):.1f}" width="{x1 - x0:.1f}" height="{cb.Y(0) - cb.Y(m):.1f}" style="fill:var(--model)"/>')
        cb.text(cb.X(i + 0.5), cb.Y(0) + 18, name, "tick", "middle")
        cb.text(cb.X(i + 0.66), cb.Y(m) - 5, f"{m:.2f}", "note", "middle")
    cb.legend([("gt", "실제"), ("model", "Cosmos (seed 12 × 셀)")], x=cb.X(3.1), y=cb.Y(0.95))
    edge_fall = cb.svg("edge fall rate")

    # Effects forests
    def row(e, name, kind, gt=None):
        return (name, kind, e["delta_mean"], e["ci_lo"], e["ci_hi"], e["delta_gt"] if gt is None else gt)
    hill_forest = forest([row(eh["relevant A->B"], "관련 A→B (h 3.6→14 cm)", "rel"), row(eh["relevant C->D"], "관련 C→D", "rel"),
                          row(eh["decoy A->C"], "미끼 A→C (뒤 언덕 ↑)", "dec"), row(eh["decoy B->D"], "미끼 B→D", "dec"),
                          row(eh["decoy off A0->A"], "미끼 없음→있음 A0→A", "dec"), row(eh["decoy off B0->B"], "미끼 없음→있음 B0→B", "dec")],
                         (-7, 3), "Δ 넘김 정도 (지름) · ◆ = 실제 효과", unit_note="")
    col_forest = forest([row(ec["relevant A->B"], "관련 A→B (S 0.5→2)", "rel"), row(ec["relevant C->D"], "관련 C→D", "rel"),
                         row(ec["decoy A->C"], "미끼 A→C (뒤 공 ↑)", "dec"), row(ec["decoy B->D"], "미끼 B→D", "dec"),
                         row(ec["decoy off A0->A"], "미끼 없음→있음 A0→A", "dec"), row(ec["decoy off B0->B"], "미끼 없음→있음 B0→B", "dec")],
                        (-0.8, 0.5), "Δ 속도비 · ◆ = 실제 효과")
    def srow(lab, name, kind, gt):
        r = spe[lab]
        return (name, kind, r["delta"], r["ci"][0], r["ci"][1], gt)
    edge_forest = forest([srow("relevant A->B", "관련 A→B (모서리 멀리)", "rel", 0.0), srow("relevant C->D", "관련 C→D", "rel", 0.0),
                          srow("decoy A->C", "미끼 A→C (왼쪽 끝 가까이)", "dec", 0.0), srow("decoy B->D", "미끼 B→D", "dec", 0.0),
                          srow("decoy off A0->A", "미끼 없음→있음 A0→A", "dec", 0.0), srow("joint vs hold XJ-N1", "이음매 XJ − 유지 N1", "dec", 0.0)],
                         (-0.5, 0.9), "Δ 접근 속도비 (사후 지표) · ◆ = 실제(실제 속도는 모서리와 무관 = 0)")

    var_chart = stacked([("Hill", vd["hill"]["frac_cond"], vd["hill"]["frac_seed"], vd["hill"]["frac_resid"]),
                         ("Collide", vd["collide_ball"]["frac_cond"], vd["collide_ball"]["frac_seed"], vd["collide_ball"]["frac_resid"]),
                         ("Edge", vd["edge"]["frac_cond"], vd["edge"]["frac_seed"], vd["edge"]["frac_resid"])])

    # attribution table rows
    def att_rows(scene):
        s = att[scene]
        out = []
        for r in s["table"]:
            gd, at = r["grad_density"], r["attention_ball_late_density"]
            rg = gd["structure"] / gd["decoy"] if gd["decoy"] and not math.isnan(gd.get("structure", float("nan"))) else float("nan")
            ra = at["structure"] / at["decoy"] if at["decoy"] and not math.isnan(at.get("structure", float("nan"))) else float("nan")
            out.append((r["cond"], gd["structure"], gd["decoy"], gd["background"], rg, at["structure"], at["decoy"], at["ball"], ra))
        return out

    # ---------------- frames
    fr_h = frames_for(root, "hill", "B", 2)
    fr_c = frames_for(root, "collide", "B", 2)
    fr_e = frames_for(root, "edge", "A", 3)

    # ---------------- HTML
    P = []
    P.append('<title>PVR PoC 해설서</title>')
    P.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
    P.append(f"<style>{CSS}</style>{DEFS}<div class=\"page\">")
    P.append('<header class="top"><div class="eyebrow">SimDROID · PhysicsGen 트랙 · 예측적 시각 정보 사용(PVR) PoC · 2026-09-07 KST</div>'
             '<h1>세계 모델은 사건이 일어나기 전에, 그 사건을 결정할 것을 보고 있는가</h1>'
             '<p class="lede">굴러가는 공 앞에 언덕·다른 공·판의 모서리가 있다. 사건이 일어나기 0.25 초 전 장면 5장을 주고 Cosmos-Predict2-2B에 다음 1 초를 그리게 했다. '
             '이 페이지는 <b>어떤 장면을 왜 만들었는지</b>, <b>무엇을 어떻게 쟀는지</b>, <b>무엇이 나왔고 무엇을 뜻하는지</b>를 순서대로 설명한다. 수치 전체는 결과 보고 페이지에, 설계 근거는 DESIGN 문서에 있다.</p>'
             '<nav class="toc"><a href="#glance">00 한눈에</a><a href="#scenes">01 씬 구성과 목적</a><a href="#method">02 검증 방식과 목적</a><a href="#results">03 결과</a><a href="#impl">04 시사점</a></nav></header>')

    # 00 glance
    P.append('<section id="glance"><h2><span class="num">00</span>한눈에</h2><div class="prose">'
             '<p><b>질문.</b> 물리 법칙을 "알고 있느냐"가 아니라, 미래를 그릴 때 <b>결정적인 구조를 골라서, 정확한 크기로</b> 쓰느냐를 묻는다. '
             '그래서 실제 미래를 바꾸는 편집(관련)과 바꾸지 않는 편집(미끼, 같은 종류의 물체를 공 <i>뒤</i>에)을 짝지어 넣고, 모델의 예측이 어느 쪽에 반응하는지를 봤다.</p></div>'
             '<div class="answer"><b>답.</b> 모델은 결정 구조를 <b>골라서 읽는다</b> — 미끼에는 어느 씬에서도 반응하지 않고, 판의 이음매와 진짜 모서리를 구별한다. '
             '그러나 <b>정확하게 쓰지는 못한다</b> — 반응 크기는 실제의 1/3 이하이고 사건의 방향이 틀린다(넘어야 할 언덕은 넘지만 되돌아와야 할 언덕도 넘고, 되튀어야 할 공이 계속 가고, 떨어져야 할 공이 모서리 앞에서 멈춘다). '
             '게다가 결과의 대부분은 장면이 아니라 잡음 seed가 정한다.</div>'
             '<div class="verdicts">'
             f'<div class="verdict"><div class="name">Predictive Hill · 기하</div><div class="big">언덕을 읽지만 에너지 문턱을 모른다</div><div class="sub">넘김 정도가 언덕 높이에 단조로 반응(실제의 20–33%)하되 14–28 cm 언덕도, 강철 벽도 대부분 넘는다.</div><span class="pill partial">부분 반응</span></div>'
             f'<div class="verdict"><div class="name">Predictive Collide · 관계</div><div class="big">접촉은 제때 그리지만 질량비를 모른다</div><div class="sub">표적이 8배 가볍든 2배 무겁든 접촉 후 속도는 ≈ 75%. 되튐은 seed가 정한다. 벽에서만 부분 반응.</div><span class="pill none">무반응</span></div>'
             f'<div class="verdict"><div class="name">Predictive Edge · 경계+속도</div><div class="big">모서리를 읽지만 떨어뜨리지 않는다</div><div class="sub">모서리에 가까울수록 공을 감속시켜 앞에서 멈춘다(거리와 ρ 0.91). 이음매는 무시. 낙하는 0–36%.</div><span class="pill partial">부분 반응 · 방향 오류</span></div>'
             '</div></section>')

    # 01 scenes
    P.append('<section id="scenes"><h2><span class="num">01</span>씬 구성과 목적</h2><div class="two"><div class="prose">'
             '<p><b>왜 세 씬인가.</b> 예측에 필요한 정보의 종류가 다르다. Hill은 경로 위 지형의 <b>기하</b>(언덕 높이 vs 공의 운동 에너지), Collide는 두 물체의 <b>관계</b>(표적 공의 크기 = 질량비), Edge는 <b>경계 위치와 속도의 결합</b>(모서리까지의 거리 ÷ 속도 = 낙하 시각). 실패가 정보 종류에 따라 다른 자리에서 나는지 보려는 것이다.</p>'
             '<p><b>공통 틀.</b> 모든 씬은 같은 2×2 설계를 쓴다. 관련 편집은 결정 구조 하나만 바꿔 실제 미래를 바꾼다. 미끼 편집은 <i>같은 종류</i>의 물체를 공이 멀어지는 쪽(뒤)에 두거나 키운다 — 시뮬레이션에서 미래가 바뀌지 않음을 확인한 셀만 쓴다. 그 위에 결정 변수 사다리(응답 함수용)와 문턱 확장(반응이 시작되는 지점 탐색용)을 얹는다.</p>'
             '<p><b>사건 전에 끊는다.</b> 모델이 받는 조건 5장은 0–0.25 s로, 사건(언덕 마루·접촉·낙하)은 그보다 0.2–1 s 뒤다. 이미 일어난 일을 이어 그리는 것으로는 맞출 수 없고, 보이는 구조에서 <i>예측</i>해야 한다.</p>'
             '<p class="small">물리는 Phase A/B의 검증된 MuJoCo 클래스를 상속하고 카메라(fovy 26)·미끼·확장만 더했다. 검증: 기존 클래스와 궤적 동일(collide·edge 오차 0, hill 정렬 후 ≤ 1.3 cm), 물리 게이트 48/48, 세그멘테이션 마스크 IoU ≥ 0.936, 판독기 GT 자가검증 48/48.</p>'
             '</div><div>' + design_grid() + '</div></div>')

    def scene_card(name, fam, schem, reads, edits, ext, fr, gt_words, model_words, purpose):
        return ('<div class="scene"><div class="head"><h3>' + name + '</h3><span class="fam">' + fam + '</span></div>'
                '<div class="two"><div>' + schem + '</div><dl class="kv">'
                '<dt>모델이 읽어야 하는 것</dt><dd>' + reads + '</dd>'
                '<dt><span class="tag rel">관련 편집</span></dt><dd>' + edits[0] + '</dd>'
                '<dt><span class="tag dec">미끼 편집</span></dt><dd>' + edits[1] + '</dd>'
                '<dt>사다리·확장</dt><dd>' + ext + '</dd>'
                '<dt>목적</dt><dd>' + purpose + '</dd></dl></div>'
                + frame_row(fr, name, gt_words, model_words) + '</div>')

    P.append(scene_card("Predictive Hill", "hill_roll_pvr · 조건 15 × seed 12", schematic("hill"),
                        "공의 속도(1.0 m/s, 조건 프레임에서 보임)와 경로 위 언덕의 높이. 넘는 조건은 ½v₀² > g·h (마찰 포함 S = 0.7 v₀²/(g h) > 1).",
                        ("언덕 높이 h 3.6 cm(넘음) → 14.3 cm(되돌아옴). A→B, C→D", "공 뒤의 언덕 h_d 3.6 → 14.3 cm. A→C, B→D. A0·B0는 뒤 언덕 없음"),
                        "높이 사다리 L1–L5(11.3 → 4.5 cm), 확장: 20·28 cm 언덕, 수직 강철 벽(XW), 평지(XF)",
                        fr_h, ("14 cm 언덕: 마루 못 넘고 되돌아오는 중", "출발 쪽으로 돌아와 있음"), ("공이 마루를 넘어가는 중", "언덕 너머로 사라짐"),
                        "Phase A의 \"언덕 무반응\"이 관측 실패인지 동역학 실패인지 가르고, 반응이 시작되는 높이를 찾는다."))
    P.append(scene_card("Predictive Collide", "two_ball_pvr · 조건 15 × seed 12", schematic("collide"),
                        "다가오는 공과 표적 공의 크기 비(같은 재질 → 질량비 S = (r₂/r₁)³). 정면 탄성 충돌 후 속도비 = (1−S)/(1+S): 가벼운 표적이면 계속, 무거우면 되튐.",
                        ("표적 반지름 3.2 cm(S 0.5, 계속) → 5.0 cm(S 2, 되튐). A→B, C→D", "공 뒤의 공 반지름 3.2 → 5.0 cm. A→C, B→D. A0·B0는 뒤 공 없음"),
                        "질량비 사다리 L1–L5(S 0.63 → 1.59), 확장: 강철 벽(XW, S→∞), 아주 가벼운 표적(XL, S 0.125), 빗나감(XM), 표적 없음(XF)",
                        fr_c, ("무거운 표적(S 2): 공이 되튀어 왼쪽으로", "공은 왼쪽, 표적은 오른쪽으로 갈라짐"), ("접촉은 그렸으나 공이 계속 오른쪽", "공이 표적을 밀고 오른쪽으로 나감"),
                        "\"반전 무작위\"가 표적을 못 보는 것인지 운동량 교환을 못 하는 것인지 가르고, 반응이 시작되는 질량비를 찾는다."))
    P.append(scene_card("Predictive Edge / Fall", "support_edge_pvr · 조건 18 × seed 12", schematic("edge"),
                        "공의 속도와 모서리까지의 거리. 낙하 시각 t_edge = 거리 ÷ v₀ 이므로 (거리, 속도) 조합이 같은 시각을 줄 수 있다.",
                        ("모서리 위치: t_edge 0.6 s(프레임 12 낙하) → 1.05 s(프레임 19). A→B, C→D", "판의 보이는 왼쪽 끝을 공 뒤 멀리(−1.08 m) → 가까이(−0.65 m). A0·B0는 왼쪽 끝이 화면 밖"),
                        "등시각 삼중항 T1–T7((v₀, t_edge) 격자: 같은 낙하 시각을 다른 거리·속도로), 유지 셀 N1·N2(모서리가 지평 너머), 확장: 판 이음매 선(XJ, 가짜 모서리), 5 cm 단차(XS), 3° 경사(XT)",
                        fr_e, ("모서리를 넘어 떨어지기 시작", "판 아래로 떨어져 있음"), ("모서리 앞에서 느려져 멈춤", "여전히 판 위, 모서리 근처"),
                        "\"부양\" 실패가 모서리를 못 보는 것인지 낙하 동역학을 못 그리는 것인지 가르고, 모델이 시각(거리÷속도)을 읽는지 거리만 읽는지 본다."))
    P.append("</section>")

    # 02 method
    P.append('<section id="method"><h2><span class="num">02</span>모델 검증 방식과 목적</h2>'
             '<div class="prose"><p><b>모델 계약.</b> Cosmos-Predict2-2B Video2World(480p·16 fps)에 조건 5프레임과 씬당 하나의 중립 프롬프트를 주고 35 step, guidance 7.0으로 21프레임(조건 5 + 예측 16)을 생성한다. '
             '씬당 조건 15·15·18 × seed 12 = 576 롤아웃. <b>모든 조건에 같은 12개 seed</b>를 쓴다(common random numbers) — 조건 간 차이가 잡음이 아니라 편집 때문임을 seed별 짝으로 확인하기 위해서다.</p></div>'
             '<div style="margin:1rem 0">' + layers_diagram() + '</div>'
             '<div class="grid3">'
             '<div class="step"><div class="n">① 정확도 — "맞히는가, 얼마나?"</div><p>이진 결과(넘음/되튐/낙하)만 보면 "무반응"과 "약한 반응"이 구별되지 않는다. 그래서 동결 판독기(색 마스크 중심 궤적)로 <b>연속 사건 점수</b>를 읽는다: Hill은 넘김 정도(마루 기준 최대 진행, 지름 단위), Collide는 접촉 후/전 속도비, Edge는 낙하 시작 프레임(21에서 검열). 결정 변수 사다리에서 모델 응답 함수 s_model(h)를 실제 s_GT(h)와 겹쳐 본다. 실제 클립에 같은 판독기를 돌려 GT 자가검증(48/48)을 통과시켰다.</p></div>'
             '<div class="step"><div class="n">② 의존 — "무엇에 반응해서?"</div><p>짝 편집 효과 Δ = mean_seed[s_e(편집 후) − s_e(편집 전)]와 95% 부트스트랩 CI. 관련 편집의 Δ는 실제 효과 Δ_GT와 같은 방향으로 커야 하고, 미끼 편집의 Δ는 0이어야 한다. 정규화 Δ/Δ_GT가 "얼마나 정확히"의 척도, Relevance Index = |Δ관련| − |Δ미끼|(seed별 짝)가 "얼마나 선택적으로"의 척도다. 이것이 관측 실패(미끼처럼 무반응)와 동역학 실패(반응하되 크기·방향이 틀림)를 가른다.</p></div>'
             '<div class="step"><div class="n">③ 위치 — "입력의 어디를?"</div><p>파이프라인의 샘플링을 같은 seed로 재현하고 step 20/24/28에서 한 번의 x₀ 예측을 잠재 볼 probe 점수(공 위치의 미분 가능 대리)로 미분해 조건 5프레임에 대한 |∂s/∂x|를 얻는다. 어텐션 라우팅은 예측 프레임의 공 토큰이 조건 토큰 어디를 읽는지를 28블록에서 재계산한다. 둘 다 세그멘테이션 영역(공·관련 구조·미끼·지지·배경)으로 집계하고 면적 대비 밀도(1 = 균일)로 본다. 정합 검사(무관 스칼라 지도 상관·블록 무작위화·seed 변동·가장자리 기준선)와 사전 등록 가설 H1(관련 &gt; 미끼)·H2(구조 없는 셀에서 감소)·H3(② 순서와 일치) 중 하나라도 깨지면 열지도는 "선별 신호"로 강등한다.</p></div>'
             '</div>'
             '<div class="callout" style="margin-top:1rem"><b>왜 이렇게 재는가.</b> 기존 결과("언덕 무반응", "반전 무작위", "부양")는 모두 이진 사건 판정이었고, 관측 실패와 동역학 실패를 가를 수 없었다. 연속 점수 + 관련/미끼 짝 편집 + 응답 함수는 "봤는데 잘못 쓴 것"과 "안 본 것"을 분리하고, 세 씬을 같은 표에 놓아 정보 종류별로 실패 위치가 같은지 비교하게 한다. 열지도는 그 의존이 입력 위치로 국소화되는지를 확인하는 보조 수단이다.</div>'
             '</section>')

    # 03 results
    def att_table(scene, title):
        rows = att_rows(scene)
        tr = []
        for cond, gs, gd, gb, rg, as_, ad, ab, ra in rows:
            def cell(v, good):
                if math.isnan(v):
                    return '<td class="num">–</td>'
                return f'<td class="num {"ok" if good(v) else "bad"}">{v:.2f}</td>'
            def n(v, d=2):
                return '–' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{d}f}'
            tr.append(f'<tr><td class="mono">{cond}</td>' + f'<td class="num">{n(gs)}</td><td class="num">{n(gd)}</td><td class="num">{n(gb)}</td>' + cell(rg, lambda v: v > 1.2)
                      + f'<td class="num">{n(as_)}</td><td class="num">{n(ad)}</td><td class="num">{n(ab, 1)}</td>' + cell(ra, lambda v: v > 1.2) + '</tr>')
        s = att[scene]
        rk = [(k, v["corr_with_original"]) for d in s["randomization"] for k, v in d.items()][:4]
        return (f'<h4 style="margin-top:.8rem">{title}</h4><div class="tablewrap"><table><thead><tr><th>셀</th><th colspan="4">그래디언트 밀도 (1 = 균일)</th><th colspan="4">어텐션 밀도 (예측 공 토큰 → 조건 토큰, 후기 블록)</th></tr>'
                '<tr><th></th><th>관련 구조</th><th>미끼</th><th>배경</th><th>관련/미끼</th><th>관련 구조</th><th>미끼</th><th>공(자기 과거)</th><th>관련/미끼</th></tr></thead><tbody>'
                + "".join(tr) + '</tbody></table></div>'
                f'<p class="small">무관 스칼라 지도와의 상관 {s["wrong_target_corr_mean"]:.2f} · 마지막 k 블록 무작위화 후 지도 상관 ' + ", ".join(f"k={k}: {v:.2f}" for k, v in rk) + ' · 초록 = 관련이 미끼보다 1.2배 이상.</p>')

    P.append('<section id="results"><h2><span class="num">03</span>결과</h2>'
             '<div class="prose"><p>씬마다 핵심 그림 하나(응답 함수)와 짝 효과 그림 하나를 보인다. <span class="tag gt">실제</span>는 MuJoCo, <span class="tag model">Cosmos</span>는 seed 12개 평균이다. 이어서 세 씬을 가로지르는 두 결과(분산의 주인, 열지도)를 놓는다.</p></div>')

    # Hill results
    P.append('<div class="scene"><div class="head"><h3>Hill — 언덕 높이를 읽되 문턱을 모른다</h3><span class="pill partial">부분 반응 · 실제의 20–33%</span></div>'
             '<div class="two"><figure class="fig">' + hill_chart + '<figcaption>응답 함수. 실제는 7 cm 부근에서 "넘음(+9)"에서 "되돌아옴(−1 ~ −2)"으로 꺾이지만, 모델은 모든 높이에서 마루를 넘는다(양수). 다만 넘김 정도가 높이에 따라 5.5 → 2.9로 단조 감소한다 — 언덕이 높을수록 덜 나아간다.</figcaption></figure>'
             '<div><figure class="fig">' + hill_forest + '<figcaption>짝 편집 효과(seed 12 짝, 95% CI). 관련 편집(초록)은 CI가 0을 벗어나 실제와 같은 방향, 미끼 편집(보라)은 모두 0을 포함.</figcaption></figure>'
             f'<div class="reading" style="margin-top:.8rem">읽기: P(넘음)는 A 1.00 / B 0.88 / 28 cm 0.67 / <b>강철 벽 0.91</b>로 이진으로는 무반응. 연속 점수는 관련 A→B <b class="v">{eh["relevant A->B"]["delta_mean"]:+.2f}</b> [{eh["relevant A->B"]["ci_lo"]:+.2f}, {eh["relevant A->B"]["ci_hi"]:+.2f}], C→D <b class="v">{eh["relevant C->D"]["delta_mean"]:+.2f}</b>(실제 −10.8, 정규화 0.20·0.33), 미끼 +0.68·−0.74(0 포함), Relevance Index 1.94 [0.34, 3.80]·2.43 [0.49, 4.55]. → <b>관측은 하나 에너지 문턱 오예측</b>.</div></div></div></div>')

    # Collide results
    P.append('<div class="scene"><div class="head"><h3>Collide — 접촉은 제때, 질량비는 무시</h3><span class="pill none">무반응 · 정규화 0.01</span></div>'
             '<div class="two"><figure class="fig">' + col_chart + '<figcaption>응답 함수. 실제 속도비는 S에 따라 +0.78(가벼운 표적, 계속)에서 −0.33(무거운 표적, 되튐)으로 내려가지만, 모델은 S와 무관하게 0.66–0.85로 평평하다. 강철 벽에서만 0.18로 내려온다.</figcaption></figure>'
             '<div><figure class="fig">' + col_forest + '<figcaption>짝 편집 효과. 관련 편집조차 0 근처(실제 −0.67). 미끼도 0.</figcaption></figure>'
             f'<div class="reading" style="margin-top:.8rem">읽기: 접촉·감속이 그려지는 시각은 7.3–8.0 프레임으로 실제(7)와 맞는다 — 표적을 <i>본다</i>. 그러나 접촉 후 속도는 표적이 8배 가볍든 2배 무겁든 ≈ 75%이고 되튐 확률은 0–0.125(실제: S&gt;1이면 1). 벽(XW)만 되튐 0.2·속도비 0.18. RI 0.15 [0.03, 0.31]은 평균 효과 0에서 seed 산포 차로 생긴 값이라 의존의 증거로 읽지 않는다. → <b>접촉 관측은 하나 운동량 교환 오예측</b>, 반응 문턱은 "무거운 공"과 "벽" 사이.</div></div></div></div>')

    # Edge results
    P.append('<div class="scene"><div class="head"><h3>Edge — 모서리를 읽되 낙하가 아니라 제동</h3><span class="pill partial">부분 반응 · 방향 오류</span></div>'
             '<div class="two"><figure class="fig">' + edge_fall + '<figcaption>사전 등록 지표(낙하 시각)의 결과. 실제로 반드시 떨어지는 셀에서 모델의 낙하율은 0–0.36이고, 가까운 모서리일수록 조금 더 떨어뜨린다. 유지 셀과 이음매(가짜 모서리)에서의 거짓 낙하는 0.08. 낙하 셀 216개 중 151개가 21프레임까지 안 떨어져 "검열"됐다.</figcaption></figure>'
             '<figure class="fig">' + edge_chart + '<figcaption>사후 진단(사전 등록 아님). 조건 종료 시점의 모서리까지 거리가 가까울수록 모델의 공이 느리다: 1.5 지름에서 실제 속도의 33%, 9.5 지름에서 108%. 이음매 XJ(6.7 지름, 판은 이어짐)는 유지 셀 N1과 같다.</figcaption></figure></div>'
             '<div class="two" style="margin-top:1rem"><figure class="fig">' + edge_forest + '<figcaption>접근 속도비의 짝 효과. 관련 편집(모서리를 멀리)은 CI가 0 위, 미끼 편집은 0 포함, 이음매−유지 차이는 0.01. 실제 속도는 모서리와 무관하므로 실제 효과는 모두 0 — 모델의 감속 자체가 오류이되, <i>모서리에 선택적인</i> 오류다.</figcaption></figure>'
             f'<div class="reading">읽기: 모델은 진짜 모서리를 이음매와 구별해 읽고 거리에 단조로 반응한다(ρ 0.91). 하지만 반응이 "떨어짐"이 아니라 "멈춤"이다. 떨어뜨릴 때의 시각은 맞는 편(29건, MAE 2.9 프레임). 등시각 삼중항에서 같은 낙하 시각의 빠른 공(먼 모서리)이 덜 떨어져(T4 0.18 &lt; T1 0.36, T5 0.0 &lt; T2 0.25) 모델이 읽는 것은 시각(거리÷속도)이 아니라 <b>거리</b>다. 관련 편집 +0.29 [+0.14, +0.47]·+0.40 [+0.10, +0.81], 미끼 −0.10 [−0.30, +0.05]·+0.01. → <b>모서리 관측은 하나 낙하 동역학 오예측</b>.</div></div></div>')

    # cross-scene: variance
    P.append('<div class="scene"><div class="head"><h3>세 씬 공통 ① — 결과의 주인은 seed다</h3></div><div class="two"><figure class="fig">' + var_chart +
             '<figcaption>연속 사건 점수의 분산 분해(조건 × seed 표, 같은 seed를 모든 조건에 씀). 물리 조건이 설명하는 몫은 4–11%, 잡음 seed가 27–55%.</figcaption></figure>'
             '<div class="reading">같은 seed가 조건과 무관하게 멈추거나 되튄다(collide seed 1: 무거운 표적 −0.53 / 표적 없음 −0.68 / 벽 −1.84). 이는 "모델이 물리를 얼마나 아는가"를 seed 평균만으로 보고하면 잡음을 지식으로 오독할 수 있다는 뜻이다. 벤치마크 점수는 분산 분해와 함께, 모델 비교는 seed ≥ 12의 짝 설계로 해야 한다. Edge는 검열(21에서 자름) 때문에 잔차가 크다.</div></div></div>')

    # cross-scene: attribution
    P.append('<div class="scene"><div class="head"><h3>세 씬 공통 ② — 열지도는 의존을 국소화하지 못했다</h3><span class="pill none">H1 0/3 씬</span></div>'
             '<div class="prose"><p>②에서 확인된 "관련 구조 의존"이 입력의 그 자리에서 보이는가? 사전 등록 H1(핵심 셀에서 관련 구조의 그래디언트 밀도 &gt; 미끼)은 <b>세 씬 모두 불성립</b>. 원시 단일 step 그래디언트는 배경(하늘·뒷벽)에 지배되고 점수와 무관한 스칼라(잠재 평균)의 지도와 0.64–0.69 상관이다 — 즉 점수 특이적 신호가 아니다. 어텐션 라우팅은 예측된 공의 토큰이 <b>자기 과거</b>를 압도적으로 읽고(밀도 6–17), 관련 구조 &gt; 미끼의 방향성만 남긴다(11/12 셀, 비 1.5–4.4). 그러나 collide의 벽(0.13 &lt; 미끼 공 0.63)에서 깨져, 인과 관련성보다 "진행 방향의 공 같은 물체"를 고르는 물체 종류 편향이 의심된다. 사전 등록대로 열지도는 <b>선별 신호</b>로 강등하고, 반사실 편집을 주지표로 삼는다.</p></div>'
             + '</div>')
    P.append(heatmap_primer())
    P.append('<div class="scene" style="border:none;background:transparent;padding:0"><div class="head"><h3>열지도는 실제로 어떻게 보였나</h3></div>'
             '<div class="prose"><p>아래는 씬마다 한 조건(관련 구조가 결과를 바꾸는 셀)의 열지도를 seed 4개 평균으로 마지막 조건 프레임 위에 겹친 것이다. 보는 법: ②가 ①의 노랑 윤곽 안에서 밝아야 "관련 구조를 국소화했다"이고, ②와 ③이 닮았으면 그 지도는 점수와 무관한 일반 민감도다. ④는 예측된 공이 조건 프레임의 어디를 읽었는지다.</p></div></div>')
    P.append(heat_row(root, 'hill', 'B', 'Hill · B (언덕 14 cm, 실제는 되돌아옴)',
             '②의 질량은 하늘·뒷벽·바닥 무늬에 퍼져 있고 언덕(노랑 윤곽)은 배경보다 어둡다. ③이 ②와 거의 같다 — 이 지도는 "무엇이 공의 미래를 정하는가"가 아니라 "입력이 얼마나 흔들리면 잠재가 흔들리는가"를 그린 것이다. ④에서 밝은 곳은 공의 과거 위치와 지지면 띠이고 언덕은 균일 이하다.'))
    P.append(heat_row(root, 'collide', 'B', 'Collide · B (표적 S = 2, 실제는 되튐)',
             '표적 공(노랑 윤곽)에 그래디언트가 조금 더 모이지만(밀도 1.8 vs 미끼 1.0) 배경이 여전히 더 밝고, ③과의 상관이 높다. ④에서는 예측된 공이 자기 과거를 압도적으로 읽고, 진행 방향의 표적 공을 뒤쪽 미끼 공보다 더 읽는다(비 1.5) — 이것이 남은 방향성 신호다. 표적 오른쪽의 밝은 점들은 생성 영상에서 공이 도달한 자리와 같은 위치의 조건 토큰으로, 내용이 아니라 위치를 따라 읽는 성분이다.'))
    P.append(heat_row(root, 'edge', 'A', 'Edge · A (모서리 가까움, 실제는 프레임 12에 낙하)',
             '모서리(노랑 윤곽)보다 판의 왼쪽 끝(보라)이 오히려 밝고, 배경이 가장 밝다. ④의 공 토큰은 판 표면과 자기 과거를 읽으며, 모서리는 미끼보다 조금 더 읽되(비 1.6) 균일 이하다. 반사실 실험은 모델이 모서리에 반응함을 보였으므로, 국소화 실패는 모델이 아니라 이 지도 방식의 한계다.'))
    P.append('<div class="scene"><div class="head"><h3>영역별 집계표</h3></div>'
             + att_table("hill", "Hill (조건 6 × seed 4)") + att_table("collide", "Collide") + att_table("edge", "Edge") + '</div>')

    # validity
    P.append('<div class="callout warn" style="margin-top:1.2rem"><b>판독 유효성과 결함.</b> 생성 영상에서 공이 순간이동(연속 프레임 사이 120 px 초과 점프)하는 롤아웃이 hill 16%, collide 6%. collide는 공이 75% 속도를 유지해 조건당 6–8/12 롤아웃에서 오른쪽 프레임 밖으로 나간다(strict 유효율 0.26–0.34; edge는 0.91). 판독은 보간 궤적으로 계속하며 결함 롤아웃도 포함했다(제외 시 편향 우려). 표적 없음(collide XF)의 낮은 속도비 0.34는 프레임 경계 클리핑과 seed 고유 정지의 합성이다. Edge 접근 속도는 사후 지표다.</div></section>')

    # 04 implications
    P.append('<section id="impl"><h2><span class="num">04</span>시사점</h2><div class="two"><div class="prose">'
             '<p><b>1. 벤치마크 질문에 대한 첫 답 — 선택적이되 정확하지 않다.</b> 미끼 편집은 어느 씬에서도 효과가 없고(CI 모두 0 포함), Edge에서는 이음매를 모서리와 구별한다. 그러나 반응 크기는 실제의 1/3 이하이고 사건의 방향이 틀린다. 세 실패는 모두 "보이는 구조를 동역학으로 옮기는" 단계에 있다 — 관측 실패가 아니다.</p>'
             '<p><b>2. 실패 위치는 정보 종류에 따라 다르다.</b> 기하(Hill)와 경계 거리(Edge)에는 부분 반응, 관계(Collide 질량비)에는 무반응(벽에서만 반응). Edge의 등시각 삼중항은 모델이 거리는 읽지만 속도와 결합해 시각을 계산하지는 않음을 시사한다. 모델은 "언제"보다 "어디"에 반응한다.</p>'
             '<p><b>3. 이진 결과는 무반응을, 연속 지표는 부분 반응을 보여준다.</b> Phase A의 "언덕 무반응"은 연속 지표에서 단조 반응으로 바뀌었다. 정확도 층은 응답 함수로 재야 한다. 단, 사전 등록 지표가 검열되면(Edge 낙하 시각 151/216) 응답 함수가 사라지므로, 검열되지 않는 접근 지표(접근 속도·감속 시작)를 사전 등록에 넣어야 한다.</p>'
             '<p><b>4. 결과의 주인은 seed다.</b> common random numbers 덕에 조건 주효과와 seed 주효과를 분리할 수 있었고, 연속 점수 분산의 27–55%가 seed, 4–11%만 조건이었다. 벤치마크 점수는 분산 분해와 함께 보고해야 하고, 모델 간 비교에는 seed ≥ 12와 짝 설계가 필수다.</p>'
             '<p><b>5. 열지도는 주지표가 될 수 없다.</b> 단일 step 그래디언트는 세 씬 모두 배경 지배·점수 무관 성분 상관 0.64–0.69·H1 0/3. 어텐션의 방향성은 물체 종류 편향과 분리되지 않았다. 반사실 편집이 주지표, 열지도는 후보 선별용이다.</p>'
             '</div><div>'
             '<div class="callout key"><b>다음 단계 (제안, 승인 대기)</b><ol>'
             '<li><b>미끼 재설계</b>: 물체 종류를 통제한 미끼(공 뒤 언덕 대신 공, 경로 위지만 인과 무관한 물체)로 어텐션 방향성이 인과인지 종류 편향인지 판별.</li>'
             '<li><b>비검열 지표 사전 등록</b>: 접근 속도비·감속 시작 프레임·접촉 후 속도비를 세 씬 공통 연속 지표로.</li>'
             '<li><b>반응 문턱 촘촘히</b>: Collide에서 무거운 공(S 2)과 벽 사이(S 4·8·고정 블록), Hill에서 14–20 cm.</li>'
             '<li><b>모델 비교</b>: 같은 조건·seed로 Cosmos-2.5, V-JEPA 2-AC(잠재) — 실패 위치가 모델마다 같은지.</li>'
             '<li><b>생성 결함 규약</b>: 순간이동·프레임 이탈 롤아웃의 포함/제외 민감도 보고.</li></ol></div>'
             '<div class="callout warn" style="margin-top:.8rem"><b>한계.</b> 모델 1개 · seed 12 · 조건당 n = 12 · 위치 층 seed 4. Edge 접근 속도는 사후 지표. Relevance Index는 평균 효과 0에서도 seed 산포 차로 양수가 될 수 있음(Collide 0.15). 판독기는 색 마스크 궤적이라 가려짐·프레임 이탈에 취약.</div>'
             '</div></div></section>')

    P.append('<footer>근거 파일: code_vwm/artifacts/pvr/{hill,collide,edge}/analysis/summary.json · attrib/summary.json · edge/analysis/speed_diag.json · variance_decomp.json · verify.json. '
             '설계: SimDROID/DESIGN_PVR_POC_2026-09-06.md · 일지: PROGRESS (46)(47) · 정본: HANDOFF §3.10. 수치 전체 보고 페이지: pvr_poc_report_2026-09-07.html. Claude(지훈 세션) 작성, 시각 KST.</footer></div>')
    out.write_text("\n".join(P))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("artifacts/pvr"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/pvr/explainer.html"))
    a = ap.parse_args()
    build(a.root, a.out)


if __name__ == "__main__":
    main()
