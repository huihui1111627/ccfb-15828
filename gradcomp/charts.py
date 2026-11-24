"""极简 SVG 绘图辅助（无第三方依赖），供离线仪表盘使用。"""

from __future__ import annotations

import html
import math

import numpy as np

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea",
           "#ea580c", "#0891b2", "#a16207", "#475569"]


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    if vmin == vmax:
        return [vmin]
    span = vmax - vmin
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        step = m * mag
        if span / step <= n + 0.5:
            break
    start = math.ceil(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def line_chart(series: list[dict], width: int = 760, height: int = 340,
               x_label: str = "", y_label: str = "", title: str = "",
               y_min: float | None = None, y_max: float | None = None,
               legend: str = "right") -> str:
    """series: [{name, x:[...], y:[...], dash?}]

    legend: "right" 右侧图例；"bottom" 底部横向自动换行；"none" 不画图例。
    """
    pad_l, pad_t = 58, 34
    pad_b = 42
    pad_r = 16
    legend_rows = 1
    if legend == "right":
        pad_r = 210
    elif legend == "bottom":
        # 估算底部分行：每行约 8 个条目
        legend_rows = max(1, int(np.ceil(len(series) / 8.0)))
        pad_b = 42 + legend_rows * 17
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_x = [v for s in series for v in s["x"]]
    all_y = [v for s in series for v in s["y"]]
    if not all_y:
        return f"<svg width='{width}' height='{height}'></svg>"
    x0, x1 = min(all_x), max(all_x)
    lo = min(all_y) if y_min is None else y_min
    hi = max(all_y) if y_max is None else y_max
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = (hi - lo) * 0.05
    lo -= pad if y_min is None else 0
    hi += pad if y_max is None else 0

    def sx(x):
        return pad_l + (x - x0) / max(x1 - x0, 1e-12) * plot_w

    def sy(y):
        return pad_t + (1 - (y - lo) / (hi - lo)) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' "
             f"role='img' aria-label='{html.escape(title)}'>"]
    if title:
        parts.append(f"<text x='{width/2}' y='18' text-anchor='middle' "
                     f"class='chart-title'>{html.escape(title)}</text>")

    for t in _nice_ticks(lo, hi):
        y = sy(t)
        parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' "
                     f"y2='{y:.1f}' class='grid'/>")
        parts.append(f"<text x='{pad_l-6}' y='{y+4:.1f}' text-anchor='end' "
                     f"class='axis-label'>{_fmt(t)}</text>")
    n = max(len(set(all_x)), 1)
    tick_step = max(1, n // 8)
    for t in sorted(set(all_x)):
        if int(t) % tick_step != 0 and t != x1:
            continue
        x = sx(t)
        parts.append(f"<text x='{x:.1f}' y='{height-pad_b+16}' "
                     f"text-anchor='middle' class='axis-label'>{int(t)}</text>")

    parts.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' "
                 f"y2='{height-pad_b}' class='axis'/>")
    parts.append(f"<line x1='{pad_l}' y1='{height-pad_b}' "
                 f"x2='{width-pad_r}' y2='{height-pad_b}' class='axis'/>")

    for i, s in enumerate(series):
        color = s.get("color", PALETTE[i % len(PALETTE)])
        dash = " stroke-dasharray='5,4'" if s.get("dash") else ""
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}"
                       for x, y in zip(s["x"], s["y"]))
        parts.append(f"<polyline fill='none' stroke='{color}' "
                     f"stroke-width='2'{dash} points='{pts}'/>")

    if x_label:
        parts.append(f"<text x='{pad_l+plot_w/2}' y='{height-6}' "
                     f"text-anchor='middle' class='axis-title'>{html.escape(x_label)}</text>")
    if y_label:
        parts.append(f"<text transform='translate(15,{pad_t+plot_h/2}) "
                     f"rotate(-90)' text-anchor='middle' class='axis-title'>"
                     f"{html.escape(y_label)}</text>")

    if legend == "right":
        # 紧凑两列图例，长标签截断，避免相互重叠
        col_w = 96
        row_h = 17
        x0 = width - pad_r + 8
        y0 = pad_t + 4
        for i, s in enumerate(series):
            color = s.get("color", PALETTE[i % len(PALETTE)])
            col = i // 12           # 每列最多 12 行
            row = i % 12
            x = x0 + col * col_w
            y = y0 + row * row_h
            name = s["name"]
            if len(name) > 14:
                name = name[:13] + "…"
            dash = " stroke-dasharray='3,2'" if s.get("dash") else ""
            parts.append(f"<line x1='{x}' y1='{y}' x2='{x+16}' y2='{y}' "
                         f"stroke='{color}' stroke-width='2.5'{dash}/>")
            parts.append(f"<text x='{x+20}' y='{y+3.5}' class='legend'>"
                         f"{html.escape(name)}</text>")
    elif legend == "bottom":
        # 底部横向排列，自动换行
        per_row = max(1, (plot_w - 10) // 150)
        item_w = plot_w / per_row
        for i, s in enumerate(series):
            color = s.get("color", PALETTE[i % len(PALETTE)])
            r = i // per_row
            c = i % per_row
            x = pad_l + c * item_w + 4
            y = height - pad_b + 16 + r * 17
            name = s["name"]
            if len(name) > 18:
                name = name[:17] + "…"
            dash = " stroke-dasharray='3,2'" if s.get("dash") else ""
            parts.append(f"<line x1='{x:.1f}' y1='{y}' x2='{x+18:.1f}' "
                         f"y2='{y}' stroke='{color}' stroke-width='2.5'{dash}/>")
            parts.append(f"<text x='{x+23:.1f}' y='{y+3.5}' class='legend'>"
                         f"{html.escape(name)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(groups: list[dict], width: int = 720, height: int = 360,
                      title: str = "", y_label: str = "") -> str:
    """groups: [{group, bars:[{name, value, color?}]}]，同组双柱对比。"""
    pad_l, pad_r, pad_t, pad_b = 58, 16, 34, 120
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_vals = [b["value"] for g in groups for b in g["bars"]]
    lo, hi = 0.0, max(all_vals) * 1.18 if all_vals else 1.0

    def sy(v):
        return pad_t + (1 - v / hi) * plot_h

    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' "
             f"role='img' aria-label='{html.escape(title)}'>"]
    if title:
        parts.append(f"<text x='{width/2}' y='18' text-anchor='middle' "
                     f"class='chart-title'>{html.escape(title)}</text>")
    # 刻度最多取 5 个，避免文字相互重叠
    ticks = _nice_ticks(lo, hi)
    if len(ticks) > 5:
        step_idx = max(1, len(ticks) // 5)
        ticks = ticks[::step_idx]
    for t in ticks:
        y = sy(t)
        parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' "
                     f"y2='{y:.1f}' class='grid'/>")
        parts.append(f"<text x='{pad_l-6}' y='{y+4:.1f}' text-anchor='end' "
                     f"class='axis-label'>{_fmt_compact(t)}</text>")

    n_groups = max(len(groups), 1)
    slot = plot_w / n_groups
    bar_names = [b["name"] for b in groups[0]["bars"]] if groups else []
    n_bars = max(len(bar_names), 1)
    bar_w = min(26.0, slot / (n_bars + 1))

    for gi, group in enumerate(groups):
        cx = pad_l + slot * (gi + 0.5)
        total_w = bar_w * n_bars + 4 * (n_bars - 1)
        x_start = cx - total_w / 2
        for bi, bar in enumerate(group["bars"]):
            color = bar.get("color", PALETTE[bi % len(PALETTE)])
            x = x_start + bi * (bar_w + 4)
            y = sy(bar["value"])
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
                         f"height='{pad_t+plot_h-y:.1f}' fill='{color}' rx='2'>"
                         f"<title>{html.escape(group['group'])} / "
                         f"{html.escape(bar['name'])}: {_fmt(bar['value'])}</title></rect>")
            parts.append(f"<text x='{x+bar_w/2:.1f}' y='{y-4:.1f}' "
                         f"text-anchor='middle' class='bar-value'>"
                         f"{_fmt_compact(bar['value'])}</text>")
        # 横轴标签旋转 -28 度，长名称在 SVG 内自动截断到 24 字符
        label = group["group"]
        if len(label) > 24:
            label = label[:23] + "…"
        parts.append(
            f"<text transform='translate({cx:.1f},{height-pad_b+12}) rotate(-28)' "
            f"text-anchor='end' class='axis-label xlabel'>"
            f"{html.escape(label)}</text>")

    for bi, name in enumerate(bar_names):
        color = PALETTE[bi % len(PALETTE)]
        lx = pad_l + 12 + bi * 200
        ly = height - 16
        parts.append(f"<rect x='{lx}' y='{ly-10}' width='12' height='12' "
                     f"fill='{color}' rx='2'/>")
        parts.append(f"<text x='{lx+17}' y='{ly}' class='legend'>"
                     f"{html.escape(name)}</text>")
    if y_label:
        parts.append(f"<text transform='translate(15,{pad_t+plot_h/2}) "
                     f"rotate(-90)' text-anchor='middle' class='axis-title'>"
                     f"{html.escape(y_label)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _fmt(v: float) -> str:
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
        return f"{v:.2e}"
    return f"{v:.3g}"


def _fmt_compact(v: float) -> str:
    if abs(v) >= 1e9:
        return f"{v/1e9:.1f}G"
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.2g}"
