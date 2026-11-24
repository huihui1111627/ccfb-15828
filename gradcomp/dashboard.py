"""离线 HTML 仪表盘：单文件、内联 SVG、零外部依赖。

四个图块（与需求一一对应）：
1. 每层梯度压缩前后的大小（含压缩比）
2. 每步通信耗时（实测 loopback 耗时 + 千兆网模型耗时）
3. 误差累积量范数随步数的变化
4. 不同压缩方法的损失曲线对比
另附：最终损失/通信字节汇总表与全部自动化校验结果。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .charts import PALETTE, grouped_bar_chart, line_chart

_CSS = """
:root { --bg:#f8fafc; --card:#ffffff; --ink:#0f172a; --muted:#64748b;
        --line:#e2e8f0; --ok:#16a34a; --bad:#dc2626; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',
       'Segoe UI',sans-serif; font-size:14px; }
header { padding:22px 28px 10px; }
h1 { margin:0 0 4px; font-size:22px; }
.sub { color:var(--muted); font-size:13px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
        gap:18px; padding:16px 28px 32px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:16px 18px; box-shadow:0 1px 2px rgba(15,23,42,.05);
        overflow-x:auto; }
.card h2 { margin:0 0 2px; font-size:16px; }
.card .hint { color:var(--muted); font-size:12px; margin-bottom:8px; }
.chart { width:100%; height:auto; }
.grid { }.axis,.grid { stroke:#cbd5e1; stroke-width:1; }
.grid { stroke:#eef2f7; }
.axis-label { fill:#64748b; font-size:11px; }
.axis-title { fill:#334155; font-size:12px; }
.chart-title { fill:#0f172a; font-size:13px; font-weight:600; }
.legend { fill:#334155; font-size:11px; }
.bar-value { fill:#475569; font-size:10px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { border-bottom:1px solid var(--line); padding:7px 8px; text-align:right;
        white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; background:#f8fafc; position:sticky;
     top:0; }
.badge { display:inline-block; padding:2px 9px; border-radius:999px;
         font-size:12px; font-weight:600; }
.ok { background:#dcfce7; color:var(--ok); }
.bad { background:#fee2e2; color:var(--bad); }
.checks li { margin:5px 0; list-style:none; }
.checks ul { padding-left:0; }
.xlabel { font-weight:600; }
.full { grid-column:1 / -1; }
@media (max-width:1080px){ .grid{grid-template-columns:1fr;} }
"""


def _panel(card_id: str, title: str, hint: str, svg: str, full: bool = False) -> str:
    cls = "card full" if full else "card"
    return (f"<div class='{cls}' id='{card_id}'><h2>{html.escape(title)}</h2>"
            f"<div class='hint'>{html.escape(hint)}</div>{svg}</div>")


def _panel_layer_sizes(run: dict) -> str:
    records = run["step_records"]
    agg: dict[str, dict] = {}
    for rec in records:
        for t in rec["tensors"]:
            slot = agg.setdefault(t["name"], {"raw": 0, "comp": 0})
            slot["raw"] += t["raw_bytes"]
            slot["comp"] += t["compressed_bytes"]
            slot["shape"] = tuple(
                run["summary"]["per_tensor"][t["name"]]["shape"])
            slot["kinds"] = run["summary"]["per_tensor"][t["name"]]["kinds"]
    groups = []
    shape_txt = []
    for name in sorted(agg, key=lambda n: agg[n]["raw"]):
        slot = agg[name]
        ratio = slot["comp"] / slot["raw"] if slot["raw"] else 1.0
        groups.append({
            "group": name,
            "bars": [
                {"name": "压缩前(原始float32)", "value": slot["raw"]},
                {"name": "压缩后实际上行", "value": slot["comp"],
                 "color": PALETTE[1]},
            ],
            "_ratio": ratio,
        })
        shape_txt.append(
            f"{name}{list(slot['shape'])} {'/'.join(slot['kinds'])} "
            f"{ratio*100:.1f}%")
    svg = grouped_bar_chart(groups, title="",
                            y_label="累计上行字节数(全部训练步)")
    return _panel("p1", "① 每层梯度压缩前后大小",
                  "柱高为该层全部训练步累计线上字节；各层 [形状] 载荷类型 "
                  "压缩后/前比例：" + "；".join(shape_txt),
                  svg, full=True)


def _panel_comm_time(payload: dict) -> str:
    series_mod = []
    for i, run in enumerate(payload["runs"]):
        color = PALETTE[i % len(PALETTE)]
        steps = [r["step"] for r in run["step_records"]]
        # 模型耗时是千兆网的可信估计；实测 loopback 仅在 JSON 中保留，
        # 不叠加到图上以免 24 条线无法辨认
        series_mod.append({
            "name": f"{run['label']}", "x": steps,
            "y": [r["modeled_seconds"] * 1e3 for r in run["step_records"]],
            "color": color,
        })
    hint = ("按配置带宽(1 Gbps)/RTT(0.2ms)估算的每步通信耗时(ms)；"
            "实测 loopback 收发耗时保存在 results.json。热身步未压缩，"
            "所以前几步字节与耗时明显更高，爬坡后随压缩比例提高而下降")
    svg = line_chart(series_mod, x_label="训练步",
                     y_label="单步通信耗时 (ms)", legend="bottom")
    return _panel("p2", "② 每步通信耗时", hint, svg)


def _panel_residual(payload: dict) -> str:
    series = []
    for i, run in enumerate(payload["runs"]):
        if run["config"]["method"] == "none":
            continue
        steps = [r["step"] for r in run["step_records"]]
        series.append({
            "name": run["label"], "x": steps,
            "y": [r["total_residual_norm"] for r in run["step_records"]],
            "color": PALETTE[i % len(PALETTE)],
        })
    svg = line_chart(series, x_label="训练步",
                     y_label="||本地误差累积||₂ (全部层平方和开根)",
                     legend="right")
    return _panel("p3", "③ 误差累积量范数变化",
                  "关闭 EF 的运行每步丢弃残差，范数仅反映当前步压缩误差；"
                  "开启 EF 时范数随训练收敛而回落", svg)


def _panel_loss(payload: dict, dataset: str) -> tuple[str, list[dict]]:
    runs = [r for r in payload["runs"] if r["config"]["dataset"] == dataset]
    series = []
    for i, run in enumerate(runs):
        steps = list(range(len(run["loss_curve"])))
        series.append({
            "name": run["label"], "x": steps, "y": run["loss_curve"],
            "color": PALETTE[i % len(PALETTE)],
        })
    svg = line_chart(series, x_label="训练步", y_label="训练集 MSE",
                     legend="right")
    return _panel("p4", f"④ 损失曲线对比 — {dataset}",
                  "固定随机种子；热身 3 步不压缩，之后线性提高压缩比例", svg), runs


def _summary_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        s = r["summary"]
        rel = ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(r['label'])}</td>"
            f"<td>{r['final_train_loss']:.5f}</td>"
            f"<td>{r['final_test_loss']:.5f}</td>"
            f"<td>{s['uplink_bytes']:,}</td>"
            f"<td>{s['downlink_bytes']:,}</td>"
            f"<td>{s['compression_ratio']*100:.2f}%</td>"
            f"<td>{s['modeled_seconds']*1e3:.2f}</td>"
            "</tr>")
    table = (
        "<table><thead><tr><th>方法</th><th>最终训练损失</th>"
        "<th>最终测试损失</th><th>上行字节</th><th>下行字节</th>"
        "<th>上行压缩率</th><th>模型通信耗时(ms)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>")
    return _panel("p5", "汇总：最终损失与通信量",
                  "上行字节=各 worker 发送总量；下行所有方法均为 dense 平均梯度，"
                  "因此对比重点看上行", table, full=True)


def _checks_panel(payload: dict) -> str:
    items = []
    for c in payload["checks"]:
        badge = ("<span class='badge ok'>PASS</span>" if c["passed"]
                 else "<span class='badge bad'>FAIL</span>")
        detail = {
            "sparse_merge": (f"下标并集 {c.get('union_index_count')} 个、"
                             f"重合仅 {c.get('shared_index_count')} 个，"
                             f"合并最大误差 {c.get('max_abs_error', 0):.2e}"),
            "mp_vs_reference": (f"多进程 vs 单进程参照：参数最大差 "
                                f"{c.get('param_max_abs_diff', 0):.2e}，"
                                f"损失曲线最大差 {c.get('loss_curve_max_diff', 0):.2e}，"
                                f"字节差 {c.get('uplink_byte_diff', 0)}"),
            "ef_accuracy": (f"相对不压缩基线损失偏差 "
                            f"{c.get('relative_loss', 0)*100:.2f}% "
                            f"(容差 ≤{c.get('tolerance', 0)*100:.0f}%)"
                            + ("" if c.get("gated", True)
                               else "；非门控展示项，不参与硬性判定")),
            "no_ef_degradation": (f"相对不压缩基线损失偏差 "
                                  f"{c.get('relative_loss', 0)*100:.2f}% "
                                  f"(要求 ≥{c.get('threshold', 0)*100:.0f}% "
                                  f"才算明显变差)"
                                  + ("" if c.get("gated", True)
                                     else "；非门控展示项，不参与硬性判定")),
        }.get(c["name"], json.dumps(c, ensure_ascii=False))
        label = f"[{c['label']}] " if c.get("label") else ""
        items.append(f"<li>{badge} <b>{html.escape(c['name'])}</b> "
                     f"{html.escape(label)}— {html.escape(detail)}</li>")
    return _panel("p6", "自动化校验", "稀疏合并 / 多进程一致性 / "
                  "EF 精度 / 关闭 EF 退化",
                  f"<div class='checks'><ul>{''.join(items)}</ul></div>",
                  full=True)


def build_dashboard(payload: dict, out_path: str | Path) -> Path:
    datasets = sorted({r["config"]["dataset"] for r in payload["runs"]})
    # 图块①用 generic 数据集的代表运行（per-layer 结构各数据集一致展示其一）
    representative = next(
        r for r in payload["runs"]
        if r["config"]["dataset"] == datasets[0]
        and r["config"]["method"] == "topk")

    panels = [_panel_layer_sizes(representative), _panel_comm_time(payload),
              _panel_residual(payload)]
    loss_panels = []
    table_runs = []
    for ds in datasets:
        panel, runs = _panel_loss(payload, ds)
        loss_panels.append(panel)
        table_runs.extend(runs)
    panels.extend(loss_panels)
    panels.append(_summary_table(table_runs))
    panels.append(_checks_panel(payload))

    n_runs = len(payload["runs"])
    thr = payload.get("thresholds", {})
    head = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>梯度压缩通信效果仪表盘</title>"
        f"<style>{_CSS}</style></head><body>"
        "<header><h1>数据并行训练 · 梯度压缩通信效果仪表盘</h1>"
        f"<div class='sub'>{n_runs} 组固定种子实验 · "
        f"EF 精度容差 {thr.get('ef_tolerance', 0.15)*100:.0f}% · "
        "纯 numpy 多进程模拟，链路默认按 1 Gbps / RTT 0.2ms 建模"
        "</div></header><main class='grid'>"
    )
    doc = head + "".join(panels) + "</main></body></html>"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
