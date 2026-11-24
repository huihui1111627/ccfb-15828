"""命令行入口。

用法::

    python -m gradcomp experiment --out-dir results [--small]
    python -m gradcomp verify            # 多进程 vs 单进程参照一致性
    python -m gradcomp test              # 全部单元/集成测试
    python -m gradcomp serve --dir results --port 8000
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
from pathlib import Path

from .dashboard import build_dashboard
from .runner import (default_experiment_configs, run_experiments,
                     verify_equivalence, verify_sparse_merge)
from .training import RunConfig, run_inprocess


def _cmd_experiment(args: argparse.Namespace) -> int:
    cfgs = default_experiment_configs(small=args.small)
    out_dir = Path(args.out_dir)
    print(f"[experiment] {len(cfgs)} 组配置，每组 {cfgs[0].world_size} 个进程，"
          f"输出目录 {out_dir}")
    payload = run_experiments(
        cfgs, str(out_dir),
        ef_tolerance=args.ef_tolerance, noef_degrade=args.noef_degrade)

    html_path = build_dashboard(payload, out_dir / "dashboard.html")
    print(f"[experiment] 结果写入 {out_dir/'results.json'}")
    print(f"[experiment] 仪表盘写入 {html_path}")

    print("\n== 汇总对比 ==")
    header = f"{'label':<24}{'train_loss':>12}{'test_loss':>12}" \
             f"{'uplink(B)':>14}{'ratio':>9}"
    print(header)
    for row in payload["comparison"]:
        print(f"{row['label']:<24}{row['final_train_loss']:>12.5f}"
              f"{row['final_test_loss']:>12.5f}"
              f"{row['uplink_bytes']:>14,}{row['compression_ratio']*100:>8.2f}%")

    print("\n== 校验 ==")
    ok = True
    for c in payload["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        ok = ok and c["passed"]
        print(f"  [{mark}] {c['name']} {c.get('label', '')}")
    return 0 if ok else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    print("[verify] 1) 稀疏合并正确性 ...")
    sparse = verify_sparse_merge()
    print(f"   {'PASS' if sparse['passed'] else 'FAIL'} {sparse}")

    print("[verify] 2) 多进程 vs 单进程参照（四种方法各跑一遍短训练）...")
    results = [sparse]
    ok = sparse["passed"]
    for method in ("none", "topk", "sign", "lowrank"):
        cfg = RunConfig(method=method, dataset="generic",
                        layer_sizes=[6, 32, 16, 1], steps=12,
                        batch_size=32, world_size=4)
        rep = verify_equivalence(cfg)
        results.append(rep)
        print(f"   [{'PASS' if rep['passed'] else 'FAIL'}] {method}: {rep}")
        ok = ok and rep["passed"]

    out = Path(args.out_dir) if args.out_dir else Path("results_verify")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verify.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[verify] 详细结果 -> {out/'verify.json'}")
    return 0 if ok else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    directory = Path(args.dir).resolve()
    if not (directory / "dashboard.html").exists():
        print(f"[serve] {directory/'dashboard.html'} 不存在，先运行 experiment",
              file=sys.stderr)
        return 2

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, fmt, *fargs):
            sys.stderr.write("[http] " + fmt % fargs + "\n")

    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/dashboard.html"
        print(f"[serve] 仪表盘地址: {url}  (Ctrl-C 停止)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """不启多进程，快速用 InProcess 通道跑一组并出仪表盘（冒烟用）。"""
    from .runner import _build_report
    cfgs = default_experiment_configs(small=True)
    runs = [run_inprocess(cfg)[0] for cfg in cfgs]
    report = _build_report(cfgs, runs, args.ef_tolerance, args.noef_degrade)
    payload = {
        "runs": runs, "checks": report["checks"],
        "comparison": report["comparison"],
        "thresholds": {"ef_tolerance": args.ef_tolerance,
                       "noef_degrade": args.noef_degrade},
    }
    out = Path(args.out_dir)
    path = build_dashboard(payload, out / "dashboard.html")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[demo] 单进程参照模式完成 -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gradcomp", description="梯度压缩通信模块（numpy 多进程模拟）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("experiment", help="完整实验矩阵 + 仪表盘")
    p_exp.add_argument("--out-dir", default="results")
    p_exp.add_argument("--small", action="store_true", help="缩短步数快速跑")
    p_exp.add_argument("--ef-tolerance", type=float, default=0.15)
    p_exp.add_argument("--noef-degrade", type=float, default=0.05)
    p_exp.set_defaults(func=_cmd_experiment)

    p_ver = sub.add_parser("verify", help="多进程 vs 单进程参照一致性校验")
    p_ver.add_argument("--out-dir", default="results_verify")
    p_ver.set_defaults(func=_cmd_verify)

    p_serve = sub.add_parser("serve", help="本地 HTTP 查看仪表盘")
    p_serve.add_argument("--dir", default="results")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_demo = sub.add_parser("demo", help="单进程快速冒烟并出仪表盘")
    p_demo.add_argument("--out-dir", default="results_demo")
    p_demo.add_argument("--ef-tolerance", type=float, default=0.15)
    p_demo.add_argument("--noef-degrade", type=float, default=0.05)
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
