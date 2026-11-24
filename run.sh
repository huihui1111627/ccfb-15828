#!/usr/bin/env bash
# 便捷入口：自动注入项目根目录到 PYTHONPATH，并用 -P 保持模块隔离。
# 用法示例：
#   ./run.sh test
#   ./run.sh verify
#   ./run.sh experiment --out-dir results
#   ./run.sh serve --dir results --port 8000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cmd="${1:-test}"
if [[ "$#" -gt 0 ]]; then shift; fi

case "$cmd" in
  test)
    exec python3 -P -m unittest discover -s "$ROOT/tests" "$@"
    ;;
  verify|experiment|demo|serve)
    exec python3 -P -m gradcomp "$cmd" "$@"
    ;;
  *)
    echo "未知命令: $cmd（可选 test|verify|experiment|demo|serve）" >&2
    exit 2
    ;;
esac
