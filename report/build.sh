#!/usr/bin/env bash
# report/template.md + outputs/*/metrics.json → report/report.md → report/report.docx
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${IP_REPORT_DIR:-$ROOT/report}"
cd "$ROOT"
uv run python tools/build_report.py
cd "$REPORT_DIR"
if command -v pandoc >/dev/null 2>&1; then
  pandoc report.md -o report.docx --from=gfm+tex_math_dollars --resource-path=".:$ROOT" --toc --toc-depth=2 \
    --metadata title="YZM505 Vize Ödevi — Rapor" --metadata lang=tr-TR
  echo "report.docx (pandoc) yazıldı: $REPORT_DIR/report.docx"
else
  echo "pandoc bulunamadı → python-docx yedeği"
  uv run --directory "$ROOT" python tools/md_to_docx.py "$REPORT_DIR/report.md" "$REPORT_DIR/report.docx"
fi
