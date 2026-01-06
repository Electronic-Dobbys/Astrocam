#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

source "./venv/bin/activate"

# Cargar env ZWO (SDK)
if [[ -f "./env.sh" ]]; then
  source "./env.sh"
fi

echo "=== Self-check ZWO SDK ==="
echo "ASI_SDK_LIB=${ASI_SDK_LIB:-}"
echo "ZWO_ASI_LIB=${ZWO_ASI_LIB:-}"
ls -l "${ASI_SDK_LIB:-/usr/local/lib/libASICamera2.so}" || true
ldconfig -p | grep -i ASICamera2 || true
echo "=========================="

python3 "astro_capture_app_zwo.py"
