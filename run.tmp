#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
APP="${PROJECT_DIR}/astrocam.py"

echo "=== Astro Capture App ==="

# 1) Verificar venv
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "❌ No se encontró el entorno virtual:"
  echo "   ${VENV_DIR}"
  echo "   Ejecuta primero: ./install.sh"
  exit 1
fi

# 2) Activar venv
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 3) Verificar archivo principal
if [[ ! -f "${APP}" ]]; then
  echo "❌ No se encontró el archivo principal:"
  echo "   ${APP}"
  exit 1
fi

# 4) Detectar backend Qt instalado
echo "→ Verificando backend Qt..."
QT_BACKEND=""

python3 - <<'EOF'
try:
    import PySide6
    print("PySide6")
except Exception:
    try:
        import PyQt5
        print("PyQt5")
    except Exception:
        print("NONE")
EOF
QT_BACKEND=$(python3 - <<'EOF'
try:
    import PySide6
    print("PySide6")
except Exception:
    try:
        import PyQt5
        print("PyQt5")
    except Exception:
        print("NONE")
EOF
)

if [[ "${QT_BACKEND}" == "NONE" ]]; then
  echo "❌ No se encontró PySide6 ni PyQt5 en el venv."
  echo "   Ejecuta nuevamente: ./install.sh"
  exit 1
fi

echo "→ Backend Qt detectado: ${QT_BACKEND}"

# 5) Ejecutar aplicación
echo "→ Iniciando aplicación..."
python3 "${APP}"
