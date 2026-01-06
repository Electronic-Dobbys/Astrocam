#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_BIN="python3"

echo "=== [1/6] Actualizando sistema y dependencias base ==="
sudo apt-get update
sudo apt-get install -y \
  ${PYTHON_BIN} \
  ${PYTHON_BIN}-pip \
  ${PYTHON_BIN}-venv \
  build-essential \
  pkg-config \
  libgl1 \
  libegl1 \
  libxcb-xinerama0 \
  libxkbcommon-x11-0 \
  libfontconfig1 \
  libfreetype6 \
  libdbus-1-3 \
  libnss3 \
  libx11-6 \
  libx11-xcb1 \
  libxext6 \
  libxrender1 \
  libxi6 \
  libsm6 \
  libice6 \
  libxcb1 \
  libxfixes3 \
  libxrandr2 \
  libxcomposite1 \
  libxcursor1 \
  libxdamage1 \
  libxtst6 \
  libjpeg62-turbo \
  zlib1g

echo "=== [2/6] Creando entorno virtual (venv) ==="
if [[ -d "${VENV_DIR}" ]]; then
  echo "-> venv ya existe: ${VENV_DIR}"
else
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
  echo "-> venv creado en: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "=== [3/6] Actualizando pip/setuptools/wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== [4/6] Instalando dependencias Python comunes del proyecto ==="
pip install --upgrade numpy pillow

echo "=== [5/6] Intentando instalar PySide6 (Qt6) ==="
set +e
pip install --upgrade PySide6
PYSIDE_OK=$?
set -e

if [[ ${PYSIDE_OK} -ne 0 ]]; then
  echo "!! PySide6 falló (esto puede pasar en algunas Raspberry)."
  echo "=== Fallback: Instalando PyQt5 (Qt5) ==="
  pip install --upgrade PyQt5
  echo "-> Instalado PyQt5 como alternativa."
else
  echo "-> Instalado PySide6 correctamente."
fi

echo "=== [6/6] Creando script de ejecución run.sh ==="
cat > "${PROJECT_DIR}/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(pwd)/venv/bin/activate"

# Si tu archivo principal se llama distinto, cámbialo aquí:
APP="astro_capture_app.py"

python3 "$APP"
EOF

chmod +x "${PROJECT_DIR}/run.sh"

echo ""
echo "✅ Instalación completa."
echo "Para ejecutar:"
echo "  ./run.sh"
echo ""
echo "Nota:"
echo " - Si se instaló PyQt5 (fallback), tu app debe estar en versión PyQt5"
echo "   o adaptarla (puedo pasarte una versión compatible)."
