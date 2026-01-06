#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_BIN="python3"

APP_FILE="astro_capture_app_zwo.py"

echo "=== Proyecto: ${PROJECT_DIR}"
echo "=== App:      ${APP_FILE}"
echo "=== Venv:     ${VENV_DIR}"
echo ""

echo "=== [1/9] Paquetes base del sistema ==="
sudo apt-get update
sudo apt-get install -y \
  ${PYTHON_BIN} \
  ${PYTHON_BIN}-pip \
  ${PYTHON_BIN}-venv \
  build-essential \
  pkg-config \
  curl \
  unzip \
  ca-certificates \
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

echo "=== [2/9] Crear/activar venv ==="
if [[ ! -d "${VENV_DIR}" ]]; then
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "=== [3/9] Actualizar pip ==="
pip install --upgrade pip setuptools wheel

echo "=== [4/9] Instalar dependencias Python ==="
set +e
pip install --upgrade PySide6
PYSIDE_OK=$?
set -e
if [[ ${PYSIDE_OK} -ne 0 ]]; then
  echo ""
  echo "❌ No se pudo instalar PySide6 con pip en esta Raspberry/OS."
  echo "   Recomendación: Raspberry Pi OS 64-bit (aarch64) + Python reciente."
  exit 1
fi

pip install --upgrade numpy pillow zwoasi

echo "=== [5/9] Descargar e instalar ASICamera2 SDK (ZWO) ==="

TMP_DIR="${PROJECT_DIR}/.tmp_zwo"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
cd "${TMP_DIR}"

SDK_ZIP="ASI_Camera_SDK.zip"

echo "-> Descargando SDK (endpoint nuevo de ZWO)..."
curl -fL --retry 3 -A "Mozilla/5.0" \
  -e "https://www.zwoastro.com/" \
  -OJ "https://dl.zwoastro.com/software?app=DeveloperCameraSdk&platform=linux&region=Overseas"

# Detectar el zip descargado (nombre viene por Content-Disposition)
ZIP_FILE="$(ls *.zip | head -n 1 || true)"
if [[ -z "${ZIP_FILE}" ]]; then
  echo "❌ No se descargó ningún archivo ZIP del SDK."
  exit 1
fi

echo "-> ZIP descargado: ${ZIP_FILE}"
file "${ZIP_FILE}"

echo "-> Extrayendo ZIP..."
unzip -q "${ZIP_FILE}"

# Buscar el tar.bz2 dentro
SDK_TARBZ2="$(find . -name "ASI_linux_mac_SDK_*.tar.bz2" | head -n 1 || true)"
if [[ -z "${SDK_TARBZ2}" ]]; then
  echo "❌ No se encontró ASI_linux_mac_SDK_*.tar.bz2 dentro del ZIP."
  exit 1
fi

echo "-> Encontrado SDK tar.bz2: ${SDK_TARBZ2}"
file "${SDK_TARBZ2}"

echo "-> Extrayendo SDK tar.bz2..."
tar -xjf "${SDK_TARBZ2}"

echo "-> Buscando libASICamera2.so..."
FOUND_LIB="$(find . -type f -name 'libASICamera2.so*' | head -n 1 || true)"
if [[ -z "${FOUND_LIB}" ]]; then
  echo "❌ No se encontró libASICamera2.so dentro del SDK."
  exit 1
fi

echo "-> Encontrado: ${FOUND_LIB}"

BASE_NAME="$(basename "${FOUND_LIB}")"

echo "-> Instalando en /usr/local/lib/${BASE_NAME}"
sudo cp -f "${FOUND_LIB}" "/usr/local/lib/${BASE_NAME}"
sudo chmod 644 "/usr/local/lib/${BASE_NAME}"

echo "-> Creando symlink libASICamera2.so"
sudo ln -sf "/usr/local/lib/${BASE_NAME}" /usr/local/lib/libASICamera2.so

echo "-> Actualizando linker cache"
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local-zwo.conf >/dev/null
sudo ldconfig

echo "=== [6/9] Crear symlink /usr/local/lib/libASICamera2.so ==="
# Si la encontrada ya es libASICamera2.so, igual lo linkeamos a sí misma
sudo ln -sf "/usr/local/lib/${BASE_NAME}" /usr/local/lib/libASICamera2.so

echo "=== [7/9] Asegurar linker path + ldconfig ==="
# Algunas distros no incluyen /usr/local/lib por defecto en ld.so.conf.d
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local-zwo.conf >/dev/null
sudo ldconfig

echo "=== [8/9] Crear env.sh + run.sh ==="
cat > "${PROJECT_DIR}/env.sh" <<'EOF'
#!/usr/bin/env bash
export ASI_SDK_LIB="/usr/local/lib/libASICamera2.so"
export ZWO_ASI_LIB="/usr/local/lib/libASICamera2.so"
EOF
chmod +x "${PROJECT_DIR}/env.sh"

cat > "${PROJECT_DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$(cd "\$(dirname "\$0")" && pwd)"

source "./venv/bin/activate"

# Cargar env ZWO (SDK)
if [[ -f "./env.sh" ]]; then
  source "./env.sh"
fi

echo "=== Self-check ZWO SDK ==="
echo "ASI_SDK_LIB=\${ASI_SDK_LIB:-}"
echo "ZWO_ASI_LIB=\${ZWO_ASI_LIB:-}"
ls -l "\${ASI_SDK_LIB:-/usr/local/lib/libASICamera2.so}" || true
ldconfig -p | grep -i ASICamera2 || true
echo "=========================="

python3 "${APP_FILE}"
EOF
chmod +x "${PROJECT_DIR}/run.sh"

echo "=== [9/9] Listo ==="
echo "✅ Ejecuta con: ./run.sh"
