#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_BIN="python3"
APP_FILE="astro_capture_app_zwo.py"

TMP_DIR="${PROJECT_DIR}/.tmp_zwo"
DOWNLOAD_URL="https://dl.zwoastro.com/software?app=DeveloperCameraSdk&platform=linux&region=Overseas"
REFERER="https://www.zwoastro.com/"
UA="Mozilla/5.0"

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
  file \
  tar \
  bzip2 \
  libusb-1.0-0 \
  libudev1 \
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
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
cd "${TMP_DIR}"

echo "-> Descargando SDK (ZIP) desde endpoint nuevo..."
# -OJ respeta Content-Disposition (nombre real del zip)
curl -fL --retry 3 -A "${UA}" -e "${REFERER}" -OJ "${DOWNLOAD_URL}"

ZIP_FILE="$(ls -1 *.zip 2>/dev/null | head -n 1 || true)"
if [[ -z "${ZIP_FILE}" ]]; then
  echo "❌ No se encontró ZIP descargado en ${TMP_DIR}"
  ls -la
  exit 1
fi

echo "-> ZIP descargado: ${ZIP_FILE}"
file "${ZIP_FILE}"

echo "-> Extrayendo ZIP..."
unzip -q "${ZIP_FILE}"

SDK_TARBZ2="$(find . -type f -name "ASI_linux_mac_SDK_*.tar.bz2" | head -n 1 || true)"
if [[ -z "${SDK_TARBZ2}" ]]; then
  echo "❌ No se encontró ASI_linux_mac_SDK_*.tar.bz2 dentro del ZIP."
  find . -maxdepth 3 -type f | head -n 50
  exit 1
fi

echo "-> Encontrado SDK tar.bz2: ${SDK_TARBZ2}"
file "${SDK_TARBZ2}"

echo "-> Extrayendo SDK tar.bz2..."
tar -xjf "${SDK_TARBZ2}"

SDK_ROOT="$(find . -maxdepth 3 -type d -name 'ASI_linux_mac_SDK_V*' | head -n 1 || true)"
if [[ -z "${SDK_ROOT}" ]]; then
  echo "❌ No se encontró carpeta ASI_linux_mac_SDK_V* tras extraer."
  find . -maxdepth 3 -type d | head -n 50
  exit 1
fi
echo "-> SDK root: ${SDK_ROOT}"

ARCH="$(uname -m)"
case "${ARCH}" in
  aarch64) LIB_DIR="${SDK_ROOT}/lib/armv8" ;;
  armv7l)  LIB_DIR="${SDK_ROOT}/lib/armv7" ;;
  armv6l)  LIB_DIR="${SDK_ROOT}/lib/armv6" ;;
  x86_64)  LIB_DIR="${SDK_ROOT}/lib/x64" ;;
  i386|i686) LIB_DIR="${SDK_ROOT}/lib/x86" ;;
  *)
    echo "⚠️ Arquitectura no reconocida (${ARCH}). Usaré armv8 como fallback."
    LIB_DIR="${SDK_ROOT}/lib/armv8"
    ;;
esac

if [[ ! -d "${LIB_DIR}" ]]; then
  echo "❌ No existe el directorio de librerías esperado: ${LIB_DIR}"
  ls -la "${SDK_ROOT}/lib" || true
  exit 1
fi
echo "-> Usando librerías desde: ${LIB_DIR}"

# Copiar SIEMPRE la librería versionada (más estable)
FOUND_VER="$(ls -1 "${LIB_DIR}"/libASICamera2.so.* 2>/dev/null | head -n 1 || true)"
FOUND_SO="${LIB_DIR}/libASICamera2.so"

if [[ -z "${FOUND_VER}" || ! -f "${FOUND_VER}" ]]; then
  echo "❌ No encontré libASICamera2.so.* en ${LIB_DIR}"
  ls -la "${LIB_DIR}"
  exit 1
fi

echo "-> Encontrado (versionada): ${FOUND_VER}"
file "${FOUND_VER}" || true

echo "-> Instalando en /usr/local/lib..."
sudo cp -f "${FOUND_VER}" /usr/local/lib/
# Copia también el .so no versionado si existe (opcional)
if [[ -f "${FOUND_SO}" ]]; then
  sudo cp -f "${FOUND_SO}" /usr/local/lib/
fi

BASE_NAME="$(basename "${FOUND_VER}")"
sudo chmod 644 "/usr/local/lib/${BASE_NAME}"

echo "-> Symlink estable: /usr/local/lib/libASICamera2.so -> ${BASE_NAME}"
sudo ln -sf "/usr/local/lib/${BASE_NAME}" /usr/local/lib/libASICamera2.so

echo "-> Asegurar linker path + ldconfig"
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/zwo.conf >/dev/null
sudo ldconfig

echo "-> Instalando reglas udev (asi.rules) para acceso USB..."
if [[ -f "${SDK_ROOT}/lib/asi.rules" ]]; then
  sudo cp -f "${SDK_ROOT}/lib/asi.rules" /etc/udev/rules.d/99-asi.rules
  sudo udevadm control --reload-rules
  sudo udevadm trigger
else
  echo "⚠️ No encontré asi.rules en ${SDK_ROOT}/lib (continuo igual)."
fi

echo "=== [6/9] Verificación de librería instalada ==="
ls -l /usr/local/lib/libASICamera2.so* || true
echo "-> ldconfig:"
ldconfig -p | grep -i ASICamera2 || true

echo "=== [7/9] Crear env.sh + run.sh ==="
cat > "${PROJECT_DIR}/env.sh" <<'EOF'
#!/usr/bin/env bash
export ASI_SDK_LIB="/usr/local/lib/libASICamera2.so"
export ZWO_ASI_LIB="/usr/local/lib/libASICamera2.so"
# Asegura que el loader encuentre /usr/local/lib incluso si ldconfig no basta en algunos entornos
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
EOF
chmod +x "${PROJECT_DIR}/env.sh"

cat > "${PROJECT_DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$(cd "\$(dirname "\$0")" && pwd)"

source "./venv/bin/activate"
source "./env.sh"

echo "=== Self-check ZWO SDK ==="
echo "python: \$(python3 -c 'import sys; print(sys.executable)')"
echo "ASI_SDK_LIB=\${ASI_SDK_LIB}"
echo "ZWO_ASI_LIB=\${ZWO_ASI_LIB}"
echo "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}"
ls -l /usr/local/lib/libASICamera2.so* || true
ldconfig -p | grep -i ASICamera2 || true

echo "--- Test import zwoasi ---"
python3 - <<'PY'
import zwoasi
print("zwoasi OK")
print("num cameras:", zwoasi.get_num_cameras())
PY
echo "-------------------------"

python3 "${APP_FILE}"
EOF
chmod +x "${PROJECT_DIR}/run.sh"

echo "=== [8/9] Limpieza (opcional) ==="
echo "-> Dejo el SDK extraído en: ${TMP_DIR} (útil para debug)."

echo "=== [9/9] Listo ==="
echo "✅ Ejecuta con: ./run.sh"
