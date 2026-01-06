#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
PYTHON_BIN="python3"

# Cambia esto si tu archivo principal se llama distinto:
APP_FILE="astro_capture_app_zwo.py"

echo "=== [0/8] Info ==="
echo "Proyecto: ${PROJECT_DIR}"
echo "Venv:     ${VENV_DIR}"
echo "App:      ${APP_FILE}"
echo ""

echo "=== [1/8] Paquetes base del sistema ==="
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

echo "=== [2/8] Crear/activar venv ==="
if [[ ! -d "${VENV_DIR}" ]]; then
  ${PYTHON_BIN} -m venv "${VENV_DIR}"
  echo "-> venv creado"
else
  echo "-> venv ya existe"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "=== [3/8] Actualizar pip/setuptools/wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== [4/8] Instalar dependencias Python ==="
# PySide6 puede no tener wheel en algunas Raspberry. Aquí lo intentamos,
# si falla, detenemos y lo informamos claramente (ya que tu app usa PySide6).
set +e
pip install --upgrade PySide6
PYSIDE_OK=$?
set -e

if [[ ${PYSIDE_OK} -ne 0 ]]; then
  echo ""
  echo "❌ No se pudo instalar PySide6 con pip en esta Raspberry/OS."
  echo "   Soluciones típicas:"
  echo "   1) usar Raspberry Pi OS 64-bit (aarch64) con Python reciente"
  echo "   2) o migrar el código a PyQt5 (puedo adaptarlo)"
  echo ""
  exit 1
fi

pip install --upgrade numpy pillow zwoasi

echo "=== [5/8] Descargar e instalar ASICamera2 SDK (ZWO) ==="
ARCH="$(uname -m)"
SDK_DIR="${PROJECT_DIR}/zwo_sdk"
TMP_DIR="${PROJECT_DIR}/.tmp_zwo"
mkdir -p "${SDK_DIR}" "${TMP_DIR}"

# Heurística: aarch64 -> ARM64, armv7l -> ARM32
# (ZWO suele publicar diferentes zips)
if [[ "${ARCH}" == "aarch64" ]]; then
  SDK_URL="https://astronomy-imaging-camera.com/software/ASI_linux_mac_SDK_V1.30.tar.bz2"
elif [[ "${ARCH}" == "armv7l" || "${ARCH}" == "armv6l" ]]; then
  SDK_URL="https://astronomy-imaging-camera.com/software/ASI_linux_mac_SDK_V1.30.tar.bz2"
else
  echo "⚠️ Arquitectura no reconocida (${ARCH}). Intentaré igual con SDK genérico."
  SDK_URL="https://astronomy-imaging-camera.com/software/ASI_linux_mac_SDK_V1.30.tar.bz2"
fi

echo "-> Descargando SDK desde:"
echo "   ${SDK_URL}"

SDK_TARBZ2="${TMP_DIR}/ASI_SDK.tar.bz2"
curl -L "${SDK_URL}" -o "${SDK_TARBZ2}"

echo "-> Extrayendo SDK..."
tar -xjf "${SDK_TARBZ2}" -C "${TMP_DIR}"

# Buscar libASICamera2.so dentro del SDK
LIB_PATH_FOUND="$(find "${TMP_DIR}" -type f -name "libASICamera2.so" | head -n 1 || true)"
if [[ -z "${LIB_PATH_FOUND}" ]]; then
  echo ""
  echo "❌ No encontré libASICamera2.so dentro del SDK descargado."
  echo "   Estructura del SDK puede haber cambiado."
  echo "   Revisa manualmente el contenido en: ${TMP_DIR}"
  exit 1
fi

echo "-> Encontrado: ${LIB_PATH_FOUND}"
echo "-> Copiando a /usr/local/lib ..."
sudo cp -f "${LIB_PATH_FOUND}" /usr/local/lib/libASICamera2.so
sudo chmod 644 /usr/local/lib/libASICamera2.so
sudo ldconfig

echo "=== [6/8] Crear env.sh para ASI_SDK_LIB ==="
cat > "${PROJECT_DIR}/env.sh" <<'EOF'
#!/usr/bin/env bash
# Carga variables para ZWO SDK
export ASI_SDK_LIB="/usr/local/lib/libASICamera2.so"
EOF
chmod +x "${PROJECT_DIR}/env.sh"

echo "=== [7/8] Crear run.sh ==="
cat > "${PROJECT_DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "\$(cd "\$(dirname "\$0")" && pwd)"

# Cargar ASI SDK env
if [[ -f "./env.sh" ]]; then
  source "./env.sh"
fi

# Activar venv
source "./venv/bin/activate"

python3 "${APP_FILE}"
EOF
chmod +x "${PROJECT_DIR}/run.sh"

echo "=== [8/8] Listo ==="
echo ""
echo "✅ Instalación completa."
echo "Ejecuta la app con:"
echo "  ./run.sh"
echo ""
echo "Si no detecta libASICamera2.so, verifica:"
echo "  ls -l /usr/local/lib/libASICamera2.so"
echo "  echo \$ASI_SDK_LIB"
