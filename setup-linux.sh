#!/usr/bin/env bash
# ============================================================
#  OMNI-RECO v2.1 — Setup Linux universel
#  Supporte : Debian/Ubuntu/Mint | Fedora/RHEL/CentOS/Rocky
#             Arch/Manjaro       | openSUSE Leap/Tumbleweed
# ============================================================
set -euo pipefail

VENV_NAME="venv_omni"
PYTHON_REQUIRED="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERRORS=0

# ── Couleurs ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[ERREUR]${NC} $*"; ERRORS=$((ERRORS+1)); }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
step() { echo -e "\n${BOLD}$*${NC}"; }

echo ""
echo "============================================================"
echo "  OMNI-RECO v2.1 — Setup Linux"
echo "============================================================"
echo ""

# ══════════════════════════════════════════════════════════════
# ÉTAPE 1 — Détection de la distribution
# ══════════════════════════════════════════════════════════════
step "[1/7] Détection de la distribution Linux..."

DISTRO=""
PKG_MANAGER=""

if command -v apt-get &>/dev/null; then
    DISTRO="debian"
    PKG_MANAGER="apt-get"
    ok "Debian/Ubuntu/Mint détecté (apt)"
elif command -v dnf &>/dev/null; then
    DISTRO="fedora"
    PKG_MANAGER="dnf"
    ok "Fedora/RHEL/CentOS/Rocky détecté (dnf)"
elif command -v yum &>/dev/null; then
    DISTRO="centos"
    PKG_MANAGER="yum"
    ok "CentOS/RHEL (yum) détecté"
elif command -v pacman &>/dev/null; then
    DISTRO="arch"
    PKG_MANAGER="pacman"
    ok "Arch/Manjaro détecté (pacman)"
elif command -v zypper &>/dev/null; then
    DISTRO="opensuse"
    PKG_MANAGER="zypper"
    ok "openSUSE détecté (zypper)"
else
    err "Distribution non reconnue. Installe Python 3.10 manuellement."
    exit 1
fi

# ══════════════════════════════════════════════════════════════
# ÉTAPE 2 — Installation Python 3.10
# ══════════════════════════════════════════════════════════════
step "[2/7] Vérification Python 3.10..."

PYTHON_EXE=""

# Chercher python3.10 ou python3 >= 3.10
for cmd in python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -eq 10 ]; then
            PYTHON_EXE="$cmd"
            ok "Python 3.10 trouvé : $cmd ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON_EXE" ]; then
    warn "Python 3.10 absent. Installation via gestionnaire de paquets..."

    if [ "$DISTRO" = "debian" ]; then
        # Ubuntu 22.04+ a python3.10 en natif. Sinon deadsnakes PPA
        sudo apt-get update -qq
        sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
             python3-pip build-essential libssl-dev libffi-dev \
             libopencv-dev cmake git curl || {
            warn "Tentative via deadsnakes PPA..."
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update -qq
            sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
        }

    elif [ "$DISTRO" = "fedora" ]; then
        sudo dnf install -y python3.10 python3-pip python3-devel \
             gcc gcc-c++ cmake git curl openssl-devel libffi-devel \
             opencv-devel || {
            warn "Tentative compilation depuis source..."
            _install_python310_from_source
        }

    elif [ "$DISTRO" = "centos" ]; then
        sudo yum install -y epel-release
        sudo yum install -y python310 python310-pip python310-devel \
             gcc gcc-c++ cmake git curl openssl-devel || {
            warn "python310 absent dans EPEL. Compilation depuis source..."
            _install_python310_from_source
        }

    elif [ "$DISTRO" = "arch" ]; then
        # Arch : python = python3 (souvent 3.12+), on installe pyenv ou AUR
        sudo pacman -Sy --noconfirm python python-pip base-devel git curl cmake opencv
        warn "Arch : Python système peut être > 3.10."
        warn "Si problème mediapipe, utilise pyenv : https://github.com/pyenv/pyenv"

    elif [ "$DISTRO" = "opensuse" ]; then
        sudo zypper install -y python310 python310-pip python310-devel \
             gcc cmake git curl libopenssl-devel libffi-devel
    fi

    # Chercher à nouveau
    for cmd in python3.10 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -eq 3 ] && [ "$minor" -eq 10 ]; then
                PYTHON_EXE="$cmd"
                ok "Python 3.10 installé : $cmd"
                break
            fi
        fi
    done

    if [ -z "$PYTHON_EXE" ]; then
        err "Python 3.10 introuvable après installation."
        err "Installe-le manuellement : https://www.python.org/downloads/release/python-31011/"
        exit 1
    fi
fi

# ══════════════════════════════════════════════════════════════
# Fonction helper : compilation Python 3.10 depuis source
# ══════════════════════════════════════════════════════════════
_install_python310_from_source() {
    warn "Compilation Python 3.10.11 depuis source (5-15 min)..."
    cd /tmp
    curl -O https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz
    tar xzf Python-3.10.11.tgz
    cd Python-3.10.11
    ./configure --enable-optimizations --with-ensurepip=install
    make -j"$(nproc)"
    sudo make altinstall
    cd "$SCRIPT_DIR"
    ok "Python 3.10.11 compilé et installé."
}

# ══════════════════════════════════════════════════════════════
# ÉTAPE 3 — Venv Python 3.10
# ══════════════════════════════════════════════════════════════
step "[3/7] Environnement virtuel..."

cd "$SCRIPT_DIR"

if [ -d "$VENV_NAME" ]; then
    # Vérifier que le venv est bien en Python 3.10
    VENV_VER=$("$SCRIPT_DIR/$VENV_NAME/bin/python" -c \
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    VENV_MAJOR=$(echo "$VENV_VER" | cut -d. -f1)
    VENV_MINOR=$(echo "$VENV_VER" | cut -d. -f2)

    if [ "$VENV_MAJOR" -eq 3 ] && [ "$VENV_MINOR" -eq 10 ]; then
        ok "Venv existant en Python 3.10 — réutilisé."
    else
        warn "Venv existant en Python $VENV_VER ≠ 3.10 — suppression et recréation..."
        rm -rf "$VENV_NAME"
        $PYTHON_EXE -m venv "$VENV_NAME"
        ok "Venv recréé en Python 3.10."
    fi
else
    $PYTHON_EXE -m venv "$VENV_NAME"
    ok "Venv créé."
fi

# Active le venv pour ce script
source "$SCRIPT_DIR/$VENV_NAME/bin/activate"
ok "Venv activé."

# ══════════════════════════════════════════════════════════════
# ÉTAPE 4 — Mise à jour pip
# ══════════════════════════════════════════════════════════════
step "[4/7] Mise à jour pip..."
python -m pip install --upgrade pip --quiet
ok "pip à jour."

# ══════════════════════════════════════════════════════════════
# ÉTAPE 5 — Dépendances
# ══════════════════════════════════════════════════════════════
step "[5/7] Installation des dépendances..."
echo "(5 à 15 minutes selon connexion)"

if [ -f "$SCRIPT_DIR/requirements_v2.txt" ]; then
    info "Utilisation requirements_v2.txt..."
    pip install -r "$SCRIPT_DIR/requirements_v2.txt"
else
    pip install "mediapipe==0.10.9" --no-cache-dir
    pip install insightface onnxruntime
    pip install opencv-python numpy scipy rich
    pip install fastapi uvicorn python-multipart httpx
    pip install pillow scikit-image scikit-learn
    pip install ultralytics
fi

ok "Dépendances installées."

# ══════════════════════════════════════════════════════════════
# ÉTAPE 6 — Tests
# ══════════════════════════════════════════════════════════════
step "[6/7] Tests moteur..."

# MediaPipe
python -c "
import mediapipe as mp
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
fm.close()
print('[OK] MediaPipe actif')
" || {
    warn "MediaPipe 0.10.9 KO. Tentative force-reinstall..."
    pip install "mediapipe==0.10.9" --force-reinstall --no-cache-dir
    python -c "
import mediapipe as mp
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
fm.close()
print('[OK] MediaPipe actif apres reinstall')
" || warn "MediaPipe non actif — ArcFace+Texture uniquement"
}

# InsightFace
python -c "import insightface; print('[OK] InsightFace actif')" || warn "InsightFace KO"

# Ultralytics
python -c "import ultralytics; print('[OK] Ultralytics (oreilles) actif')" || warn "Ultralytics KO"

# ══════════════════════════════════════════════════════════════
# ÉTAPE 7 — Créer launch.sh et launch-api.sh
# ══════════════════════════════════════════════════════════════
step "[7/7] Création des scripts de lancement..."

cat > "$SCRIPT_DIR/launch.sh" << 'LAUNCH'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv_omni/bin/activate"
echo "[OK] OMNI-RECO pret"
echo "Usage : python compare.py photo_a.jpg photo_b.jpg --verbose"
exec bash
LAUNCH
chmod +x "$SCRIPT_DIR/launch.sh"
ok "launch.sh créé."

cat > "$SCRIPT_DIR/api/launch-api.sh" << 'LAUNCH_API'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/venv_omni/bin/activate"

# Vérif fastapi/uvicorn
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "[!] Installation fastapi/uvicorn..."
    pip install fastapi uvicorn python-multipart --quiet
}

# Générer token si absent
if [ ! -f "$SCRIPT_DIR/api/tokens.txt" ]; then
    echo "[TOKEN] Génération token automatique..."
    python "$SCRIPT_DIR/api/generate_token.py"
fi

echo ""
echo "============================================================"
echo " OMNI-RECO API disponible sur :"
echo "   http://localhost:8000"
echo "   http://localhost:8000/docs  (Swagger UI)"
echo "   http://localhost:8000/health"
echo " Arrêt : Ctrl+C"
echo "============================================================"
echo ""

cd "$SCRIPT_DIR"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
LAUNCH_API
chmod +x "$SCRIPT_DIR/api/launch-api.sh"
ok "api/launch-api.sh créé."

echo ""
echo "============================================================"
if [ "$ERRORS" -eq 0 ]; then
    echo "  TERMINÉ — Lancement :"
    echo "    CLI : bash launch.sh"
    echo "    API : bash api/launch-api.sh"
else
    echo "  TERMINÉ AVEC $ERRORS ERREUR(S) — Lis les messages ci-dessus"
fi
echo "============================================================"
echo ""
