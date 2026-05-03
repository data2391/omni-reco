@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
cmd /k "%~f0" RUN
exit /b
)

title OMNI-RECO v2 - Installation
echo.
echo ============================================================
echo OMNI-RECO v2 - Setup Automatique
echo ============================================================
echo.

set PYTHON_EXE=
set VENV_NAME=venv_omni
set SCRIPT_DIR=%~dp0
set ERRORS=0
set DL_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
set PYTHON_INSTALLER=%TEMP%\py3010setup.exe

echo [1/6] Recherche de Python 3.10...

py -3.10 --version >nul 2>nul
if !errorlevel! == 0 (
set PYTHON_EXE=py -3.10
echo [OK] Python 3.10 detecte via py launcher
goto :venv_check
)

python3.10 --version >nul 2>nul
if !errorlevel! == 0 (
set PYTHON_EXE=python3.10
echo [OK] Python 3.10 detecte
goto :venv_check
)

echo [!] Python 3.10 absent. Telechargement...

where curl >nul 2>nul
if !errorlevel! == 0 (
curl -L --progress-bar -o "%PYTHON_INSTALLER%" "%DL_URL%"
) else (
powershell -NoProfile -Command "(New-Object Net.WebClient).DownloadFile('%DL_URL%','%PYTHON_INSTALLER%')"
)

if not exist "%PYTHON_INSTALLER%" (
echo [ERREUR] Telechargement echoue.
echo Telecharge manuellement Python 3.10.11 puis relance :
echo %DL_URL%
set ERRORS=1
goto :end
)

echo Installation Python 3.10.11 en cours...
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0
timeout /t 6 /nobreak >nul

py -3.10 --version >nul 2>nul
if !errorlevel! neq 0 (
echo [ERREUR] Python 3.10 non reconnu apres installation.
echo Ferme ce terminal, rouvre-le et relance setup.bat
set ERRORS=1
goto :end
)
set PYTHON_EXE=py -3.10
echo [OK] Python 3.10.11 installe.

:venv_check
echo.
echo [2/6] Environnement virtuel...
cd /d "%SCRIPT_DIR%"

if exist "%VENV_NAME%\Scripts\activate.bat" (
    rem === VERIFIER QUE LE VENV EST BIEN EN PYTHON 3.10 ===
    "%SCRIPT_DIR%%VENV_NAME%\Scripts\python.exe" -c "import sys; v=sys.version_info; exit(0 if v.major==3 and v.minor==10 else 1)" >nul 2>nul
    if !errorlevel! == 0 (
        echo [OK] Venv Python 3.10 existant - reutilise.
    ) else (
        echo [!] Venv existant N'EST PAS Python 3.10 - suppression et recreation...
        rmdir /s /q "%SCRIPT_DIR%%VENV_NAME%"
        echo [OK] Ancien venv supprime.
        echo Creation venv Python 3.10...
        %PYTHON_EXE% -m venv "%VENV_NAME%"
        if !errorlevel! neq 0 (
            echo [ERREUR] Creation du venv echouee.
            set ERRORS=1
            goto :end
        )
        echo [OK] Venv Python 3.10 cree.
    )
) else (
    echo Creation venv...
    %PYTHON_EXE% -m venv "%VENV_NAME%"
    if !errorlevel! neq 0 (
        echo [ERREUR] Creation du venv echouee.
        set ERRORS=1
        goto :end
    )
    echo [OK] Venv cree.
)

echo.
echo [3/6] Activation du venv...
call "%SCRIPT_DIR%%VENV_NAME%\Scripts\activate.bat"
echo [OK] Venv actif.

rem Verification finale version Python dans le venv
python -c "import sys; print(f'[OK] Python {sys.version} dans le venv')"

echo.
echo [4/6] Mise a jour pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip OK.

echo.
echo [5/6] Installation dependances...
echo (5 a 10 minutes selon connexion)
echo.

if exist "%SCRIPT_DIR%requirements_v2.txt" (
echo Utilisation requirements_v2.txt...
python -m pip install -r "%SCRIPT_DIR%requirements_v2.txt"
) else (
python -m pip install "mediapipe==0.10.9" --no-cache-dir
python -m pip install insightface
python -m pip install onnxruntime
python -m pip install opencv-python
python -m pip install numpy scipy rich
python -m pip install fastapi uvicorn python-multipart httpx
python -m pip install pillow scikit-image scikit-learn
)
echo [OK] Dependances OK.

echo.
echo [6/6] Test MediaPipe...
python -c "import mediapipe as mp; fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=True); fm.close(); print('[OK] MediaPipe actif')"
if !errorlevel! neq 0 (
    echo [!] Tentative downgrade force mediapipe 0.10.9...
    python -m pip install "mediapipe==0.10.9" --force-reinstall --no-cache-dir
    python -c "import mediapipe as mp; fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=True); fm.close(); print('[OK] MediaPipe actif apres reinstall')"
    if !errorlevel! neq 0 (
        echo [!] MediaPipe non actif - ArcFace+Texture uniquement
    ) else (
        echo [OK] Moteur complet actif
    )
) else (
    echo [OK] Moteur complet actif
)

(
echo @echo off
echo title OMNI-RECO v2
echo cd /d "%%~dp0"
echo call venv_omni\Scripts\activate
echo echo [OK] OMNI-RECO pret
echo echo Usage : python compare.py photo_a.jpg photo_b.jpg --verbose
echo cmd /k
) > "%SCRIPT_DIR%launch.bat"
echo [OK] launch.bat cree.

:end
echo.
echo ============================================================
if !ERRORS! == 0 (
echo TERMINE - Double-clique sur launch.bat pour demarrer
) else (
echo ECHEC - Lis les erreurs ci-dessus
)
echo ============================================================
echo.
pause
endlocal
