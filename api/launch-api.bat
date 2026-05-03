@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    cmd /k "%~f0" RUN
    exit /b
)

title OMNI-RECO API v1.0
cd /d "%~dp0.."

echo.
echo ============================================================
echo  OMNI-RECO API v1.0 - Demarrage
echo ============================================================
echo.

REM --- Activation venv ---
if not exist "venv_omni\Scripts\activate.bat" (
    echo [ERREUR] venv_omni introuvable.
    echo Lance d'abord setup.bat depuis le dossier racine.
    pause
    exit /b 1
)

call venv_omni\Scripts\activate.bat
echo [OK] Venv actif.

REM --- Verif fastapi/uvicorn ---
python -c "import fastapi, uvicorn" >nul 2>nul
if !errorlevel! neq 0 (
    echo [!] fastapi/uvicorn manquants. Installation...
    python -m pip install fastapi uvicorn python-multipart --quiet
    echo [OK] fastapi + uvicorn installes.
)

REM --- Verif ultralytics ---
python -c "import ultralytics" >nul 2>nul
if !errorlevel! neq 0 (
    echo [!] ultralytics manquant. Installation...
    python -m pip install ultralytics --quiet
    echo [OK] ultralytics installe - Oreilles actives.
) else (
    echo [OK] ultralytics present - Oreilles actives.
)

REM --- Generer token si tokens.txt absent ---
if not exist "api\tokens.txt" (
    echo.
    echo [TOKEN] Aucun token detecte. Generation automatique...
    python api\generate_token.py
    echo.
)

REM --- Afficher token(s) existants (hash seulement, securite) ---
echo.
echo [INFO] Tokens enregistres : 
for /f "tokens=*" %%i in (api\tokens.txt) do (
    echo   - %%i
)
echo [INFO] Pour generer un nouveau token : python api\generate_token.py
echo.

REM --- Lancement serveur ---
echo ============================================================
echo  Serveur API disponible sur :
echo    http://localhost:8000
echo    http://localhost:8000/docs  (Swagger UI)
echo    http://localhost:8000/health
echo.
echo  Arret : Ctrl+C
echo ============================================================
echo.

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
endlocal
