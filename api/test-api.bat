@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    cmd /k "%~f0" RUN
    exit /b
)

title OMNI-RECO API - Test
cd /d "%~dp0.."

echo.
echo ============================================================
echo  OMNI-RECO API - Test rapide
echo ============================================================
echo.

REM --- Verif curl ---
where curl >nul 2>nul
if !errorlevel! neq 0 (
    echo [ERREUR] curl absent. Installe curl ou utilise PowerShell.
    pause
    exit /b 1
)

REM --- Lire le premier token (hash) depuis tokens.txt ---
set TOKEN=
if not exist "api\tokens.txt" (
    echo [ERREUR] api\tokens.txt introuvable.
    echo Lance generate_token.py d'abord.
    pause
    exit /b 1
)

REM Le fichier tokens.txt contient des HASH - pour tester il faut le token en clair
REM Ce script teste uniquement le endpoint /health sans auth
echo [1/2] Test endpoint /health (sans auth)...
curl -s http://localhost:8000/health
echo.
echo.

REM --- Test /compare avec deux photos passees en argument ---
if "%~2"=="" (
    echo [INFO] Usage pour tester /compare :
    echo   test-api.bat RUN photo_a.jpg photo_b.jpg VOTRE_TOKEN
    echo.
    echo [INFO] Exemple :
    echo   test-api.bat RUN "C:\photo1.jpg" "C:\photo2.jpg" abc123...
    goto :end
)

set PHOTO_A=%~2
set PHOTO_B=%~3
set BEARER_TOKEN=%~4

if "!BEARER_TOKEN!"=="" (
    echo [ERREUR] Token manquant. Usage :
    echo   test-api.bat RUN photo_a.jpg photo_b.jpg VOTRE_TOKEN
    goto :end
)

echo [2/2] Test endpoint /compare...
curl -s -X POST http://localhost:8000/compare ^
  -H "Authorization: Bearer !BEARER_TOKEN!" ^
  -F "photo_a=@!PHOTO_A!" ^
  -F "photo_b=@!PHOTO_B!"
echo.

:end
echo.
echo ============================================================
echo  Swagger UI : http://localhost:8000/docs
echo ============================================================
pause
endlocal
