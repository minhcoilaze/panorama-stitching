@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Khong tim thay Python Launcher. Hay cai Python 3.11 tu python.org.
    pause
    exit /b 1
)

py -3.11 -m venv .venv
if errorlevel 1 goto :error

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Cai dat thanh cong. Chay run_windows.bat de mo ung dung.
pause
exit /b 0

:error
echo.
echo Cai dat that bai. Kiem tra ket noi mang va phien ban Python 3.11.
pause
exit /b 1
