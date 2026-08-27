@echo off
setlocal
cd /d "%~dp0"

set "BENCHMARK_EXE=%CD%\StripeCenterlineBenchmark.exe"
set "SOURCE_ROOT=%CD%\source"
set "VENV_DIR=%CD%\.venv-windows-benchmark"
set "RESULTS_ROOT=%CD%\windows_ab_results"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

echo Stripe Centerline Windows EXE vs Python A/B benchmark
echo.

if not exist "%BENCHMARK_EXE%" (
    echo ERROR: StripeCenterlineBenchmark.exe was not found.
    goto :failed
)
if not exist "%SOURCE_ROOT%\tools\windows_benchmark_runner.py" (
    echo ERROR: The bundled source folder was not found.
    goto :failed
)

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python Launcher was not found.
    echo Install 64-bit Python 3.11.9 from python.org, then run this file again.
    goto :failed
)

set "FOUND_PYTHON="
for /f "usebackq delims=" %%V in (`py -3.11 -c "import platform; print(platform.python_version())"`) do set "FOUND_PYTHON=%%V"
if not "%FOUND_PYTHON%"=="3.11.9" (
    echo ERROR: Python 3.11.9 is required; found %FOUND_PYTHON%.
    echo Install 64-bit Python 3.11.9 from python.org, then run this file again.
    goto :failed
)

if not exist "%PYTHON_EXE%" (
    echo Creating isolated Python environment...
    py -3.11 -m venv "%VENV_DIR%"
    if errorlevel 1 goto :failed
)

echo Installing the locked benchmark wheels...
"%PYTHON_EXE%" -m pip install --disable-pip-version-check --only-binary=:all: -r "%SOURCE_ROOT%\requirements-windows-benchmark.txt"
if errorlevel 1 goto :failed

echo.
echo Running frozen EXE: 5 fixed-point runs...
start "" /wait "%BENCHMARK_EXE%" --no-dialog --output-dir "%RESULTS_ROOT%\frozen_exe"
if errorlevel 1 goto :failed

echo Running Python source: 5 fixed-point runs...
"%PYTHON_EXE%" "%SOURCE_ROOT%\tools\windows_benchmark_runner.py" --no-dialog --output-dir "%RESULTS_ROOT%\python_source"
if errorlevel 1 goto :failed

echo Creating A/B summary...
"%PYTHON_EXE%" "%SOURCE_ROOT%\tools\compare_windows_benchmarks.py" --exe "%RESULTS_ROOT%\frozen_exe\benchmark_results.json" --python-source "%RESULTS_ROOT%\python_source\benchmark_results.json" --output-dir "%RESULTS_ROOT%"
if errorlevel 1 goto :comparison_failed

echo.
type "%RESULTS_ROOT%\ab_summary.txt"
echo.
echo Complete. Results are in:
echo %RESULTS_ROOT%
pause
exit /b 0

:comparison_failed
echo.
echo Both benchmarks finished, but their identity checks did not all pass.
echo Please send the entire windows_ab_results folder for diagnosis.
pause
exit /b 2

:failed
echo.
echo Benchmark setup or execution failed.
echo If windows_ab_results exists, please send that folder with a screenshot.
pause
exit /b 1
