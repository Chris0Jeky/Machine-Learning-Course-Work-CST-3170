@echo off

echo ==============================================
echo Machine Learning Classifier Experiments Runner
echo ==============================================

REM Create necessary directories
echo Creating directories...
if not exist "out" mkdir out
if not exist "results" mkdir results

REM Compile all Java files
echo Compiling Java files...
javac -d out src\*.java

REM Check if compilation was successful
if %errorlevel% neq 0 (
    echo Compilation failed! Please check for errors.
    pause
    exit /b 1
)

echo Compilation successful!
echo.

REM Run the main experiments
echo Running experiments...
echo This may take a few minutes depending on your system.
echo.

REM Change to out directory and run Main class
cd out
java Main

REM Check if execution was successful
if %errorlevel% equ 0 (
    echo.
    echo ==============================================
    echo Experiments completed successfully!
    echo Check the 'results' directory for output files.
    echo ==============================================
) else (
    echo.
    echo Experiment execution failed!
    cd ..
    pause
    exit /b 1
)

REM Return to original directory
cd ..

pause