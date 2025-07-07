# Setup Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 7+, macOS 10.10+, or Linux
- **Java**: JDK 8 or higher
- **RAM**: 2GB (4GB recommended for faster execution)
- **Disk Space**: 100MB

### Optional Requirements (for visualization)
- **Python**: 3.6 or higher
- **pip**: Python package manager

## Installation Steps

### 1. Install Java

#### Windows
1. Download JDK from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://adoptium.net/)
2. Run the installer
3. Add Java to PATH:
   - Right-click 'This PC' → Properties → Advanced System Settings
   - Click 'Environment Variables'
   - Add `JAVA_HOME` pointing to JDK installation
   - Add `%JAVA_HOME%\bin` to PATH

#### macOS
```bash
# Using Homebrew
brew install openjdk

# Or download from Oracle/OpenJDK website
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install default-jdk

# Fedora
sudo dnf install java-latest-openjdk-devel

# Arch
sudo pacman -S jdk-openjdk
```

### 2. Verify Java Installation
```bash
java -version
javac -version
```

### 3. Clone the Repository
```bash
git clone https://github.com/yourusername/Machine-Learning-Course-Work-CST-3170.git
cd Machine-Learning-Course-Work-CST-3170
```

### 4. (Optional) Setup Python for Visualization

#### Install Python
- Download from [python.org](https://www.python.org/downloads/)
- Or use package manager (brew, apt, etc.)

#### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project

### Quick Start
1. Open terminal/command prompt
2. Navigate to project directory
3. Run the appropriate script:

**Windows:**
```cmd
run_experiments.bat
```

**Linux/macOS:**
```bash
./run_experiments.sh
```

### Manual Compilation
If the scripts don't work, compile and run manually:

```bash
# Create output directory
mkdir -p out

# Compile all Java files
javac -d out src/*.java

# Run the main program
cd out
java Main
cd ..
```

### Visualizing Results (Optional)
After running experiments:
```bash
python visualize_results.py
```

## Troubleshooting

### Common Issues

#### "javac: command not found"
- Java is not installed or not in PATH
- Solution: Install Java and add to PATH

#### "Error: Could not find or load main class Main"
- Compilation failed or running from wrong directory
- Solution: Ensure you're in the `out` directory when running `java Main`

#### Permission denied (Linux/macOS)
- Scripts don't have execute permission
- Solution: `chmod +x run_experiments.sh`

#### Out of Memory Error
- Not enough heap space for larger experiments
- Solution: Run with more memory:
  ```bash
  java -Xmx2g Main
  ```

### Dataset Issues
- Ensure `dataSet1.csv` and `dataSet2.csv` are in the `datasets/` folder
- Files should be comma-separated with 65 columns (64 features + 1 label)

## IDE Setup (Optional)

### IntelliJ IDEA
1. Open IntelliJ IDEA
2. Select "Open" and choose the project directory
3. IntelliJ should auto-detect the project structure
4. Run `Main.java` using the green arrow

### Eclipse
1. File → Import → Existing Projects into Workspace
2. Select the project directory
3. Right-click `Main.java` → Run As → Java Application

### VS Code
1. Install Java Extension Pack
2. Open the project folder
3. Open `Main.java`
4. Click "Run" above the main method

## Performance Notes

- First run may take 1-5 minutes depending on system
- Results are saved to `results/` directory with timestamps
- Each experiment runs 2-fold cross-validation
- Total runtime depends on CPU speed and available memory

## Next Steps

After setup:
1. Run the experiments
2. Check `results/` folder for output
3. (Optional) Run `visualize_results.py` for charts
4. Modify classifiers or add new ones in `src/`
5. Adjust parameters in `Main.java` for different experiments