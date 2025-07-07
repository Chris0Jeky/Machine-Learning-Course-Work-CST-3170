#!/bin/bash

# Machine Learning Classifier Experiments Runner
# This script compiles and runs all ML experiments

echo "=============================================="
echo "Machine Learning Classifier Experiments Runner"
echo "=============================================="

# Create necessary directories
echo "Creating directories..."
mkdir -p out
mkdir -p results

# Compile all Java files
echo "Compiling Java files..."
javac -d out src/*.java

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed! Please check for errors."
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run the main experiments
echo "Running experiments..."
echo "This may take a few minutes depending on your system."
echo ""

# Change to out directory and run Main class
cd out
java Main

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Experiments completed successfully!"
    echo "Check the 'results' directory for output files."
    echo "=============================================="
else
    echo ""
    echo "Experiment execution failed!"
    exit 1
fi

# Return to original directory
cd ..