# Machine Learning Classifier Comparison

A comprehensive Java implementation of various machine learning algorithms for handwritten digit recognition, developed as coursework for CST 3170.

## Overview

This project implements and compares 15+ different machine learning classifiers on a handwritten digit recognition task using 8x8 grayscale images. The implementation demonstrates fundamental ML concepts including:

- **Instance-based learning** (k-NN variants)
- **Neural networks** (MLP, Perceptron)
- **Support Vector Machines** (Linear, Kernel-based)
- **Tree-based methods** (Decision Trees, Random Forest, Gradient Boosting)
- **Ensemble methods** (Voting, Hybrid classifiers)

## Features

- **Pure Java Implementation**: All algorithms implemented from scratch without external ML libraries
- **Comprehensive Evaluation**: 2-fold cross-validation with detailed performance metrics
- **Modular Architecture**: Clean interface-based design for easy extension
- **Automated Experiments**: One-click script to run all experiments
- **Result Logging**: Automatic saving of results with timestamps

## Implemented Classifiers

### Instance-Based Learning
- **1-Nearest Neighbor**: Basic nearest neighbor classifier
- **k-Nearest Neighbors**: Configurable k parameter (k=3,5,7)
- **Weighted k-NN**: Distance-weighted voting mechanism

### Neural Networks
- **Multi-Layer Perceptron (MLP)**: 
  - Single hidden layer with ReLU activation
  - Softmax output layer
  - Backpropagation training
- **Multi-class Perceptron**: Direct multi-class classification

### Support Vector Machines
- **Linear SVM**: One-vs-All approach for multi-class
- **SVM with Centroid Features**: Enhanced feature space
- **RBF Kernel SVM**: Non-linear classification using Radial Basis Function

### Tree-Based Methods
- **Decision Tree**: Entropy-based splitting criteria
- **Random Forest**: Ensemble of 10 trees with feature randomization
- **Gradient Boosted Trees**: 50 trees with softmax for multi-class

### Ensemble Methods
- **Voting Classifier**: Combines NN, MLP, and Weighted k-NN
- **Hybrid Classifier**: Switches between NN and MLP based on distance threshold

## Dataset

The project uses two datasets (`dataSet1.csv` and `dataSet2.csv`) containing:
- **64 features**: Pixel values from 8x8 grayscale images
- **10 classes**: Digits 0-9
- **Format**: CSV with features in columns 1-64, labels in column 65

## Getting Started

### Prerequisites
- Java Development Kit (JDK) 8 or higher
- Command line terminal (bash/cmd)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Machine-Learning-Course-Work-CST-3170.git
   cd Machine-Learning-Course-Work-CST-3170
   ```

2. Ensure datasets are in the `datasets/` directory

### Running Experiments

#### Windows:
```bash
run_experiments.bat
```

#### Linux/Mac:
```bash
./run_experiments.sh
```

The script will:
1. Create necessary directories (`out/`, `results/`)
2. Compile all Java files
3. Run all experiments
4. Save results to `results/experiment_results_[timestamp].txt`

### Manual Compilation and Execution
```bash
# Compile
javac -d out src/*.java

# Run
cd out
java Main
```

## Project Structure

```
Machine-Learning-Course-Work-CST-3170/
├── src/                          # Source code
│   ├── Classifier.java          # Base interface
│   ├── Main.java               # Experiment runner
│   ├── DataLoader.java         # CSV data loading
│   ├── Utils.java              # Utility functions
│   ├── DistanceCalculator.java # Distance metrics
│   ├── NearestNeighborClassifier.java
│   ├── KNearestNeighborsClassifier.java
│   ├── WeightedKNearestNeighborsClassifier.java
│   ├── MLPClassifier.java
│   ├── MulticlassPerceptronClassifier.java
│   ├── LinearSVMClassifier.java
│   ├── MulticlassSVMClassifier.java
│   ├── KernelSVMClassifier.java
│   ├── MulticlassKernelSVMClassifier.java
│   ├── LinearKernel.java
│   ├── RBFKernel.java
│   ├── DecisionTreeClassifier.java
│   ├── SimpleDecisionTreeClassifier.java
│   ├── RandomForestClassifier.java
│   ├── GradientBoostedTreesClassifier.java
│   ├── MultiClassGradientBoostedTreesClassifier.java
│   ├── SimpleVotingClassifier.java
│   └── HybridClassifier.java
├── datasets/                    # Data files
│   ├── dataSet1.csv
│   └── dataSet2.csv
├── results/                     # Output directory (created on run)
├── run_experiments.sh          # Linux/Mac runner
├── run_experiments.bat         # Windows runner
└── README.md                   # This file
```

## Performance Metrics

Each classifier is evaluated on:
- **Accuracy**: Percentage of correct predictions
- **Training Time**: Time to train the model (ms)
- **Evaluation Time**: Time to make predictions (ms)
- **Confusion Matrix**: Detailed classification results

Results are automatically ranked by accuracy with a summary showing:
- Best and worst performers
- Performance gap analysis
- Fastest training algorithm

## Example Output

```
========== NEURAL NETWORK CLASSIFIERS ==========

--- MLP (100 hidden units) ---

Fold 1:
  Training...
  Training time: 2341 ms
  Evaluating...
  Accuracy: 97.85%
  Evaluation time: 45 ms

Average Results:
  Average Accuracy: 97.32%
  Average Training Time: 2298 ms
  Average Evaluation Time: 42 ms
```

## Extending the Project

To add a new classifier:

1. Create a new class implementing the `Classifier` interface:
```java
public class MyClassifier implements Classifier {
    @Override
    public void train(int[][] features, int[] labels) {
        // Training logic
    }
    
    @Override
    public int predict(int[] features) {
        // Prediction logic
        return predictedLabel;
    }
}
```

2. Add an experiment in `Main.java`:
```java
runExperiment("My Classifier", 
    (features, labels, numClasses) -> new MyClassifier());
```

## Key Algorithms and Techniques

### Feature Engineering
- **Centroid Features**: Distances to class centroids added as features
- **Min-Max Scaling**: Normalization support in Utils class

### Distance Metrics
- **Euclidean Distance**: Standard L2 distance

### Neural Network Features
- **ReLU Activation**: For hidden layers
- **Softmax Output**: For probability distribution
- **Cross-Entropy Loss**: For multi-class classification
- **Mini-batch Training**: Efficient gradient updates

### Tree-Based Features
- **Entropy-based Splitting**: Information gain criteria
- **Bootstrap Sampling**: For Random Forest
- **Feature Randomization**: Reduces overfitting
- **Gradient Boosting**: Sequential error correction

## Contributing

This is a coursework project, but suggestions and improvements are welcome:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is developed for educational purposes as part of CST 3170 coursework.

## Acknowledgments

- Course: CST 3170 - Machine Learning
- Dataset: Simplified handwritten digit recognition (8x8 pixels)
- Implementation: All algorithms implemented from scratch for learning purposes