public class MulticlassSVMClassifier implements Classifier {
    private int numClasses;
    private LinearSVMClassifier[] classifiers;
    private double learningRate; // Learning rate for the SVM optimization
    private double regularizationParam; // Regularization parameter to control overfitting
    private int epochs;
    private int featureSize;

    public MulticlassSVMClassifier(double learningRate, double regularizationParam, int epochs, int featureSize, int numClasses) {
        this.learningRate = learningRate;
        this.regularizationParam = regularizationParam;
        this.epochs = epochs;
        this.featureSize = featureSize;
        this.numClasses = numClasses;
        initializeClassifiers(); // Initialize the array of binary classifiers
    }

    // Helper method to initialize the binary classifiers
    private void initializeClassifiers() {
        classifiers = new LinearSVMClassifier[numClasses];
        for (int i = 0; i < numClasses; i++) {
            // Each LinearSVMClassifier handles one class in a One-vs-Rest manner
            classifiers[i] = new LinearSVMClassifier(learningRate, regularizationParam, epochs, featureSize);
        }
    }

    @Override
    public void train(int[][] features, int[] labels) {
        initializeClassifiers(); // Reset classifiers for each fold
        int n = features.length; // Number of training

        // For each class, train a binary classifier
        for (int c = 0; c < numClasses; c++) {
            int[] binaryLabels = new int[n];
            for (int i = 0; i < n; i++) {
                // Assign +1 for the current class and -1 for all other classes
                binaryLabels[i] = (labels[i] == c) ? 1 : -1;
            }
            // Train the binary classifier for class c
            classifiers[c].train(features, binaryLabels);
        }
    }

    @Override
    public int predict(int[] sample) {
        // Predict the class by selecting the classifier with the highest decision function score
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = -1;
        for (int c = 0; c < numClasses; c++) {
            // Compute the decision function score for the current class
            double score = classifiers[c].decisionFunction(sample);
            if (score > maxScore) {
                // Update the predicted class if the score is the highest so far
                maxScore = score;
                predictedClass = c;
            }
        }
        // Return the class with the highest decision function score
        return predictedClass;
    }
}
