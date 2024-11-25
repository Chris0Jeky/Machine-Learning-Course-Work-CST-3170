public class MulticlassSVMClassifier implements Classifier {
    private int numClasses;
    private LinearSVMClassifier[] classifiers;
    private double learningRate;
    private double regularizationParam;
    private int epochs;
    private int featureSize;

    public MulticlassSVMClassifier(double learningRate, double regularizationParam, int epochs, int featureSize, int numClasses) {
        this.learningRate = learningRate;
        this.regularizationParam = regularizationParam;
        this.epochs = epochs;
        this.featureSize = featureSize;
        this.numClasses = numClasses;
        initializeClassifiers();
    }

    private void initializeClassifiers() {
        classifiers = new LinearSVMClassifier[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classifiers[i] = new LinearSVMClassifier(learningRate, regularizationParam, epochs, featureSize);
        }
    }

    @Override
    public void train(int[][] features, int[] labels) {
        initializeClassifiers(); // Reset classifiers for each fold
        int n = features.length;

        // For each class, train a binary classifier
        for (int c = 0; c < numClasses; c++) {
            int[] binaryLabels = new int[n];
            for (int i = 0; i < n; i++) {
                binaryLabels[i] = (labels[i] == c) ? 1 : -1;
            }
            classifiers[c].train(features, binaryLabels);
        }
    }

    @Override
    public int predict(int[] sample) {
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = -1;
        for (int c = 0; c < numClasses; c++) {
            double score = classifiers[c].decisionFunction(sample);
            if (score > maxScore) {
                maxScore = score;
                predictedClass = c;
            }
        }
        return predictedClass;
    }
}
