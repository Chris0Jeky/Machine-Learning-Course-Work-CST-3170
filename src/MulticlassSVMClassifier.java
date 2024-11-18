public class MulticlassSVMClassifier implements Classifier {
    private int numClasses;
    private LinearSVMClassifier[] classifiers;

    public MulticlassSVMClassifier(double learningRate, double regularizationParam, int epochs, int featureSize, int numClasses) {
        this.numClasses = numClasses;
        this.classifiers = new LinearSVMClassifier[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classifiers[i] = new LinearSVMClassifier(learningRate, regularizationParam, epochs, featureSize);
        }
    }

    @Override
    public void train(int[][] features, int[] labels) {
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
