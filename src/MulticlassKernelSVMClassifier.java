public class MulticlassKernelSVMClassifier implements Classifier {
    private int numClasses;
    private KernelSVMClassifier[] classifiers;
    private double C;
    private double tol;
    private int maxPasses;
    private Kernel kernel;

    public MulticlassKernelSVMClassifier(double C, double tol, int maxPasses, Kernel kernel, int numClasses) {
        this.C = C;
        this.tol = tol;
        this.maxPasses = maxPasses;
        this.kernel = kernel;
        this.numClasses = numClasses;
        initializeClassifiers();
    }

    private void initializeClassifiers() {
        classifiers = new KernelSVMClassifier[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classifiers[i] = new KernelSVMClassifier(C, tol, maxPasses, kernel, numClasses);
        }
    }

    @Override
    public void train(int[][] features, int[] labels) {
        initializeClassifiers(); // Reset
        int n = features.length;

        // One-vs-Rest approach for multiclass
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
        // Predict by picking the class with highest decision function value
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
