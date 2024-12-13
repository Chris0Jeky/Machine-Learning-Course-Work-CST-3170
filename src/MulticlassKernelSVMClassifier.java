//public class MulticlassKernelSVMClassifier implements Classifier {
//    private int numClasses; // Number of classes in the dataset
//    private KernelSVMClassifier[] classifiers;
//    // Hyperparameters for the Kernel SVM
//    private double C; // Regularization parameter
//    private double tol; // Tolerance for convergence
//    private int maxPasses; // Maximum number of iterations over the training set
//    private Kernel kernel; // Kernel function used for SVM
//
//    // Constructor to initialize the multiclass classifier with given hyperparameters
//    public MulticlassKernelSVMClassifier(double C, double tol, int maxPasses, Kernel kernel, int numClasses) {
//        this.C = C;
//        this.tol = tol;
//        this.maxPasses = maxPasses;
//        this.kernel = kernel;
//        this.numClasses = numClasses;
//        initializeClassifiers();
//    }
//
//    // Helper method to initialize the array of KernelSVMClassifiers
//    private void initializeClassifiers() {
//        classifiers = new KernelSVMClassifier[numClasses];
//        for (int i = 0; i < numClasses; i++) {
//            classifiers[i] = new KernelSVMClassifier(C, tol, maxPasses, kernel, numClasses);
//        }
//    }
//
//    @Override
//    public void train(int[][] features, int[] labels) {
//        // Reset classifiers before training
//        initializeClassifiers(); // Reset
//        int n = features.length;
//
//        // Train each binary classifier in a One-vs-Rest approach
//        for (int c = 0; c < numClasses; c++) {
//            int[] binaryLabels = new int[n];
//            for (int i = 0; i < n; i++) {
//                // Assign label +1 for the current class and -1 for all other classes
//                binaryLabels[i] = (labels[i] == c) ? 1 : -1;
//            }
//            // Train the binary classifier for the current class
//            classifiers[c].train(features, binaryLabels);
//        }
//    }
//
//    @Override
//    public int predict(int[] sample) {
//        // Predict by picking the class with highest decision function value
//        double maxScore = Double.NEGATIVE_INFINITY;
//        int predictedClass = -1;
//
//        for (int c = 0; c < numClasses; c++) {
//            // Compute the decision function value for the current class
//            double score = classifiers[c].decisionFunction(sample);
//            if (score > maxScore) {
//                // Update the predicted class if the current score is higher
//                maxScore = score;
//                predictedClass = c;
//            }
//        }
//        // Return the class with the highest decision function value
//        return predictedClass;
//    }
//}
