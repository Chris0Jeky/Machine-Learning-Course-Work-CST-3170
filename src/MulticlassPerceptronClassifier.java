//public class MulticlassPerceptronClassifier implements Classifier {
//    private int numClasses;
//    private double[][] weights;  // Weights for each class (each class has a weight vector)
//    private double[] biases; // Biases for each class
//    private int epochs; // Number of epochs (iterations over the training dataset)
//    private int featureSize;  // Size of feature vectors
//
//    // Constructor to initialize the classifier with the given parameters
//    public MulticlassPerceptronClassifier(int epochs, int featureSize, int numClasses) {
//        this.epochs = epochs;
//        this.featureSize = featureSize;
//        this.numClasses = numClasses;
//    }
//
//    @Override
//    public void train(int[][] features, int[] labels) {
//        initializeWeightsAndBiases(); // Initialize weights and biases to zero
//        int n = features.length; // Number of training samples
//
//        // Training loop for the specified number of epochs
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            for (int i = 0; i < n; i++) {
//                int[] x = features[i]; // Feature vector for the current sample
//                int y = labels[i]; // True label for the current sample
//
//                // Compute scores for all classes
//                double[] scores = new double[numClasses];
//                for (int c = 0; c < numClasses; c++) {
//                    scores[c] = dotProduct(weights[c], x) + biases[c];
//                }
//
//                // Predicted class is the one with the highest score
//                int predictedClass = argMax(scores);
//
//                // If prediction is incorrect, update weights and biases
//                if (predictedClass != y) {
//                    // Update weights and biases
//                    for (int j = 0; j < featureSize; j++) {
//                        // Decrease weight for wrong class
//                        weights[predictedClass][j] -= x[j];
//                        // Increase weight for correct class
//                        weights[y][j] += x[j];
//                    }
//                    // Adjust biases for the incorrect and correct classes
//                    biases[predictedClass] -= 1;
//                    biases[y] += 1;
//                }
//            }
//        }
//    }
//
//    // Helper method to initialize weights and biases to zero
//    private void initializeWeightsAndBiases() {
//        weights = new double[numClasses][featureSize];
//        biases = new double[numClasses];
//    }
//
//    @Override
//    public int predict(int[] sample) {
//        // Compute scores for each class
//        double[] scores = new double[numClasses];
//        for (int c = 0; c < numClasses; c++) {
//            scores[c] = dotProduct(weights[c], sample) + biases[c];
//        }
//        // Return the class with the highest score
//        return argMax(scores);
//    }
//
//// Helper method to find the index of the maximum score
//private int argMax(double[] scores) {
//        double maxScore = Double.NEGATIVE_INFINITY;
//        int maxIndex = -1;
//        for (int i = 0; i < scores.length; i++) {
//            if (scores[i] > maxScore) {
//                maxScore = scores[i];
//                maxIndex = i;
//            }
//        }
//        return maxIndex;
//    }
//
//    // Helper method to compute the dot product of a weight vector and a feature vector
//    private double dotProduct(double[] w, int[] x) {
//        double result = 0.0;
//        for (int i = 0; i < w.length; i++) {
//            result += w[i] * x[i];
//        }
//        return result;
//    }
//}
