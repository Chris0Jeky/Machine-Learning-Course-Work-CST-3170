//public class KNearestNeighborsClassifier implements Classifier {
//    // A basic k-NN classifier without fancy data structures.
//    // Distances are computed at prediction time, sorted, and the majority label chosen.
//    private int[][] trainingFeatures;
//    private int[] trainingLabels;
//    private int k;
//    private int numClasses;
//
//    public KNearestNeighborsClassifier(int k, int numClasses) {
//        this.k = k;
//        this.numClasses = numClasses;
//    }
//
//    @Override
//    public void train(int[][] features, int[] labels) {
//        // Just store training data, no actual model training for k-NN.
//        this.trainingFeatures = features;
//        this.trainingLabels = labels;
//    }
//
//    @Override
//    public int predict(int[] testImage) {
//        int n = trainingFeatures.length;
//        double[] distances = new double[n];
//        int[] labels = new int[n];
//
//        // Compute distances to all training samples
//        for (int i = 0; i < n; i++) {
//            distances[i] = DistanceCalculator.euclideanDistance(testImage, trainingFeatures[i]);
//            labels[i] = trainingLabels[i];
//        }
//
//        // Selection sort for the first k neighbors
//        for (int i = 0; i < k; i++) {
//            // Find the index of the minimum distance
//            int minIndex = i;
//            for (int j = i + 1; j < n; j++) {
//                if (distances[j] < distances[minIndex]) {
//                    minIndex = j;
//                }
//            }
//            // Swap to bring nearest neighbor into top k range
//            double tempDist = distances[i];
//            distances[i] = distances[minIndex];
//            distances[minIndex] = tempDist;
//
//            // Swap labels[i] and labels[minIndex]
//            int tempLabel = labels[i];
//            labels[i] = labels[minIndex];
//            labels[minIndex] = tempLabel;
//        }
//
//        // Count occurrences among the k nearest
//        int[] labelCounts = new int[numClasses];
//        for (int i = 0; i < k; i++) {
//            int label = labels[i];
//            if (label >= 0 && label < numClasses) {
//                labelCounts[label]++;
//            } else {
//                System.err.println("Warning: Label out of bounds: " + label);
//            }
//        }
//
//        // Determine class with maximum count
//        int predictedLabel = -1;
//        int maxCount = -1;
//        for (int i = 0; i < numClasses; i++) {
//            if (labelCounts[i] > maxCount) {
//                maxCount = labelCounts[i];
//                predictedLabel = i;
//            }
//        }
//
//        // If tie, choose label with smallest cumulative distance among tied labels
//        int numMaxLabels = 0;
//        for (int i = 0; i < numClasses; i++) {
//            if (labelCounts[i] == maxCount) {
//                numMaxLabels++;
//            }
//        }
//
//        if (numMaxLabels == 1) {
//            // Unique label with max count
//            return predictedLabel;
//        } else {
//            // Tie occurred
//            // Among the tied labels, choose the one with the smallest cumulative distance
//            double[] cumulativeDistances = new double[numClasses];
//            for (int i = 0; i < k; i++) {
//                int label = labels[i];
//                if (labelCounts[label] == maxCount) {
//                    cumulativeDistances[label] += distances[i];
//                }
//            }
//
//            double minCumulativeDistance = Double.MAX_VALUE;
//            for (int i = 0; i < numClasses; i++) {
//                if (labelCounts[i] == maxCount && cumulativeDistances[i] < minCumulativeDistance) {
//                    minCumulativeDistance = cumulativeDistances[i];
//                    predictedLabel = i;
//                }
//            }
//            return predictedLabel;
//        }
//    }
//}
