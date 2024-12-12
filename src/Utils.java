import java.util.Arrays;

public class Utils {
    // Extract features (first 64 columns)
    public static int[][] extractFeatures(int[][] data) {
        int[][] features = new int[data.length][64];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, features[i], 0, 64);
        }
        return features;
    }

    // Extract labels (last column)
    public static int[] extractLabels(int[][] data) {
        int[] labels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            labels[i] = data[i][64];
        }
        return labels;
    }

    public static int getMaxLabel(int[] labels) {
        int max = Integer.MIN_VALUE;
        for (int label : labels) {
            if (label > max) {
                max = label;
            }
        }
        return max;
    }

    // Get maximum label from both label arrays
    public static int getMaxLabel(int[] labels1, int[] labels2) {
        int max = Integer.MIN_VALUE;
        for (int label : labels1) {
            if (label > max) {
                max = label;
            }
        }
        for (int label : labels2) {
            if (label > max) {
                max = label;
            }
        }
        return max;
    }

    // Feature scaling to [0, 1] using min-max normalization
    public static void scaleFeatures(int[][] trainFeatures, int[][] testFeatures) {
        int numFeatures = trainFeatures[0].length;

        // Find min and max for each feature in the training set
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];
        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);

        for (int[] sample : trainFeatures) {
            for (int i = 0; i < numFeatures; i++) {
                if (sample[i] < minValues[i]) {
                    minValues[i] = sample[i];
                }
                if (sample[i] > maxValues[i]) {
                    maxValues[i] = sample[i];
                }
            }
        }

        // Scale training features
        for (int[] sample : trainFeatures) {
            for (int i = 0; i < numFeatures; i++) {
                if (maxValues[i] != minValues[i]) {
                    sample[i] = (int) ((sample[i] - minValues[i]) / (maxValues[i] - minValues[i]) * 255);
                } else {
                    sample[i] = 0;
                }
            }
        }

        // Scale test features using training min and max
        for (int[] sample : testFeatures) {
            for (int i = 0; i < numFeatures; i++) {
                if (maxValues[i] != minValues[i]) {
                    sample[i] = (int) ((sample[i] - minValues[i]) / (maxValues[i] - minValues[i]) * 255);
                } else {
                    sample[i] = 0;
                }
            }
        }
    }

    // Compute centroids for each class
    public static double[][] computeCentroids(int[][] features, int[] labels, int numClasses) {
        int numFeatures = features[0].length;
        double[][] centroids = new double[numClasses][numFeatures];
        int[] counts = new int[numClasses];

        // Sum features for each class
        for (int i = 0; i < features.length; i++) {
            int label = labels[i];
            for (int j = 0; j < numFeatures; j++) {
                centroids[label][j] += features[i][j];
            }
            counts[label]++;
        }

        // Compute average
        for (int c = 0; c < numClasses; c++) {
            if (counts[c] > 0) {
                for (int j = 0; j < numFeatures; j++) {
                    centroids[c][j] /= counts[c];
                }
            }
        }
        return centroids;
    }

    // Add centroid distance features
    public static int[][] addCentroidFeatures(int[][] features, double[][] centroids) {
        int n = features.length;
        int originalFeatureSize = features[0].length;
        int numClasses = centroids.length;
        int[][] newFeatures = new int[n][originalFeatureSize + numClasses];

        for (int i = 0; i < n; i++) {
            // Copy original features
            System.arraycopy(features[i], 0, newFeatures[i], 0, originalFeatureSize);

            // Compute distances to centroids
            for (int c = 0; c < numClasses; c++) {
                double distance = euclideanDistance(features[i], centroids[c]);
                newFeatures[i][originalFeatureSize + c] = (int) distance;
            }
        }
        return newFeatures;
    }

    // Euclidean distance between feature vector and centroid
    public static double euclideanDistance(int[] x1, double[] x2) {
        double sum = 0.0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    public static int[][] deepCopy(int[][] original) {
        int[][] copy = new int[original.length][];
        for (int i = 0; i < original.length; i++) {
            copy[i] = Arrays.copyOf(original[i], original[i].length);
        }
        return copy;
    }

}
