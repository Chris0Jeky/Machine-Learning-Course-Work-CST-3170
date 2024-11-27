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

    // Feature scaling to [0, 1]
    public static void scaleFeatures(int[][] features) {
        int numFeatures = features[0].length;
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];

        // Initialize min and max values
        Arrays.fill(minValues, Double.MAX_VALUE);
        Arrays.fill(maxValues, Double.MIN_VALUE);

        // Find min and max for each feature
        for (int[] sample : features) {
            for (int i = 0; i < numFeatures; i++) {
                if (sample[i] < minValues[i]) {
                    minValues[i] = sample[i];
                }
                if (sample[i] > maxValues[i]) {
                    maxValues[i] = sample[i];
                }
            }
        }

        // Scale features
        for (int[] sample : features) {
            for (int i = 0; i < numFeatures; i++) {
                if (maxValues[i] != minValues[i]) {
                    sample[i] = (int) ((sample[i] - minValues[i]) / (maxValues[i] - minValues[i]));
                } else {
                    sample[i] = 0;
                }
            }
        }
    }
}
