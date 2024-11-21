import java.util.Random;

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

    // Combine two datasets
    public static int[][] combineDatasets(int[][] dataSet1, int[][] dataSet2) {
        int totalLength = dataSet1.length + dataSet2.length;
        int[][] combinedData = new int[totalLength][];
        System.arraycopy(dataSet1, 0, combinedData, 0, dataSet1.length);
        System.arraycopy(dataSet2, 0, combinedData, dataSet1.length, dataSet2.length);
        return combinedData;
    }

    // Shuffle the dataset
    public static void shuffleDataset(int[][] data) {
        Random rand = new Random();
        for (int i = data.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int[] temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    // Split dataset into k folds
    public static int[][][] splitIntoFolds(int[][] features, int[] labels, int k) {
        int n = features.length;
        int foldSize = n / k;
        int[][][] folds = new int[k][2][]; // Each fold has [features][labels]

        for (int fold = 0; fold < k; fold++) {
            int start = fold * foldSize;
            int end = (fold == k - 1) ? n : start + foldSize;

            int[][] foldFeatures = new int[end - start][];
            int[] foldLabels = new int[end - start];

            System.arraycopy(features, start, foldFeatures, 0, end - start);
            System.arraycopy(labels, start, foldLabels, 0, end - start);

            folds[fold][0] = foldFeatures;
            folds[fold][1] = foldLabels;
        }
        return folds;
    }
}
