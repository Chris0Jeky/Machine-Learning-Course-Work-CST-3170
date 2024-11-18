public class KNearestNeighborsClassifier implements Classifier {
    private int[][] trainingFeatures;
    private int[] trainingLabels;
    private int k;

    public KNearestNeighborsClassifier(int k) {
        this.k = k;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        this.trainingFeatures = features;
        this.trainingLabels = labels;
    }

    @Override
    public int predict(int[] testImage) {
        int n = trainingFeatures.length;
        double[] distances = new double[n];
        int[] labels = new int[n];

        // Compute distances
        for (int i = 0; i < n; i++) {
            distances[i] = DistanceCalculator.euclideanDistance(testImage, trainingFeatures[i]);
            labels[i] = trainingLabels[i];
        }

        // Sort distances and labels together using selection sort
        for (int i = 0; i < k; i++) {
            // Find the index of the minimum distance
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (distances[j] < distances[minIndex]) {
                    minIndex = j;
                }
            }
            // Swap distances[i] and distances[minIndex]
            double tempDist = distances[i];
            distances[i] = distances[minIndex];
            distances[minIndex] = tempDist;

            // Swap labels[i] and labels[minIndex]
            int tempLabel = labels[i];
            labels[i] = labels[minIndex];
            labels[minIndex] = tempLabel;
        }

        // Count labels in the first k entries
        int[] labelCounts = new int[10]; // Assuming labels are digits 0-9
        for (int i = 0; i < k; i++) {
            int label = labels[i];
            labelCounts[label]++;
        }

        // Find the label with the highest count
        int predictedLabel = 0;
        int maxCount = labelCounts[0];
        for (int i = 1; i < 10; i++) {
            if (labelCounts[i] > maxCount) {
                maxCount = labelCounts[i];
                predictedLabel = i;
            }
        }

        // Handle ties
        int numMaxLabels = 0;
        for (int count : labelCounts) {
            if (count == maxCount) {
                numMaxLabels++;
            }
        }

        if (numMaxLabels == 1) {
            // Unique label with max count
            return predictedLabel;
        } else {
            // Tie occurred
            // Among the tied labels, choose the one with the smallest cumulative distance
            double[] cumulativeDistances = new double[10];
            for (int i = 0; i < k; i++) {
                int label = labels[i];
                if (labelCounts[label] == maxCount) {
                    cumulativeDistances[label] += distances[i];
                }
            }

            double minCumulativeDistance = Double.MAX_VALUE;
            for (int i = 0; i < 10; i++) {
                if (labelCounts[i] == maxCount && cumulativeDistances[i] < minCumulativeDistance) {
                    minCumulativeDistance = cumulativeDistances[i];
                    predictedLabel = i;
                }
            }
            return predictedLabel;
        }
    }
}
