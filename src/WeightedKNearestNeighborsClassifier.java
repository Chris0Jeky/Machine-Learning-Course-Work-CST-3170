public class WeightedKNearestNeighborsClassifier implements Classifier {
    private int[][] trainingFeatures;
    private int[] trainingLabels;
    private int k;
    private int numClasses;

    public WeightedKNearestNeighborsClassifier(int k, int numClasses) {
        this.k = k;
        this.numClasses = numClasses;
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

        // Partial sort: find the k nearest neighbors
        // For simplicity, use selection sort for the first k neighbors:
        for (int i = 0; i < k; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (distances[j] < distances[minIndex]) {
                    minIndex = j;
                }
            }
            double tempDist = distances[i];
            distances[i] = distances[minIndex];
            distances[minIndex] = tempDist;

            int tempLabel = labels[i];
            labels[i] = labels[minIndex];
            labels[minIndex] = tempLabel;
        }

        // Weighted voting
        // weight = 1/(distance+epsilon)
        double[] labelWeights = new double[numClasses];
        double epsilon = 1e-5;
        for (int i = 0; i < k; i++) {
            int label = labels[i];
            if (label >= 0 && label < numClasses) {
                double weight = 1.0 / (distances[i] + epsilon);
                labelWeights[label] += weight;
            }
        }

        int predictedLabel = -1;
        double maxWeight = -1.0;
        for (int c = 0; c < numClasses; c++) {
            if (labelWeights[c] > maxWeight) {
                maxWeight = labelWeights[c];
                predictedLabel = c;
            }
        }

        return predictedLabel;
    }
}
