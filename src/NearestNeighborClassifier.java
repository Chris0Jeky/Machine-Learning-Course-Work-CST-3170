public class NearestNeighborClassifier implements Classifier {
    private int[][] trainingFeatures;
    private int[] trainingLabels;

    public NearestNeighborClassifier(int[][] trainingFeatures, int[] trainingLabels) {
        this.trainingFeatures = trainingFeatures;
        this.trainingLabels = trainingLabels;
    }

    @Override
    public int predict(int[] testImage) {
        double minDistance = Double.MAX_VALUE;
        int predictedLabel = -1;

        for (int i = 0; i < trainingFeatures.length; i++) {
            double distance = DistanceCalculator.euclideanDistance(testImage, trainingFeatures[i]);
            if (distance < minDistance) {
                minDistance = distance;
                predictedLabel = trainingLabels[i];
            }
        }
        return predictedLabel;
    }
}
