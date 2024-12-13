public class NearestNeighborClassifier implements Classifier {
    private int[][] trainingFeatures; // Stores the training feature vectors
    private int[] trainingLabels;  // Stores the corresponding labels for the training data

    // Getter for training features (used for external access if needed)
    public int[][] getTrainingFeatures() {
        return trainingFeatures;
    }

    // Getter for training labels (used for external access if needed)
    public int[] getTrainingLabels() {
        return trainingLabels;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        // Simply store the training features and labels for future predictions
        this.trainingFeatures = features;
        this.trainingLabels = labels;
    }

    @Override
    public int predict(int[] testImage) {
        double minDistance = Double.MAX_VALUE; // Initialize the minimum distance to a very large value
        int predictedLabel = -1;  // Initialize the predicted label as invalid (-1)

        // Iterate through all training samples to find the nearest neighbor
        for (int i = 0; i < trainingFeatures.length; i++) {
            // Compute the Euclidean distance between the test image and the current training sample
            double distance = DistanceCalculator.euclideanDistance(testImage, trainingFeatures[i]);
            // Update the minimum distance and predicted label if a closer neighbor is found
            if (distance < minDistance) {
                minDistance = distance;
                predictedLabel = trainingLabels[i];
            }
        }
        // Return the label of the nearest neighbor
        return predictedLabel;
    }
}
