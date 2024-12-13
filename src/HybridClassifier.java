public class HybridClassifier implements Classifier {
    // Uses NearestNeighbor for close samples, and a fallback if distance is large.
    private NearestNeighborClassifier nn;
    private Classifier fallback;
    private double distanceThreshold;

    public HybridClassifier(NearestNeighborClassifier nn, Classifier fallback, double distanceThreshold) {
        this.nn = nn;
        this.fallback = fallback;
        this.distanceThreshold = distanceThreshold;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        nn.train(features, labels);
        fallback.train(features, labels);
    }

    @Override
    public int predict(int[] sample) {
        // Compute nearest neighbor distance
        double minDistance = Double.MAX_VALUE;
        int predictedLabel = -1;
        for (int i = 0; i < nn.getTrainingFeatures().length; i++) {
            double dist = DistanceCalculator.euclideanDistance(sample, nn.getTrainingFeatures()[i]);
            if (dist < minDistance) {
                minDistance = dist;
                predictedLabel = nn.getTrainingLabels()[i];
            }
        }

        if (minDistance > distanceThreshold) {
            // Use fallback classifier if distance is high
            return fallback.predict(sample);
        } else {
            return predictedLabel;
        }
    }
}
