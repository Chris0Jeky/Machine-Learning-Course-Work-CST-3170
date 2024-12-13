public class DistanceCalculator {
    // Computes Euclidean distance between two feature vectors
    public static double euclideanDistance(int[] vector1, int[] vector2) {
        double sum = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            double diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
