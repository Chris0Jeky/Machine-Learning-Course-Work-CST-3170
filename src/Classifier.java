public interface Classifier {
    // A generic interface for classifiers.
    // Implementations must provide a training method and a prediction method.
    void train(int[][] features, int[] labels);
    int predict(int[] sample);
}
