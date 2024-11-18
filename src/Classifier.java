public interface Classifier {
    void train(int[][] features, int[] labels);
    int predict(int[] sample);
}
