public class LinearSVMClassifier {
    private double[] weights;
    private double bias;
    private double learningRate;
    private double regularizationParam;
    private int epochs;

    public LinearSVMClassifier(double learningRate, double regularizationParam, int epochs, int featureSize) {
        this.learningRate = learningRate;
        this.regularizationParam = regularizationParam;
        this.epochs = epochs;
        this.weights = new double[featureSize];
        this.bias = 0.0;
    }

    public void train(int[][] features, int[] labels) {
        int n = features.length;
        int d = features[0].length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                int[] x = features[i];
                int y = labels[i];

                double linearOutput = dotProduct(weights, x) + bias;
                double yPred = y * linearOutput;

                if (yPred >= 1) {
                    // Update weights
                    for (int j = 0; j < d; j++) {
                        weights[j] -= learningRate * (2 * regularizationParam * weights[j]);
                    }
                    // No update for bias
                } else {
                    // Update weights
                    for (int j = 0; j < d; j++) {
                        weights[j] -= learningRate * (2 * regularizationParam * weights[j] - y * x[j]);
                    }
                    // Update bias
                    bias -= learningRate * (-y);
                }
            }
        }
    }

    public int predict(int[] sample) {
        double linearOutput = dotProduct(weights, sample) + bias;
        return linearOutput >= 0 ? 1 : -1;
    }

    public double decisionFunction(int[] sample) {
        return dotProduct(weights, sample) + bias;
    }

    private double dotProduct(double[] w, int[] x) {
        double result = 0.0;
        for (int i = 0; i < w.length; i++) {
            result += w[i] * x[i];
        }
        return result;
    }
}
