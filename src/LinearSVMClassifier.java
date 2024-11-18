public class LinearSVMClassifier implements Classifier {
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

    @Override
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
                    // Update bias
                    // No update needed for bias in this case
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

    @Override
    public int predict(int[] sample) {
        double linearOutput = dotProduct(weights, sample) + bias;
        return linearOutput >= 0 ? 1 : -1;
    }

    private double dotProduct(double[] w, int[] x) {
        double result = 0.0;
        for (int i = 0; i < w.length; i++) {
            result += w[i] * x[i];
        }
        return result;
    }

    public double decisionFunction(int[] sample) {
        return dotProduct(weights, sample) + bias;
    }

}
