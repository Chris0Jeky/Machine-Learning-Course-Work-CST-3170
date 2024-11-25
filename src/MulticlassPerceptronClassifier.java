public class MulticlassPerceptronClassifier implements Classifier {
    private int numClasses;
    private double[][] weights;
    private double[] biases;
    private int epochs;
    private int featureSize;

    public MulticlassPerceptronClassifier(int epochs, int featureSize, int numClasses) {
        this.epochs = epochs;
        this.featureSize = featureSize;
        this.numClasses = numClasses;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        initializeWeightsAndBiases();
        int n = features.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                int[] x = features[i];
                int y = labels[i];

                // Compute scores for all classes
                double[] scores = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    scores[c] = dotProduct(weights[c], x) + biases[c];
                }

                // Predicted class is the one with the highest score
                int predictedClass = argMax(scores);

                // If prediction is incorrect, update weights and biases
                if (predictedClass != y) {
                    // Update weights and biases
                    for (int j = 0; j < featureSize; j++) {
                        // Decrease weight for wrong class
                        weights[predictedClass][j] -= x[j];
                        // Increase weight for correct class
                        weights[y][j] += x[j];
                    }
                    // Update biases
                    biases[predictedClass] -= 1;
                    biases[y] += 1;
                }
            }
        }
    }

    private void initializeWeightsAndBiases() {
        weights = new double[numClasses][featureSize];
        biases = new double[numClasses];
    }

    @Override
    public int predict(int[] sample) {
        double[] scores = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            scores[c] = dotProduct(weights[c], sample) + biases[c];
        }
        return argMax(scores);
    }

private int argMax(double[] scores) {
        double maxScore = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private double dotProduct(double[] w, int[] x) {
        double result = 0.0;
        for (int i = 0; i < w.length; i++) {
            result += w[i] * x[i];
        }
        return result;
    }
}
