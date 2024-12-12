public class MLPClassifier implements Classifier {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] W1; // input -> hidden
    private double[] b1;
    private double[][] W2; // hidden -> output
    private double[] b2;
    private double learningRate;
    private int epochs;

    public MLPClassifier(int inputSize, int hiddenSize, int outputSize, double learningRate, int epochs) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        initWeights();
    }

    private void initWeights() {
        W1 = new double[hiddenSize][inputSize];
        b1 = new double[hiddenSize];
        W2 = new double[outputSize][hiddenSize];
        b2 = new double[outputSize];

        // Initialize weights with small random values
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] = rand.nextGaussian() * 0.01;
            }
        }
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] = rand.nextGaussian() * 0.01;
            }
        }
    }

    @Override
    public void train(int[][] features, int[] labels) {
        int n = features.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                double[] x = toDouble(features[i]);
                int y = labels[i];

                // Forward pass
                double[] h = new double[hiddenSize];
                for (int hh = 0; hh < hiddenSize; hh++) {
                    double sum = b1[hh];
                    for (int jj = 0; jj < inputSize; jj++) {
                        sum += W1[hh][jj] * x[jj];
                    }
                    h[hh] = relu(sum);
                }

                double[] o = new double[outputSize];
                for (int oo = 0; oo < outputSize; oo++) {
                    double sum = b2[oo];
                    for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                        sum += W2[oo][hh2] * h[hh2];
                    }
                    o[oo] = sum;
                }

                double[] probs = softmax(o);

                // Compute gradients w.r.t output (Cross-entropy + softmax)
                // One-hot encoding for y
                double[] deltaO = new double[outputSize];
                for (int oo = 0; oo < outputSize; oo++) {
                    deltaO[oo] = probs[oo];
                }
                deltaO[y] -= 1.0;

                // Backprop to W2, b2
                double[][] gradW2 = new double[outputSize][hiddenSize];
                double[] gradb2 = new double[outputSize];
                for (int oo = 0; oo < outputSize; oo++) {
                    for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                        gradW2[oo][hh2] = deltaO[oo] * h[hh2];
                    }
                    gradb2[oo] = deltaO[oo];
                }

                // Backprop to hidden
                double[] deltaH = new double[hiddenSize];
                for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                    double sum = 0.0;
                    for (int oo = 0; oo < outputSize; oo++) {
                        sum += deltaO[oo] * W2[oo][hh2];
                    }
                    deltaH[hh2] = (h[hh2] > 0 ? sum : 0); // derivative of ReLU
                }

                // Backprop to W1, b1
                double[][] gradW1 = new double[hiddenSize][inputSize];
                double[] gradb1 = new double[hiddenSize];
                for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                    for (int jj = 0; jj < inputSize; jj++) {
                        gradW1[hh2][jj] = deltaH[hh2] * x[jj];
                    }
                    gradb1[hh2] = deltaH[hh2];
                }

                // Update weights
                for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                    for (int jj = 0; jj < inputSize; jj++) {
                        W1[hh2][jj] -= learningRate * gradW1[hh2][jj];
                    }
                    b1[hh2] -= learningRate * gradb1[hh2];
                }

                for (int oo = 0; oo < outputSize; oo++) {
                    for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                        W2[oo][hh2] -= learningRate * gradW2[oo][hh2];
                    }
                    b2[oo] -= learningRate * gradb2[oo];
                }
            }
        }
    }

    @Override
    public int predict(int[] sample) {
        double[] x = toDouble(sample);

        double[] h = new double[hiddenSize];
        for (int hh = 0; hh < hiddenSize; hh++) {
            double sum = b1[hh];
            for (int jj = 0; jj < inputSize; jj++) {
                sum += W1[hh][jj] * x[jj];
            }
            h[hh] = relu(sum);
        }

        double[] o = new double[outputSize];
        for (int oo = 0; oo < outputSize; oo++) {
            double sum = b2[oo];
            for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                sum += W2[oo][hh2] * h[hh2];
            }
            o[oo] = sum;
        }

        int pred = argMax(o);
        return pred;
    }

    private double[] toDouble(int[] arr) {
        double[] res = new double[arr.length];
        for (int i = 0; i < arr.length; i++) res[i] = arr[i];
        return res;
    }

    private double relu(double x) {
        return x > 0 ? x : 0;
    }

    private double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            if (v > max) max = v;
        }
        double sum = 0.0;
        for (double v : x) {
            sum += Math.exp(v - max);
        }
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.exp(x[i] - max) / sum;
        }
        return out;
    }

    private int argMax(double[] arr) {
        int idx = 0;
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }
}
