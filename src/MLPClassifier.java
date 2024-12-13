public class MLPClassifier implements Classifier {
    // A simple Multi-Layer Perceptron (MLP) classifier with a single hidden layer.
    // Uses softmax output layer and cross-entropy loss for multi-class classification.

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

    // Initialize all weights and biases with small random values.
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
        // For multiple epochs, run through all samples and update weights via backpropagation.
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                double[] x = toDouble(features[i]);
                int y = labels[i];

                // Forward pass: input -> hidden
                double[] h = new double[hiddenSize];
                for (int hh = 0; hh < hiddenSize; hh++) {
                    double sum = b1[hh];
                    for (int jj = 0; jj < inputSize; jj++) {
                        sum += W1[hh][jj] * x[jj];
                    }
                    h[hh] = relu(sum);
                }
                // Forward pass: hidden -> output
                double[] o = new double[outputSize];
                for (int oo = 0; oo < outputSize; oo++) {
                    double sum = b2[oo];
                    for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                        sum += W2[oo][hh2] * h[hh2];
                    }
                    o[oo] = sum;
                }

                double[] probs = softmax(o);

                // Compute output layer delta: deltaO = probs - one_hot(y)
                double[] deltaO = new double[outputSize];
                for (int oo = 0; oo < outputSize; oo++) {
                    deltaO[oo] = probs[oo];
                }
                deltaO[y] -= 1.0;

                // Compute gradients for W2 and b2
                double[][] gradW2 = new double[outputSize][hiddenSize];
                double[] gradb2 = new double[outputSize];
                for (int oo = 0; oo < outputSize; oo++) {
                    for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                        gradW2[oo][hh2] = deltaO[oo] * h[hh2];
                    }
                    gradb2[oo] = deltaO[oo];
                }

                // Compute delta for hidden layer: deltaH
                double[] deltaH = new double[hiddenSize];
                for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                    double sum = 0.0;
                    for (int oo = 0; oo < outputSize; oo++) {
                        sum += deltaO[oo] * W2[oo][hh2];
                    }
                    // ReLU derivative: pass gradient if h>0, else 0
                    deltaH[hh2] = (h[hh2] > 0 ? sum : 0); // derivative of ReLU
                }

                // Compute gradients for W1 and b1
                double[][] gradW1 = new double[hiddenSize][inputSize];
                double[] gradb1 = new double[hiddenSize];
                for (int hh2 = 0; hh2 < hiddenSize; hh2++) {
                    for (int jj = 0; jj < inputSize; jj++) {
                        gradW1[hh2][jj] = deltaH[hh2] * x[jj];
                    }
                    gradb1[hh2] = deltaH[hh2];
                }

                // Update all weights and biases
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
        // Single forward pass for prediction

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
        // The predicted class is the argmax of the output scores (before softmax)

        int pred = argMax(o);
        return pred;
    }

    // Convert int array to double array for calculations
    private double[] toDouble(int[] arr) {
        double[] res = new double[arr.length];
        for (int i = 0; i < arr.length; i++) res[i] = arr[i];
        return res;
    }

    // ReLU activation function
    private double relu(double x) {
        return x > 0 ? x : 0;
    }

    // Softmax normalization for probability outputs
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

    // Find index of maximum value in array (for class prediction)
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
