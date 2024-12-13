import java.util.*;

public class GradientBoostedTreesClassifier implements Classifier {
    // A simplified binary classification GBT using logistic loss (no second-order yet).
    private int numTrees;
    private double eta; // learning rate
    private int maxDepth;
    private int minSamplesLeaf;
    private int numClasses; // Assuming binary {0,1} for now
    private List<SimpleGradientTree> trees;

    public GradientBoostedTreesClassifier(int numTrees, double eta, int maxDepth, int minSamplesLeaf, int numClasses) {
        this.numTrees = numTrees;
        this.eta = eta;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.numClasses = numClasses; // For now, assume numClasses=2
    }

    @Override
    public void train(int[][] features, int[] labels) {
        int n = features.length;
        // Convert labels {0,1} to {+1,-1}
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = (labels[i] == 1) ? 1 : -1;
        }

        double[] predictions = new double[n]; // initial predictions = 0
        trees = new ArrayList<>();

        for (int iter = 0; iter < numTrees; iter++) {
            double[] gradients = computeGradients(y, predictions);

            // Train a tree on gradients
            SimpleGradientTree tree = new SimpleGradientTree(maxDepth, minSamplesLeaf);
            tree.train(features, gradients);

            // Update predictions
            for (int i = 0; i < n; i++) {
                predictions[i] += eta * tree.predict(features[i]);
            }

            trees.add(tree);
        }
    }

    @Override
    public int predict(int[] sample) {
        // Compute final score
        double score = 0.0;
        for (SimpleGradientTree tree : trees) {
            score += eta * tree.predict(sample);
        }
        // Convert score to probability: p = sigmoid(score)
        double p = 1.0 / (1.0 + Math.exp(-score));
        return (p >= 0.5) ? 1 : 0;
    }
    // Computes gradient of logistic loss for binary classification.
    private double[] computeGradients(int[] y, double[] predictions) {
        // logistic loss gradient w.r.t. raw score:
        // gradient_i = -y_i * p_i where p_i = 1/(1+exp(y_i*prediction_i))
        // Actually, logistic derivative:
        // p_i = 1/(1+exp(-prediction_i)) in binary classification if y_i in {+1,-1}:
        // gradient_i = d/d_prediction log(1+exp(-y_i*prediction_i)) = -y_i/(1+exp(y_i*prediction_i))
        int n = y.length;
        double[] gradients = new double[n];
        for (int i = 0; i < n; i++) {
            double expTerm = Math.exp(y[i] * predictions[i]);
            double p = 1.0 / (1.0 + expTerm);
            // gradient = -(y[i]) * p
            gradients[i] = -y[i] * p;
        }
        return gradients;
    }
}
