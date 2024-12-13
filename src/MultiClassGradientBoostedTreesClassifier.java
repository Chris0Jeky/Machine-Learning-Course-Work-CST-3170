import java.util.*;

public class MultiClassGradientBoostedTreesClassifier implements Classifier {
    private int numClasses;
    private int numTrees;     // number of boosting rounds
    private double eta;       // learning rate
    private int maxDepth;
    private int minSamplesLeaf;
    private int minSamplesSplit;
    private double lambda;    // regularization parameter for leaf values if using second-order

    // Storing raw predictions: predictions[sample][class]
    private double[][] predictions;
    // Store trees: For each iteration, we have one tree per class
    private List<List<GradientTree>> allTrees;

    public MultiClassGradientBoostedTreesClassifier(int numClasses, int numTrees, double eta, int maxDepth, int minSamplesLeaf, int minSamplesSplit, double lambda) {
        this.numClasses = numClasses;
        this.numTrees = numTrees;
        this.eta = eta;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.minSamplesSplit = minSamplesSplit;
        this.lambda = lambda;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        int n = features.length;

        // Convert labels from {0,...,numClasses-1} to categorical
        // Already suitable for multi-class softmax.

        // Initialize predictions (raw scores) = 0
        predictions = new double[n][numClasses];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < numClasses; c++) {
                predictions[i][c] = 0.0;
            }
        }

        allTrees = new ArrayList<>();
        for (int round = 0; round < numTrees; round++) {
            // Compute probabilities using softmax
            double[][] probs = computeSoftmaxProbabilities(predictions);

            // Compute gradients
            double[][] gradients = new double[n][numClasses];
            // If second-order, we would also compute Hessians: double[][] hessians = ...
            // For now, first-order only
            computeGradients(labels, probs, gradients);

            // Train one tree per class on respective gradients
            List<GradientTree> treesThisRound = new ArrayList<>();
            for (int c = 0; c < numClasses; c++) {
                GradientTree tree = new GradientTree(maxDepth, minSamplesLeaf, minSamplesSplit, /* secondOrder: false */false, lambda);
                // Extract gradient vector for class c
                double[] classGradients = new double[n];
                for (int i = 0; i < n; i++) {
                    classGradients[i] = gradients[i][c];
                }

                tree.train(features, classGradients);
                treesThisRound.add(tree);

                // Update predictions: predictions[i][c] += eta * tree.predict(features[i])
                for (int i = 0; i < n; i++) {
                    predictions[i][c] += eta * tree.predict(features[i]);
                }
            }

            allTrees.add(treesThisRound);
        }
    }

    @Override
    public int predict(int[] sample) {
        // Compute final raw scores by summing all trees' contributions
        double[] finalScores = new double[numClasses];
        for (int c = 0; c < numClasses; c++) finalScores[c] = 0.0;

        for (List<GradientTree> roundTrees : allTrees) {
            for (int c = 0; c < numClasses; c++) {
                finalScores[c] += eta * roundTrees.get(c).predict(sample);
            }
        }

        // Apply softmax
        double[] probs = softmax(finalScores);
        return argMax(probs);
    }

    private double[][] computeSoftmaxProbabilities(double[][] pred) {
        int n = pred.length;
        int k = numClasses;
        double[][] probs = new double[n][k];
        for (int i = 0; i < n; i++) {
            probs[i] = softmax(pred[i]);
        }
        return probs;
    }

    private double[] softmax(double[] scores) {
        double max = Double.NEGATIVE_INFINITY;
        for (double s : scores) {
            if (s > max) max = s;
        }
        double sum = 0.0;
        for (double s : scores) {
            sum += Math.exp(s - max);
        }
        double[] out = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            out[i] = Math.exp(scores[i] - max) / sum;
        }
        return out;
    }

    private void computeGradients(int[] labels, double[][] probs, double[][] gradients) {
        int n = labels.length;
        // gradient_i,c = p_i,c - 1(label(i)=c)
        for (int i = 0; i < n; i++) {
            int trueClass = labels[i];
            for (int c = 0; c < numClasses; c++) {
                double indicator = (c == trueClass) ? 1.0 : 0.0;
                gradients[i][c] = (probs[i][c] - indicator);
            }
        }
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
