import java.util.*;

public class MultiClassGradientBoostedTreesClassifier implements Classifier {
    private int numClasses;
    private int numTrees;     // number of boosting rounds
    private double eta;       // learning rate
    private int maxDepth;
    private int minSamplesLeaf;
    private int minSamplesSplit;
    private double lambda;    // L2 regularization on leaf values
    private double gamma;     // min split loss reduction
    private double maxFeaturesRatio;

    // predictions[sample][class]: raw scores
    private double[][] predictions;
    private List<List<GradientTree>> allTrees;

    public MultiClassGradientBoostedTreesClassifier(int numClasses, int numTrees, double eta, int maxDepth, int minSamplesLeaf, int minSamplesSplit, double lambda, double gamma, double maxFeaturesRatio) {
        this.numClasses = numClasses;
        this.numTrees = numTrees;
        this.eta = eta;
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.minSamplesSplit = minSamplesSplit;
        this.lambda = lambda;
        this.gamma = gamma;
        this.maxFeaturesRatio = maxFeaturesRatio;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        int n = features.length;
        predictions = new double[n][numClasses];
        for (int i = 0; i < n; i++) {
            Arrays.fill(predictions[i], 0.0);
        }

        allTrees = new ArrayList<>();

        for (int round = 0; round < numTrees; round++) {
            double[][] probs = computeSoftmaxProbabilities(predictions);
            // Compute gradients and hessians for multi-class softmax
            double[][] gradients = new double[n][numClasses];
            double[][] hessians = new double[n][numClasses];
            computeGradientsAndHessians(labels, probs, gradients, hessians);

            List<GradientTree> treesThisRound = new ArrayList<>();
            for (int c = 0; c < numClasses; c++) {
                double[] gC = new double[n];
                double[] hC = new double[n];
                for (int i = 0; i < n; i++) {
                    gC[i] = gradients[i][c];
                    hC[i] = hessians[i][c];
                }

                GradientTree tree = new GradientTree(maxDepth, minSamplesLeaf, minSamplesSplit, lambda, gamma, maxFeaturesRatio, true);
                tree.train(features, gC, hC);
                treesThisRound.add(tree);

                // Update predictions
                for (int i = 0; i < n; i++) {
                    double update = eta * tree.predict(features[i]);
                    predictions[i][c] += update;
                }
            }

            allTrees.add(treesThisRound);
        }
    }

    @Override
    public int predict(int[] sample) {
        double[] finalScores = new double[numClasses];
        for (List<GradientTree> roundTrees : allTrees) {
            for (int c = 0; c < numClasses; c++) {
                finalScores[c] += eta * roundTrees.get(c).predict(sample);
            }
        }

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

    private void computeGradientsAndHessians(int[] labels, double[][] probs, double[][] gradients, double[][] hessians) {
        int n = labels.length;
        // For softmax cross-entropy:
        // gradient(i,c) = p(i,c) - 1(label(i)=c)
        // hessian(i,c) = p(i,c)*(1 - p(i,c))
        for (int i = 0; i < n; i++) {
            int trueClass = labels[i];
            for (int c = 0; c < numClasses; c++) {
                double indicator = (c == trueClass) ? 1.0 : 0.0;
                double p = probs[i][c];
                gradients[i][c] = (p - indicator);
                hessians[i][c] = p * (1.0 - p);
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
