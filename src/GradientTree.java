import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;


class GradientTree {
    private Node root;
    private int maxDepth;
    private int minSamplesLeaf;
    private int minSamplesSplit;
    private double lambda; // L2 reg on leaf
    private double gamma;  // min split loss reduction
    private double maxFeaturesRatio;
    private boolean secondOrder; // if true, we use hessians

    private Random rand;

    public GradientTree(int maxDepth, int minSamplesLeaf, int minSamplesSplit, double lambda, double gamma, double maxFeaturesRatio, boolean secondOrder) {
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.minSamplesSplit = minSamplesSplit;
        this.lambda = lambda;
        this.gamma = gamma;
        this.maxFeaturesRatio = maxFeaturesRatio;
        this.secondOrder = secondOrder;
        this.rand = new Random();
    }

    public void train(int[][] features, double[] gradients, double[] hessians) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < gradients.length; i++) indices.add(i);
        root = buildNode(features, gradients, hessians, indices, 0);
    }

    private Node buildNode(int[][] features, double[] gradients, double[] hessians, List<Integer> indices, int depth) {
        if (depth >= maxDepth || indices.size() < minSamplesSplit) {
            return createLeaf(gradients, hessians, indices);
        }

        Split best = findBestSplit(features, gradients, hessians, indices);
        if (best == null || best.gain < gamma) {
            return createLeaf(gradients, hessians, indices);
        }

        List<Integer> leftIdx = new ArrayList<>();
        List<Integer> rightIdx = new ArrayList<>();
        for (int i : indices) {
            if (features[i][best.feature] <= best.threshold) leftIdx.add(i);
            else rightIdx.add(i);
        }

        if (leftIdx.size() < minSamplesLeaf || rightIdx.size() < minSamplesLeaf) {
            return createLeaf(gradients, hessians, indices);
        }

        Node node = new Node();
        node.feature = best.feature;
        node.threshold = best.threshold;
        node.isLeaf = false;
        node.left = buildNode(features, gradients, hessians, leftIdx, depth + 1);
        node.right = buildNode(features, gradients, hessians, rightIdx, depth + 1);
        return node;
    }

    private Node createLeaf(double[] gradients, double[] hessians, List<Integer> indices) {
        Node leaf = new Node();
        leaf.isLeaf = true;

        double sumG = 0.0;
        double sumH = 0.0;
        for (int i : indices) {
            sumG += gradients[i];
            sumH += hessians[i];
        }

        // Leaf value for second-order:
        // leafValue = - sumG / (sumH + lambda)
        double leafValue = -sumG / (sumH + lambda);
        leaf.value = leafValue;

        return leaf;
    }

    private Split findBestSplit(int[][] features, double[] gradients, double[] hessians, List<Integer> indices) {
        int n = indices.size();
        int d = features[0].length;
        int maxFeatures = (int)(d * maxFeaturesRatio);

        // Compute total stats
        double sumG = 0.0, sumH = 0.0;
        for (int i : indices) {
            sumG += gradients[i];
            sumH += hessians[i];
        }
        double baseScore = leafScore(sumG, sumH, lambda);
        // We'll try to find a split that increases the gain: gain = leftScore + rightScore - baseScore

        // Random subset of features
        List<Integer> candidateFeatures = new ArrayList<>();
        for (int f = 0; f < d; f++) candidateFeatures.add(f);
        Collections.shuffle(candidateFeatures, rand);
        candidateFeatures = candidateFeatures.subList(0, Math.min(maxFeatures, d));

        double bestGain = 0.0;
        Split best = null;

        for (int f : candidateFeatures) {
            // Sort indices by feature f
            indices.sort((a,b) -> Integer.compare(features[a][f], features[b][f]));

            double leftG = 0.0, leftH = 0.0;
            for (int i = 0; i < n - 1; i++) {
                int idx = indices.get(i);
                leftG += gradients[idx];
                leftH += hessians[idx];

                if (features[indices.get(i)][f] == features[indices.get(i+1)][f]) continue;

                int leftSize = i+1;
                int rightSize = n - leftSize;
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) continue;

                double rightG = sumG - leftG;
                double rightH = sumH - leftH;

                double leftScore = leafScore(leftG, leftH, lambda);
                double rightScore = leafScore(rightG, rightH, lambda);
                double gain = leftScore + rightScore - baseScore;

                if (gain > bestGain) {
                    bestGain = gain;
                    Split s = new Split();
                    s.feature = f;
                    s.threshold = features[indices.get(i)][f];
                    s.gain = gain;
                    best = s;
                }
            }
        }

        return best;
    }

    private double leafScore(double sumG, double sumH, double lambda) {
        // (-(sumG)^2) / (sumH + lambda) is basically the gain formula's leaf part
        return -(sumG * sumG) / (sumH + lambda);
    }

    public double predict(int[] sample) {
        Node node = root;
        while (node != null && !node.isLeaf) {
            if (sample[node.feature] <= node.threshold) node = node.left;
            else node = node.right;
        }
        return node.value;
    }

    class Node {
        boolean isLeaf;
        int feature;
        int threshold;
        double value;
        Node left, right;
    }

    class Split {
        int feature;
        int threshold;
        double gain;
    }
}

