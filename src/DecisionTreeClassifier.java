import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.Collections;
import java.util.Arrays;

class DecisionTreeClassifier implements Classifier {
    // A basic decision tree for classification using Gini impurity.
    // Uses a subset of features for each split if maxFeatures < total features.

    private int numClasses;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private int maxFeatures; // number of features to consider at each split
    private Random rand;
    private Node root;

    public DecisionTreeClassifier(int numClasses, int maxDepth, int minSamplesSplit, int minSamplesLeaf, int maxFeatures, Random rand) {
        this.numClasses = numClasses;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.rand = rand;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < features.length; i++) indices.add(i);
        root = buildNode(features, labels, indices, 0);
    }

    // Recursively builds the tree by finding splits that reduce impurity.
    private Node buildNode(int[][] features, int[] labels, List<Integer> indices, int depth) {
        int n = indices.size();
        if (n == 0) return null;

        // Compute class distribution
        int[] counts = new int[numClasses];
        for (int i : indices) {
            counts[labels[i]]++;
        }

        // If pure node or maxDepth reached or not enough samples to split further
        int majorityClass = argMax(counts);
        if (depth >= maxDepth || n < minSamplesSplit || isPure(counts)) {
            return createLeaf(counts);
        }

        // Find best split
        Split bestSplit = findBestSplit(features, labels, indices, counts);
        if (bestSplit == null || bestSplit.gain <= 0) {
            return createLeaf(counts);
        }

        // Partition indices
        List<Integer> leftIdx = new ArrayList<>();
        List<Integer> rightIdx = new ArrayList<>();
        for (int i : indices) {
            if (features[i][bestSplit.feature] <= bestSplit.threshold) {
                leftIdx.add(i);
            } else {
                rightIdx.add(i);
            }
        }

        if (leftIdx.size() < minSamplesLeaf || rightIdx.size() < minSamplesLeaf) {
            return createLeaf(counts);
        }

        Node node = new Node();
        node.feature = bestSplit.feature;
        node.threshold = bestSplit.threshold;
        node.isLeaf = false;
        node.left = buildNode(features, labels, leftIdx, depth + 1);
        node.right = buildNode(features, labels, rightIdx, depth + 1);
        return node;
    }

    private Node createLeaf(int[] counts) {
        Node leaf = new Node();
        leaf.isLeaf = true;
        leaf.classLabel = argMax(counts);
        return leaf;
    }

    @Override
    public int predict(int[] sample) {
        Node node = root;
        while (node != null && !node.isLeaf) {
            if (sample[node.feature] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.classLabel;
    }

    private Split findBestSplit(int[][] features, int[] labels, List<Integer> indices, int[] parentCounts) {
        int d = features[0].length;
        int n = indices.size();

        // Random subset of features
        List<Integer> candidateFeatures = new ArrayList<>();
        for (int f = 0; f < d; f++) candidateFeatures.add(f);
        Collections.shuffle(candidateFeatures, rand);
        candidateFeatures = candidateFeatures.subList(0, Math.min(maxFeatures, d));

        double parentImpurity = gini(parentCounts);
        double bestGain = 0.0;
        Split best = null;

        // For each feature in candidateFeatures
        for (int f : candidateFeatures) {
            // Sort indices by this feature
            indices.sort((a, b) -> Integer.compare(features[a][f], features[b][f]));

            // Try splits between distinct feature values
            int[] leftCounts = new int[numClasses];
            int[] rightCounts = Arrays.copyOf(parentCounts, parentCounts.length);

            for (int i = 0; i < n - 1; i++) {
                int idx = indices.get(i);
                int c = labels[idx];
                leftCounts[c]++;
                rightCounts[c]--;

                if (features[indices.get(i)][f] == features[indices.get(i+1)][f]) {
                    // Same value, no split here
                    continue;
                }

                int leftSize = i + 1;
                int rightSize = n - leftSize;
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) continue;

                double leftImpurity = gini(leftCounts);
                double rightImpurity = gini(rightCounts);
                double wLeft = (double)leftSize / n;
                double wRight = 1.0 - wLeft;
                double gain = parentImpurity - (wLeft * leftImpurity + wRight * rightImpurity);

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

    private double gini(int[] counts) {
        int sum = 0;
        for (int c : counts) sum += c;
        double g = 1.0;
        for (int c : counts) {
            double p = (double)c / sum;
            g -= p*p;
        }
        return g;
    }

    private boolean isPure(int[] counts) {
        int nonZero = 0;
        for (int c : counts) {
            if (c > 0) nonZero++;
            if (nonZero > 1) return false;
        }
        return true;
    }

    private int argMax(int[] arr) {
        int idx = 0;
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    class Node {
        boolean isLeaf;
        int feature;
        int threshold;
        int classLabel;
        Node left, right;
    }

    class Split {
        int feature;
        int threshold;
        double gain;
    }
}
