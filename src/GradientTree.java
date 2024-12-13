import java.util.List;
import java.util.ArrayList;
import java.util.Random;


class GradientTree {
    private Node root;
    private int maxDepth;
    private int minSamplesLeaf;
    private int minSamplesSplit;
    private boolean secondOrder;
    private double lambda;

    public GradientTree(int maxDepth, int minSamplesLeaf, int minSamplesSplit, boolean secondOrder, double lambda) {
        this.maxDepth = maxDepth;
        this.minSamplesLeaf = minSamplesLeaf;
        this.minSamplesSplit = minSamplesSplit;
        this.secondOrder = secondOrder;
        this.lambda = lambda;
    }

    public void train(int[][] features, double[] gradients) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < gradients.length; i++) indices.add(i);

        root = buildNode(features, gradients, indices, 0);
    }

    private Node buildNode(int[][] features, double[] gradients, List<Integer> indices, int depth) {
        if (depth >= maxDepth || indices.size() < minSamplesSplit) {
            return createLeaf(gradients, indices);
        }

        Split best = findBestSplit(features, gradients, indices);
        if (best == null) {
            return createLeaf(gradients, indices);
        }

        List<Integer> leftIdx = new ArrayList<>();
        List<Integer> rightIdx = new ArrayList<>();
        for (int i : indices) {
            if (features[i][best.feature] <= best.threshold) leftIdx.add(i);
            else rightIdx.add(i);
        }

        if (leftIdx.size() < minSamplesLeaf || rightIdx.size() < minSamplesLeaf) {
            return createLeaf(gradients, indices);
        }

        Node node = new Node();
        node.feature = best.feature;
        node.threshold = best.threshold;
        node.left = buildNode(features, gradients, leftIdx, depth + 1);
        node.right = buildNode(features, gradients, rightIdx, depth + 1);
        node.isLeaf = false;
        return node;
    }

    private Node createLeaf(double[] gradients, List<Integer> indices) {
        Node leaf = new Node();
        leaf.isLeaf = true;

        double sumGrad = 0.0;
        for (int i : indices) sumGrad += gradients[i];

        // First-order leaf value = average negative gradient
        double leafValue = -sumGrad / (indices.size());
        // If secondOrder, would involve Hessians and lambda.

        leaf.value = leafValue;
        return leaf;
    }

    private Split findBestSplit(int[][] features, double[] gradients, List<Integer> indices) {
        int n = indices.size();
        int d = features[0].length;

        // We'll pick best split by variance reduction on gradients
        double sumAll = 0.0;
        for (int i : indices) sumAll += gradients[i];
        double meanAll = sumAll / n;
        double baseVar = 0.0;
        for (int i : indices) {
            double diff = gradients[i] - meanAll;
            baseVar += diff * diff;
        }

        double bestGain = 0.0;
        Split best = null;
        Random rand = new Random();

        // Try all features or a subset:
        // For simplicity, try a few random features:
        int featureTries = Math.min(d, 10);
        for (int t = 0; t < featureTries; t++) {
            int f = rand.nextInt(d);

            // Sort indices by feature f
            indices.sort((a,b) -> Integer.compare(features[a][f], features[b][f]));

            double leftSum = 0.0;
            int leftCount = 0;

            // Try splits between distinct feature values
            for (int i = 0; i < n - 1; i++) {
                int idx = indices.get(i);
                leftSum += gradients[idx];
                leftCount++;
                if (features[indices.get(i)][f] == features[indices.get(i+1)][f]) {
                    continue;
                }

                int leftSize = leftCount;
                int rightSize = n - leftSize;
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) continue;

                double rightSum = sumAll - leftSum;
                double leftMean = leftSum / leftSize;
                double rightMean = rightSum / rightSize;

                double leftVar = 0.0;
                double rightVar = 0.0;
                // For a more efficient code, precompute prefix sums of squared gradients or just sum of gradients,
                // but here we do brute force:
                for (int j = 0; j < leftSize; j++) {
                    int idL = indices.get(j);
                    double diff = gradients[idL] - leftMean;
                    leftVar += diff*diff;
                }
                for (int k = leftSize; k < n; k++) {
                    int idR = indices.get(k);
                    double diff = gradients[idR] - rightMean;
                    rightVar += diff*diff;
                }

                double gain = baseVar - (leftVar + rightVar);
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
