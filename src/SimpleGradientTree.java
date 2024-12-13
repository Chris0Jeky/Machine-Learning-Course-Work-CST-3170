//import java.util.List;
//import java.util.ArrayList;
//import java.util.Random;
//
//class SimpleGradientTree {
//    private Node root;
//    private int maxDepth;
//    private int minSamplesLeaf;
//
//    public SimpleGradientTree(int maxDepth, int minSamplesLeaf) {
//        this.maxDepth = maxDepth;
//        this.minSamplesLeaf = minSamplesLeaf;
//    }
//
//    public void train(int[][] features, double[] gradients) {
//        // We'll implement a very simple tree: just one split if possible, otherwise a leaf
//        // If maxDepth=1 means one decision stump; if maxDepth > 1 we could add recursive logic
//        List<Integer> indices = new ArrayList<>();
//        for (int i = 0; i < features.length; i++) indices.add(i);
//
//        root = buildNode(features, gradients, indices, maxDepth);
//    }
//
//    private Node buildNode(int[][] features, double[] gradients, List<Integer> indices, int depth) {
//        if (depth == 0 || indices.size() < minSamplesLeaf) {
//            return createLeaf(gradients, indices);
//        }
//
//        // Find best split
//        Split best = findBestSplit(features, gradients, indices);
//        if (best == null) {
//            return createLeaf(gradients, indices);
//        }
//
//        List<Integer> leftIdx = new ArrayList<>();
//        List<Integer> rightIdx = new ArrayList<>();
//        for (int i : indices) {
//            if (features[i][best.feature] <= best.threshold) leftIdx.add(i);
//            else rightIdx.add(i);
//        }
//
//        // If no improvement or one side empty
//        if (leftIdx.size() < minSamplesLeaf || rightIdx.size() < minSamplesLeaf) {
//            return createLeaf(gradients, indices);
//        }
//
//        Node node = new Node();
//        node.isLeaf = false;
//        node.feature = best.feature;
//        node.threshold = best.threshold;
//
//        node.left = buildNode(features, gradients, leftIdx, depth - 1);
//        node.right = buildNode(features, gradients, rightIdx, depth - 1);
//
//        return node;
//    }
//
//    private Node createLeaf(double[] gradients, List<Integer> indices) {
//        Node leaf = new Node();
//        leaf.isLeaf = true;
//        double sum = 0.0;
//        for (int i : indices) sum += gradients[i];
//        leaf.value = sum / indices.size();
//        return leaf;
//    }
//
//    private Split findBestSplit(int[][] features, double[] gradients, List<Integer> indices) {
//        // We'll try a random subset of features and a few thresholds
//        // A better approach: sort indices by feature and try splits.
//        // For simplicity, try a few random features and random thresholds.
//        Random rand = new Random();
//        int tries = Math.min(5, features[0].length); // try up to 5 random features
//        double bestGain = 0.0;
//        Split best = null;
//
//        for (int t = 0; t < tries; t++) {
//            int f = rand.nextInt(features[0].length);
//            // choose a random threshold from data
//            int idx = indices.get(rand.nextInt(indices.size()));
//            int candidateThreshold = features[idx][f];
//
//            double gain = computeGain(features, gradients, indices, f, candidateThreshold);
//            if (gain > bestGain) {
//                bestGain = gain;
//                best = new Split();
//                best.feature = f;
//                best.threshold = candidateThreshold;
//                best.gain = gain;
//            }
//        }
//        return best;
//    }
//
//    private double computeGain(int[][] features, double[] gradients, List<Integer> indices, int f, int threshold) {
//        // Split data
//        double sumLeft = 0.0;
//        int countLeft = 0;
//        double sumRight = 0.0;
//        int countRight = 0;
//
//        for (int i : indices) {
//            if (features[i][f] <= threshold) {
//                sumLeft += gradients[i];
//                countLeft++;
//            } else {
//                sumRight += gradients[i];
//                countRight++;
//            }
//        }
//
//        if (countLeft == 0 || countRight == 0) return 0.0;
//
//        // Before split variance
//        double sumAll = sumLeft + sumRight;
//        double varBefore = sumAll * sumAll / indices.size();
//
//        // After split variance reduction (we use sum of squares trick)
//        double varLeft = (sumLeft * sumLeft) / countLeft;
//        double varRight = (sumRight * sumRight) / countRight;
//        double varAfter = varLeft + varRight;
//
//        double gain = varBefore - varAfter;
//        return (gain > 0) ? gain : 0.0;
//    }
//
//    public double predict(int[] sample) {
//        Node node = root;
//        while (!node.isLeaf) {
//            if (sample[node.feature] <= node.threshold) {
//                node = node.left;
//            } else {
//                node = node.right;
//            }
//        }
//        return node.value;
//    }
//
//    class Node {
//        boolean isLeaf;
//        int feature;
//        int threshold;
//        double value;
//        Node left, right;
//    }
//
//    class Split {
//        int feature;
//        int threshold;
//        double gain;
//    }
//}
