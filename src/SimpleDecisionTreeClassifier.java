public class SimpleDecisionTreeClassifier implements Classifier {
    private int featureIndex;
    private int threshold;
    private int majorityClassLeft;
    private int majorityClassRight;

    private int numClasses;

    public SimpleDecisionTreeClassifier(int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        // Build a single-level decision stump:
        // 1. Choose the best feature and threshold to split the data
        // 2. Store majority classes on each side

        int n = features.length;
        int d = features[0].length;

        // If there's no variation, just pick majority class overall
        int[] globalCounts = classCounts(labels, numClasses);
        int globalMajority = argMax(globalCounts);

        // Try a few random features and thresholds to find best split
        java.util.Random rand = new java.util.Random();
        int tries = Math.min(d, 10); // Try up to 10 random features
        double bestGini = gini(labels);
        featureIndex = 0;
        threshold = 0;
        majorityClassLeft = globalMajority;
        majorityClassRight = globalMajority;

        for (int t = 0; t < tries; t++) {
            int f = rand.nextInt(d);
            // Pick a random threshold from data
            int idx = rand.nextInt(n);
            int candidateThreshold = features[idx][f];

            // Split data
            java.util.List<Integer> leftIdx = new java.util.ArrayList<>();
            java.util.List<Integer> rightIdx = new java.util.ArrayList<>();

            for (int i = 0; i < n; i++) {
                if (features[i][f] <= candidateThreshold) {
                    leftIdx.add(i);
                } else {
                    rightIdx.add(i);
                }
            }

            if (leftIdx.isEmpty() || rightIdx.isEmpty()) continue;

            int[] leftLabels = subsetLabels(labels, leftIdx);
            int[] rightLabels = subsetLabels(labels, rightIdx);

            double g = weightedGini(leftLabels, rightLabels);
            if (g < bestGini) {
                bestGini = g;
                featureIndex = f;
                threshold = candidateThreshold;
                majorityClassLeft = argMax(classCounts(leftLabels, numClasses));
                majorityClassRight = argMax(classCounts(rightLabels, numClasses));
            }
        }
    }

    @Override
    public int predict(int[] sample) {
        if (sample[featureIndex] <= threshold) {
            return majorityClassLeft;
        } else {
            return majorityClassRight;
        }
    }

    // Helper methods

    private int[] classCounts(int[] labels, int numClasses) {
        int[] counts = new int[numClasses];
        for (int l : labels) {
            counts[l]++;
        }
        return counts;
    }

    private double gini(int[] labels) {
        int[] counts = classCounts(labels, numClasses);
        int total = labels.length;
        double sum = 0.0;
        for (int c : counts) {
            double p = (double) c / total;
            sum += p * p;
        }
        return 1.0 - sum;
    }

    private double weightedGini(int[] leftLabels, int[] rightLabels) {
        int total = leftLabels.length + rightLabels.length;
        double gLeft = gini(leftLabels);
        double gRight = gini(rightLabels);
        return (leftLabels.length * gLeft + rightLabels.length * gRight) / total;
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

    private int[] subsetLabels(int[] labels, java.util.List<Integer> idx) {
        int[] sub = new int[idx.size()];
        for (int i = 0; i < idx.size(); i++) {
            sub[i] = labels[idx.get(i)];
        }
        return sub;
    }
}
