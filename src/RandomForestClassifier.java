public class RandomForestClassifier implements Classifier {
    private int numTrees;
    private int numClasses;
    private java.util.List<SimpleDecisionTreeClassifier> trees;
    private double sampleRatio; // what fraction of samples to use for each tree
    private double featureRatio; // what fraction of features to consider for splits

    public RandomForestClassifier(int numTrees, int numClasses, double sampleRatio, double featureRatio) {
        this.numTrees = numTrees;
        this.numClasses = numClasses;
        this.sampleRatio = sampleRatio;
        this.featureRatio = featureRatio;
        this.trees = new java.util.ArrayList<>();
    }

    @Override
    public void train(int[][] features, int[] labels) {
        trees.clear();
        int n = features.length;
        int d = features[0].length;
        int sampleSize = (int) (n * sampleRatio);
        int featureCount = (int) (d * featureRatio);

        java.util.Random rand = new java.util.Random();

        for (int t = 0; t < numTrees; t++) {
            // Bootstrap sample
            int[][] sampledFeatures = new int[sampleSize][d];
            int[] sampledLabels = new int[sampleSize];
            for (int i = 0; i < sampleSize; i++) {
                int idx = rand.nextInt(n);
                sampledFeatures[i] = features[idx];
                sampledLabels[i] = labels[idx];
            }

            // We could limit features here by randomly selecting a subset of features,
            // but since our tree is a stump that tries random features anyway, we rely on that randomness.
            // Or we can pre-prune features:
            // For simplicity, we'll rely on decision stump's internal random selection of features.

            SimpleDecisionTreeClassifier stump = new SimpleDecisionTreeClassifier(numClasses);
            stump.train(sampledFeatures, sampledLabels);
            trees.add(stump);
        }
    }

    @Override
    public int predict(int[] sample) {
        // Majority vote among all trees
        java.util.Map<Integer, Integer> counts = new java.util.HashMap<>();
        for (SimpleDecisionTreeClassifier tree : trees) {
            int p = tree.predict(sample);
            counts.put(p, counts.getOrDefault(p, 0) + 1);
        }

        int majorityClass = -1;
        int maxCount = -1;
        for (java.util.Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorityClass = entry.getKey();
            }
        }
        return majorityClass;
    }
}
