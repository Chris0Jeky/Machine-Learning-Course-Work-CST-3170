import java.util.*;

public class RandomForestClassifier implements Classifier {
    private int numTrees;
    private int numClasses;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private double maxFeaturesRatio;
    private List<DecisionTreeClassifier> trees;
    private double sampleRatio; // Ratio of samples to use for bootstrap

    public RandomForestClassifier(int numTrees, int numClasses, int maxDepth, int minSamplesSplit, int minSamplesLeaf, double maxFeaturesRatio, double sampleRatio) {
        this.numTrees = numTrees;
        this.numClasses = numClasses;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeaturesRatio = maxFeaturesRatio;
        this.sampleRatio = sampleRatio;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        trees = new ArrayList<>();
        int n = features.length;
        int d = features[0].length;
        int sampleSize = (int) (n * sampleRatio);
        int maxFeatures = (int) (d * maxFeaturesRatio);

        Random rand = new Random();

        for (int t = 0; t < numTrees; t++) {
            int[][] sampledFeatures = new int[sampleSize][d];
            int[] sampledLabels = new int[sampleSize];

            // Bootstrap sampling
            for (int i = 0; i < sampleSize; i++) {
                int idx = rand.nextInt(n);
                sampledFeatures[i] = features[idx];
                sampledLabels[i] = labels[idx];
            }

            DecisionTreeClassifier tree = new DecisionTreeClassifier(numClasses, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, rand);
            tree.train(sampledFeatures, sampledLabels);
            trees.add(tree);
        }
    }

    @Override
    public int predict(int[] sample) {
        // Aggregate predictions from all trees
        int[] classCounts = new int[numClasses];
        for (DecisionTreeClassifier tree : trees) {
            int prediction = tree.predict(sample);
            classCounts[prediction]++;
        }

        // Majority vote
        int bestClass = 0;
        int bestCount = classCounts[0];
        for (int c = 1; c < numClasses; c++) {
            if (classCounts[c] > bestCount) {
                bestCount = classCounts[c];
                bestClass = c;
            }
        }
        return bestClass;
    }
}
