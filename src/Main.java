public class Main {
    public static void main(String[] args) {
        // Paths to CSV files
        String csvFileName1 = "datasets/dataSet1.csv";
        String csvFileName2 = "datasets/dataSet2.csv";

        // Load data
        int[][] dataSet1 = DataLoader.loadData(csvFileName1);
        int[][] dataSet2 = DataLoader.loadData(csvFileName2);

        // Prepare features and labels
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);

        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);

        // Nearest Neighbor Classification
        NearestNeighborClassifier nnClassifier = new NearestNeighborClassifier(features1, labels1);
        double nnAccuracy = evaluateClassifier(nnClassifier, features2, labels2);
        System.out.println("Nearest Neighbor Accuracy: " + nnAccuracy + "%\n");

        // k-Nearest Neighbors Classification
        int[] kValues = {1, 3, 5, 7};
        for (int k : kValues) {
            KNearestNeighborsClassifier knnClassifier = new KNearestNeighborsClassifier(features1, labels1, k);
            double knnAccuracy = evaluateClassifier(knnClassifier, features2, labels2);
            System.out.println("k-NN Accuracy with k=" + k + ": " + knnAccuracy + "%");
        }
    }

    // Method to evaluate a classifier and return accuracy
    private static double evaluateClassifier(Classifier classifier, int[][] testFeatures, int[] testLabels) {
        int correctPredictions = 0;
        for (int i = 0; i < testFeatures.length; i++) {
            int predictedLabel = classifier.predict(testFeatures[i]);
            if (predictedLabel == testLabels[i]) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / testFeatures.length * 100;
    }
}
