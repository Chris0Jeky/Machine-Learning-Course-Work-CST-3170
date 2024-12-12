public class Main {
    public static void main(String[] args) {
        System.out.println("Starting the machine learning project...");

        // Paths to CSV files
        String csvFileName1 = "datasets/dataSet1.csv";
        String csvFileName2 = "datasets/dataSet2.csv";

        // Load datasets
        System.out.println("Loading datasets...");
        int[][] dataSet1 = DataLoader.loadData(csvFileName1);
        int[][] dataSet2 = DataLoader.loadData(csvFileName2);
        System.out.println("Datasets loaded successfully!");

        // Extract original features and labels (no scaling, no centroid features)
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);

        // Determine number of classes
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
        // Baseline scenario: Just Nearest Neighbor

        // Two-fold testing with just Nearest Neighbor
        String[] classifierNames = {"Nearest Neighbor"};
        double[][] accuracies = new double[classifierNames.length][2];

        for (int fold = 0; fold < 2; fold++) {
            System.out.println("\n=== Fold " + (fold + 1) + " ===");

            // Prepare training and testing data for this fold
            int[][] trainFeatures, testFeatures;
            int[] trainLabels, testLabels;

            if (fold == 0) {
                // First fold: Train on dataSet1, test on dataSet2
                trainFeatures = features1;
                trainLabels = labels1;
                testFeatures = features2;
                testLabels = labels2;
            } else {
                // Second fold: Train on dataSet2, test on dataSet1
                trainFeatures = features2;
                trainLabels = labels2;
                testFeatures = features1;
                testLabels = labels1;
            }

            // Just Nearest Neighbor
            Classifier nn = new NearestNeighborClassifier();

            Classifier[] classifiers = {nn};

            // Train and evaluate each classifier
            for (int i = 0; i < classifiers.length; i++) {
                System.out.println("\nTraining " + classifierNames[i] + "...");
                long startTime = System.currentTimeMillis();
                classifiers[i].train(trainFeatures, trainLabels);
                long endTime = System.currentTimeMillis();
                System.out.println(classifierNames[i] + " training took: " + (endTime - startTime) + " ms");

                System.out.println("Evaluating " + classifierNames[i] + "...");
                int correctPredictions = 0;
                startTime = System.currentTimeMillis();
                for (int j = 0; j < testFeatures.length; j++) {
                    int predictedLabel = classifiers[i].predict(testFeatures[j]);
                    if (predictedLabel == testLabels[j]) {
                        correctPredictions++;
                    }
                }
                endTime = System.currentTimeMillis();
                System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                double accuracy = (double) correctPredictions / testFeatures.length * 100;
                accuracies[i][fold] = accuracy;
                System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");
            }
        }

        // Calculate and display average accuracies
        System.out.println("\n=== Average Accuracies ===");
        for (int i = 0; i < classifierNames.length; i++) {
            double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
            System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
        }

        System.out.println("Two-fold testing completed successfully!");

        // 1. Weighted K-NN:
        // Uncomment to try Weighted K-NN (immediately after original NN test)
        // Classifier weightedKnn = new WeightedKNearestNeighborsClassifier(3, numClasses);
        // Compare accuracy to NN:
        // Add "Weighted K-NN" to classifierNames and weightedKnn to classifiers array.

        // 2. MLP Classifier:
        // Add an MLPClassifier:
        // Classifier mlp = new MLPClassifier(featureSize, 100, numClasses, 0.002, 100);
        // Train mlp on trainFeatures (or scaled, if you implement scaling)
        // Compare accuracy to NN

        // 3. Centroid Features + SVM:
        // double[][] centroids = Utils.computeCentroids(trainFeatures, trainLabels, numClasses);
        // trainFeatures = Utils.addCentroidFeatures(trainFeatures, centroids);
        // testFeatures = Utils.addCentroidFeatures(testFeatures, centroids);
        // Then:
        // Classifier svmWithCentroids = new MulticlassSVMClassifier(0.001, 0.01, 1000, trainFeatures[0].length, numClasses);

        // 4. Scaling for MLP or SVM only:
        // int[][] trainFeaturesScaled = Utils.deepCopy(trainFeatures);
        // int[][] testFeaturesScaled = Utils.deepCopy(testFeatures);
        // Utils.scaleFeatures(trainFeaturesScaled, testFeaturesScaled);
        // mlp.train(trainFeaturesScaled, trainLabels);

        // 5. Hybrid (NN+MLP):
        // Classifier hybrid = new HybridClassifier((NearestNeighborClassifier) nn, mlp, 20.0);
        // Add "Hybrid (NN+MLP)" to classifierNames and hybrid to classifiers array.

        // Remember to uncomment corresponding lines and adjust arrays and loops as needed.
    }
}
