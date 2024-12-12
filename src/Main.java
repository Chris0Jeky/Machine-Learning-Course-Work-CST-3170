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

        // Prepare features and labels (no scaling, no centroid features)
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);

        // Determine number of classes
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;

        // We'll use just three classifiers: Perceptron, k-NN, and Nearest Neighbor
        // This was stable and gave you near-baseline accuracy previously.

        // Two-fold testing
        // Classifier names and arrays must match in length
       // String[] classifierNames = {"Multiclass Perceptron", "k-NN", "Nearest Neighbor"};

        String[] classifierNames = {"Hybrid (NN+MLP)", "Nearest Neighbor"};
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
            // Compute centroids from training data
            double[][] centroids = Utils.computeCentroids(trainFeatures, trainLabels, numClasses);

// Add centroid features
            trainFeatures = Utils.addCentroidFeatures(trainFeatures, centroids);
            testFeatures = Utils.addCentroidFeatures(testFeatures, centroids);

            int featureSize = trainFeatures[0].length;

// Initialize a MulticlassSVMClassifier
            Classifier svmWithCentroids = new MulticlassSVMClassifier(0.001, 0.01, 1000, featureSize, numClasses);

// Train and test svmWithCentroids similarly to other classifiers.

//            // Initialize classifiers
//            System.out.println("Initializing classifiers...");
//            Classifier perceptron = new MulticlassPerceptronClassifier(200, featureSize, numClasses);
//            // Reduced epochs to 200 for speed - you can adjust if needed
//
//            Classifier knn = new KNearestNeighborsClassifier(3, numClasses);
//            Classifier nearestNeighbor = new NearestNeighborClassifier();
//
//            Classifier[] classifiers = {perceptron, knn, nearestNeighbor};

            // Suppose we pick distanceThreshold = 20 as a starting point
            Classifier nn = new NearestNeighborClassifier();
            Classifier mlp = new MLPClassifier(featureSize, 50, numClasses, 0.001, 50);
            Classifier hybrid = new HybridClassifier((NearestNeighborClassifier)nn, mlp, 20.0);

            Classifier[] classifiers = {hybrid, nn};

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
    }
}
