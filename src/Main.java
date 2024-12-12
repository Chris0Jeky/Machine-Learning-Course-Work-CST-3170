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

        // Two-fold testing variables
        String[] classifierNames;
        double[][] accuracies;

        // NOTE:
        // We'll have multiple sections below, each dedicated to a single experiment.
        // Uncomment ONE section at a time and run. This ensures no interference.

        // ------------------------------------------------------------
        // SECTION 1: Nearest Neighbor (Baseline)
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Running Nearest Neighbor Baseline ===");
            classifierNames = new String[]{"Nearest Neighbor"};
            accuracies = new double[classifierNames.length][2];

            for (int fold = 0; fold < 2; fold++) {
                System.out.println("\n=== Fold " + (fold + 1) + " ===");
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

                Classifier nn = new NearestNeighborClassifier();
                Classifier[] classifiers = {nn};

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

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }

            System.out.println("Two-fold testing completed successfully!");
        }

        // ------------------------------------------------------------
        // SECTION 2: Weighted k-NN (Uncomment to run)
        // ------------------------------------------------------------
        /*
        {
            System.out.println("\n=== Running Weighted k-NN ===");
            classifierNames = new String[]{"Weighted k-NN"};
            accuracies = new double[classifierNames.length][2];

            for (int fold = 0; fold < 2; fold++) {
                System.out.println("\n=== Fold " + (fold + 1) + " ===");
                int[][] trainFeatures, testFeatures;
                int[] trainLabels, testLabels;

                if (fold == 0) {
                    trainFeatures = features1;
                    trainLabels = labels1;
                    testFeatures = features2;
                    testLabels = labels2;
                } else {
                    trainFeatures = features2;
                    trainLabels = labels2;
                    testFeatures = features1;
                    testLabels = labels1;
                }

                Classifier weightedKnn = new WeightedKNearestNeighborsClassifier(3, numClasses);
                Classifier[] classifiers = {weightedKnn};

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

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }

            System.out.println("Two-fold testing completed successfully!");
        }
        */

        // ------------------------------------------------------------
        // SECTION 3: MLP only (without scaling/centroids)
        // ------------------------------------------------------------
        /*
        {
            System.out.println("\n=== Running MLP Only ===");
            classifierNames = new String[]{"MLP"};
            accuracies = new double[classifierNames.length][2];

            for (int fold = 0; fold < 2; fold++) {
                System.out.println("\n=== Fold " + (fold + 1) + " ===");
                int[][] trainFeatures, testFeatures;
                int[] trainLabels, testLabels;

                if (fold == 0) {
                    trainFeatures = features1;
                    trainLabels = labels1;
                    testFeatures = features2;
                    testLabels = labels2;
                } else {
                    trainFeatures = features2;
                    trainLabels = labels2;
                    testFeatures = features1;
                    testLabels = labels1;
                }

                int featureSize = trainFeatures[0].length;
                Classifier mlp = new MLPClassifier(featureSize, 100, numClasses, 0.002, 100);

                Classifier[] classifiers = {mlp};

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

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }

            System.out.println("Two-fold testing completed successfully!");
        }
        */

        // ------------------------------------------------------------
        // SECTION 4: SVM with Centroid Features
        // ------------------------------------------------------------
        /*
        {
            System.out.println("\n=== Running SVM with Centroid Features ===");
            classifierNames = new String[]{"SVM with Centroids"};
            accuracies = new double[classifierNames.length][2];

            for (int fold = 0; fold < 2; fold++) {
                System.out.println("\n=== Fold " + (fold + 1) + " ===");
                int[][] trainFeatures, testFeatures;
                int[] trainLabels, testLabels;

                if (fold == 0) {
                    trainFeatures = features1;
                    trainLabels = labels1;
                    testFeatures = features2;
                    testLabels = labels2;
                } else {
                    trainFeatures = features2;
                    trainLabels = labels2;
                    testFeatures = features1;
                    testLabels = labels1;
                }

                // Compute centroids and add them
                double[][] centroids = Utils.computeCentroids(trainFeatures, trainLabels, numClasses);
                trainFeatures = Utils.addCentroidFeatures(trainFeatures, centroids);
                testFeatures = Utils.addCentroidFeatures(testFeatures, centroids);

                int featureSize = trainFeatures[0].length;
                Classifier svmWithCentroids = new MulticlassSVMClassifier(0.001, 0.01, 1000, featureSize, numClasses);

                Classifier[] classifiers = {svmWithCentroids};

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

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }

            System.out.println("Two-fold testing completed successfully!");
        }
        */

        // ------------------------------------------------------------
        // SECTION 5: Hybrid (NN+MLP)
        // ------------------------------------------------------------
        /*
        {
            System.out.println("\n=== Running Hybrid (NN+MLP) ===");
            classifierNames = new String[]{"Hybrid (NN+MLP)"};
            accuracies = new double[classifierNames.length][2];

            for (int fold = 0; fold < 2; fold++) {
                System.out.println("\n=== Fold " + (fold + 1) + " ===");
                int[][] trainFeatures, testFeatures;
                int[] trainLabels, testLabels;

                if (fold == 0) {
                    trainFeatures = features1;
                    trainLabels = labels1;
                    testFeatures = features2;
                    testLabels = labels2;
                } else {
                    trainFeatures = features2;
                    trainLabels = labels2;
                    testFeatures = features1;
                    testLabels = labels1;
                }

                int featureSize = trainFeatures[0].length;
                Classifier nn = new NearestNeighborClassifier();
                Classifier mlp = new MLPClassifier(featureSize, 100, numClasses, 0.002, 100);
                double threshold = 20.0;
                Classifier hybrid = new HybridClassifier((NearestNeighborClassifier) nn, mlp, threshold);

                Classifier[] classifiers = {hybrid};

                // Train NN and MLP separately if needed:
                nn.train(trainFeatures, trainLabels);
                mlp.train(trainFeatures, trainLabels);

                // The hybrid might just rely on these trained classifiers:
                // If hybrid calls nn and mlp internally without separate training,
                // consider modifying HybridClassifier to store references.

                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("\nEvaluating " + classifierNames[i] + "...");
                    int correctPredictions = 0;
                    long startTime = System.currentTimeMillis();
                    for (int j = 0; j < testFeatures.length; j++) {
                        int predictedLabel = classifiers[i].predict(testFeatures[j]);
                        if (predictedLabel == testLabels[j]) {
                            correctPredictions++;
                        }
                    }
                    long endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                    double accuracy = (double) correctPredictions / testFeatures.length * 100;
                    accuracies[i][fold] = accuracy;
                    System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");
                }
            }

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }

            System.out.println("Two-fold testing completed successfully!");
        }
        */

        // After confirming each section works individually, you can combine or further experiment.
    }
}
