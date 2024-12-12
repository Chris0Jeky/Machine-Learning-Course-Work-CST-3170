public class Main {
    public static void main(String[] args) {
        System.out.println("Starting the machine learning project...");

        // ------------------------------------------------------------
        // EXPERIMENT 1: Nearest Neighbor Baseline
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment 1: Nearest Neighbor Baseline ===");

            // Load datasets fresh for this experiment
            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
            String[] classifierNames = {"Nearest Neighbor"};
            double[][] accuracies = new double[classifierNames.length][2];

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
        }

        // ------------------------------------------------------------
        // EXPERIMENT 2: Weighted k-NN
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment 2: Weighted k-NN ===");

            // Reload datasets for a clean start
            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
            String[] classifierNames = {"Weighted k-NN"};
            double[][] accuracies = new double[classifierNames.length][2];

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

                Classifier weightedKnn = new WeightedKNearestNeighborsClassifier(1, numClasses);
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
        }

        // ------------------------------------------------------------
        // EXPERIMENT 3: MLP Only (no scaling, no centroid features)
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment 3: MLP Only ===");

            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
            String[] classifierNames = {"MLP"};
            double[][] accuracies = new double[classifierNames.length][2];

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
        }

        // ------------------------------------------------------------
        // EXPERIMENT 4: SVM with Centroid Features
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment 4: SVM with Centroids ===");

            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
            String[] classifierNames = {"SVM with Centroids"};
            double[][] accuracies = new double[classifierNames.length][2];

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
        }

        // ------------------------------------------------------------
        // EXPERIMENT 5: Hybrid (NN+MLP)
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment 5: Hybrid (NN+MLP) ===");

            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
            String[] classifierNames = {"Hybrid (NN+MLP)"};
            double[][] accuracies = new double[classifierNames.length][2];

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

                // Train NN and MLP first
                nn.train(trainFeatures, trainLabels);
                mlp.train(trainFeatures, trainLabels);

                // Hybrid uses the trained nn and mlp internally.
                Classifier[] classifiers = {hybrid};

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
        }

        System.out.println("\nAll experiments completed successfully!");
    }
}
