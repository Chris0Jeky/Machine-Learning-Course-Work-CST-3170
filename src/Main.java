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

            // Two-fold testing: first fold trains on dataSet1, tests on dataSet2; second fold reverses it.
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

                // Evaluate each classifier in this experiment
                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("\nTraining " + classifierNames[i] + "...");
                    long startTime = System.currentTimeMillis();
                    classifiers[i].train(trainFeatures, trainLabels);
                    long endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " training took: " + (endTime - startTime) + " ms");

                    System.out.println("Evaluating " + classifierNames[i] + "...");

                    int[][] confusionMatrix = new int[numClasses][numClasses];

                    int correctPredictions = 0;
                    startTime = System.currentTimeMillis();
                    // Test the classifier on the test set
                    for (int j = 0; j < testFeatures.length; j++) {
                        int predictedLabel = classifiers[i].predict(testFeatures[j]);
                        int actualLabel = testLabels[j];
                        if (predictedLabel == testLabels[j]) {
                            correctPredictions++;
                        }
                        confusionMatrix[actualLabel][predictedLabel]++;
                    }
                    endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                    double accuracy = (double) correctPredictions / testFeatures.length * 100;
                    accuracies[i][fold] = accuracy;
                    System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");

                    System.out.println("Confusion Matrix for " + classifierNames[i] + ":");
                    for (int a = 0; a < numClasses; a++) {
                        for (int p = 0; p < numClasses; p++) {
                            System.out.print(confusionMatrix[a][p] + " ");
                        }
                        System.out.println();
                    }
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

            // Two-fold cross-validation again
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

                // Weighted k-NN with k=4 here just as a placeholder, could try different ks (although 4 seems to work best)
                Classifier weightedKnn = new WeightedKNearestNeighborsClassifier(4, numClasses);
                Classifier[] classifiers = {weightedKnn};

                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("\nTraining " + classifierNames[i] + "...");
                    long startTime = System.currentTimeMillis();
                    classifiers[i].train(trainFeatures, trainLabels);
                    long endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " training took: " + (endTime - startTime) + " ms");

                    System.out.println("Evaluating " + classifierNames[i] + "...");

                    int[][] confusionMatrix = new int[numClasses][numClasses];

                    int correctPredictions = 0;
                    startTime = System.currentTimeMillis();
                    // Evaluate on test set
                    for (int j = 0; j < testFeatures.length; j++) {
                        int predictedLabel = classifiers[i].predict(testFeatures[j]);
                        int actualLabel = testLabels[j];
                        if (predictedLabel == testLabels[j]) {
                            correctPredictions++;
                        }
                        confusionMatrix[actualLabel][predictedLabel]++;
                    }
                    endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                    double accuracy = (double) correctPredictions / testFeatures.length * 100;
                    accuracies[i][fold] = accuracy;
                    System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");

                    System.out.println("Confusion Matrix for " + classifierNames[i] + ":");
                    for (int a = 0; a < numClasses; a++) {
                        for (int p = 0; p < numClasses; p++) {
                            System.out.print(confusionMatrix[a][p] + " ");
                        }
                        System.out.println();
                    }
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

            // Two-fold test again
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
                // MLP with a hidden layer of 100, output = numClasses
                Classifier mlp = new MLPClassifier(featureSize, 100, numClasses, 0.002, 100);

                Classifier[] classifiers = {mlp};

                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("\nTraining " + classifierNames[i] + "...");
                    long startTime = System.currentTimeMillis();
                    classifiers[i].train(trainFeatures, trainLabels);
                    long endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " training took: " + (endTime - startTime) + " ms");

                    System.out.println("Evaluating " + classifierNames[i] + "...");

                    int[][] confusionMatrix = new int[numClasses][numClasses];

                    int correctPredictions = 0;
                    startTime = System.currentTimeMillis();
                    for (int j = 0; j < testFeatures.length; j++) {
                        int predictedLabel = classifiers[i].predict(testFeatures[j]);
                        int actualLabel = testLabels[j];
                        if (predictedLabel == testLabels[j]) {
                            correctPredictions++;
                        }
                        confusionMatrix[actualLabel][predictedLabel]++;
                    }
                    endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                    double accuracy = (double) correctPredictions / testFeatures.length * 100;
                    accuracies[i][fold] = accuracy;
                    System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");

                    System.out.println("Confusion Matrix for " + classifierNames[i] + ":");
                    for (int a = 0; a < numClasses; a++) {
                        for (int p = 0; p < numClasses; p++) {
                            System.out.print(confusionMatrix[a][p] + " ");
                        }
                        System.out.println();
                    }
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

                // Compute centroids and add as extra features
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

                    int[][] confusionMatrix = new int[numClasses][numClasses];

                    int correctPredictions = 0;
                    startTime = System.currentTimeMillis();
                    for (int j = 0; j < testFeatures.length; j++) {
                        int predictedLabel = classifiers[i].predict(testFeatures[j]);
                        int actualLabel = testLabels[j];
                        if (predictedLabel == testLabels[j]) {
                            correctPredictions++;
                        }
                        confusionMatrix[actualLabel][predictedLabel]++;
                    }
                    endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                    double accuracy = (double) correctPredictions / testFeatures.length * 100;
                    accuracies[i][fold] = accuracy;
                    System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");

                    System.out.println("Confusion Matrix for " + classifierNames[i] + ":");
                    for (int a = 0; a < numClasses; a++) {
                        for (int p = 0; p < numClasses; p++) {
                            System.out.print(confusionMatrix[a][p] + " ");
                        }
                        System.out.println();
                    }
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
                // Hybrid: If NN distance > threshold, fallback to MLP
                Classifier hybrid = new HybridClassifier((NearestNeighborClassifier) nn, mlp, threshold);

                // Train NN and MLP first
                nn.train(trainFeatures, trainLabels);
                mlp.train(trainFeatures, trainLabels);

                // Hybrid uses the trained nn and mlp internally.
                Classifier[] classifiers = {hybrid};

                for (int i = 0; i < classifiers.length; i++) {
                    System.out.println("\nEvaluating " + classifierNames[i] + "...");

                    int[][] confusionMatrix = new int[numClasses][numClasses];

                    int correctPredictions = 0;
                    long startTime = System.currentTimeMillis();
                    for (int j = 0; j < testFeatures.length; j++) {
                        int predictedLabel = classifiers[i].predict(testFeatures[j]);
                        int actualLabel = testLabels[j];
                        if (predictedLabel == testLabels[j]) {
                            correctPredictions++;
                        }
                        confusionMatrix[actualLabel][predictedLabel]++;
                    }
                    long endTime = System.currentTimeMillis();
                    System.out.println(classifierNames[i] + " evaluation took: " + (endTime - startTime) + " ms");

                    double accuracy = (double) correctPredictions / testFeatures.length * 100;
                    accuracies[i][fold] = accuracy;
                    System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");

                    System.out.println("Confusion Matrix for " + classifierNames[i] + ":");
                    for (int a = 0; a < numClasses; a++) {
                        for (int p = 0; p < numClasses; p++) {
                            System.out.print(confusionMatrix[a][p] + " ");
                        }
                        System.out.println();
                    }
                }
            }

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }
        }

        // ------------------------------------------------------------
        // EXPERIMENT: Simple Voting Classifier
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment: Simple Voting Classifier ===");

            // Reload datasets for a clean test
            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;

            // Let's combine 3 classifiers: NearestNeighbor, MLP, WeightedKNN
            String[] classifierNames = {"Simple Voting Classifier"};
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

                // Initialize individual classifiers
                // Voting ensemble of NN, MLP, and WeightedKNN
                Classifier nn = new NearestNeighborClassifier();
                Classifier mlp = new MLPClassifier(featureSize, 100, numClasses, 0.002, 100);
                Classifier weightedKnn = new WeightedKNearestNeighborsClassifier(3, numClasses);

                // Train each classifier individually
                System.out.println("\nTraining base classifiers for voting...");
                long startTime = System.currentTimeMillis();
                nn.train(trainFeatures, trainLabels);
                mlp.train(trainFeatures, trainLabels);
                weightedKnn.train(trainFeatures, trainLabels);
                long endTime = System.currentTimeMillis();
                System.out.println("Base classifiers training took: " + (endTime - startTime) + " ms");

                // Create a SimpleVotingClassifier with these base classifiers
                Classifier voting = new SimpleVotingClassifier(new Classifier[]{nn, mlp, weightedKnn});

                System.out.println("\nEvaluating " + classifierNames[0] + "...");

                int[][] confusionMatrix = new int[numClasses][numClasses];

                int correctPredictions = 0;
                startTime = System.currentTimeMillis();
                for (int j = 0; j < testFeatures.length; j++) {
                    int predictedLabel = voting.predict(testFeatures[j]);
                    int actualLabel = testLabels[j];
                    if (predictedLabel == testLabels[j]) {
                        correctPredictions++;
                    }
                    confusionMatrix[actualLabel][predictedLabel]++;
                }
                endTime = System.currentTimeMillis();
                System.out.println(classifierNames[0] + " evaluation took: " + (endTime - startTime) + " ms");

                double accuracy = (double) correctPredictions / testFeatures.length * 100;
                accuracies[0][fold] = accuracy;
                System.out.println(classifierNames[0] + " Accuracy: " + accuracy + "%");

                System.out.println("Confusion Matrix for " + classifierNames[0] + ":");
                for (int a = 0; a < numClasses; a++) {
                    for (int p = 0; p < numClasses; p++) {
                        System.out.print(confusionMatrix[a][p] + " ");
                    }
                    System.out.println();
                }
            }

            System.out.println("\n=== Average Accuracies ===");
            for (int i = 0; i < classifierNames.length; i++) {
                double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
                System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
            }
        }

        // ------------------------------------------------------------
        // EXPERIMENT: Random Forest (Simplified)
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment: Random Forest (Simplified) ===");

            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
            String[] classifierNames = {"Random Forest"};
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

                // Let's say we want 10 trees, each stump tries random features internally.
                // We'll use a sampleRatio = 1.0 (bootstrap same size as dataset), featureRatio = 1.0
                // since we rely on stump randomness.
                Classifier rf = new RandomForestClassifier(
                        10,        // numTrees
                        numClasses,
                        5,         // maxDepth
                        2,         // minSamplesSplit
                        1,         // minSamplesLeaf
                        0.5,       // maxFeaturesRatio
                        1.0        // sampleRatio
                );
                Classifier[] classifiers = {rf};

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
        // EXPERIMENT: Multi-class Gradient Boosted Trees with Softmax
        // ------------------------------------------------------------
        {
            System.out.println("\n=== Experiment: Multi-class Gradient Boosted Trees ===");

            int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
            int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");

            int[][] features1 = Utils.extractFeatures(dataSet1);
            int[] labels1 = Utils.extractLabels(dataSet1);
            int[][] features2 = Utils.extractFeatures(dataSet2);
            int[] labels2 = Utils.extractLabels(dataSet2);

            int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;

            String[] classifierNames = {"Multi-class GBT"};
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

                // Choose parameters
                MultiClassGradientBoostedTreesClassifier mcGBT = new MultiClassGradientBoostedTreesClassifier(
                        numClasses,
                        50,    // numTrees
                        0.1,   // eta
                        3,     // maxDepth
                        5,     // minSamplesLeaf
                        10,    // minSamplesSplit
                        1.0    // lambda (regularization)
                );

                mcGBT.train(trainFeatures, trainLabels);

                int correctPredictions = 0;
                for (int i = 0; i < testFeatures.length; i++) {
                    int pred = mcGBT.predict(testFeatures[i]);
                    if (pred == testLabels[i]) correctPredictions++;
                }

                double accuracy = (double)correctPredictions / testFeatures.length * 100;
                accuracies[0][fold] = accuracy;
                System.out.println(classifierNames[0] + " Accuracy: " + accuracy + "%");
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
