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

        // Prepare features and labels for both datasets
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);

        // Number of classes
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1; // Assuming labels start from 0

        // Two-fold testing
        double[][] accuracies = new double[4][2]; // [classifier][fold]
        String[] classifierNames = {"Voting Classifier", "Multiclass Perceptron", "k-NN", "Nearest Neighbor"};

        for (int fold = 0; fold < 2; fold++) {
            System.out.println("\n=== Fold " + (fold + 1) + " ===");

            // Prepare training and testing data
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

            // Scale features
            System.out.println("Scaling features...");
            Utils.scaleFeatures(trainFeatures, testFeatures);

            // Compute centroids from training data
            System.out.println("Computing centroids...");
            double[][] centroids = Utils.computeCentroids(trainFeatures, trainLabels, numClasses);

            // Add centroid features to training and testing data
            System.out.println("Adding centroid-based features...");
            trainFeatures = Utils.addCentroidFeatures(trainFeatures, centroids);
            testFeatures = Utils.addCentroidFeatures(testFeatures, centroids);

            int featureSize = trainFeatures[0].length;

            // Initialize classifiers
            System.out.println("Initializing classifiers...");
            Classifier perceptron = new MulticlassPerceptronClassifier(1000, featureSize, numClasses);
            Classifier knn = new KNearestNeighborsClassifier(3, numClasses);
            Classifier nearestNeighbor = new NearestNeighborClassifier();
            Classifier votingClassifier = new VotingClassifier(new Classifier[]{perceptron, knn});

            Classifier[] classifiers = {votingClassifier, perceptron, knn, nearestNeighbor};

            // For each classifier, train and evaluate
            for (int i = 0; i < classifiers.length; i++) {
                System.out.println("\nTraining " + classifierNames[i] + "...");
                classifiers[i].train(trainFeatures, trainLabels);

                System.out.println("Evaluating " + classifierNames[i] + "...");
                int correctPredictions = 0;
                for (int j = 0; j < testFeatures.length; j++) {
                    int predictedLabel = classifiers[i].predict(testFeatures[j]);
                    if (predictedLabel == testLabels[j]) {
                        correctPredictions++;
                    }
                }
                double accuracy = (double) correctPredictions / testFeatures.length * 100;
                accuracies[i][fold] = accuracy;
                System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");
            }
        }

        // Calculate and display average accuracies
        System.out.println("\n=== Average Accuracies ===");
        for (int i = 0; i < 4; i++) {
            double averageAccuracy = (accuracies[i][0] + accuracies[i][1]) / 2;
            System.out.println(classifierNames[i] + " Average Accuracy: " + averageAccuracy + "%");
        }

        System.out.println("Two-fold testing completed successfully!");
    }
}
