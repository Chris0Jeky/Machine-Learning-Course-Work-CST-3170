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

        // Combine datasets
        System.out.println("Combining datasets...");
        int[][] combinedData = Utils.combineDatasets(dataSet1, dataSet2);
        System.out.println("Datasets combined successfully!");

// Shuffle combined dataset
        System.out.println("Shuffling the combined dataset...");
        Utils.shuffleDataset(combinedData);
        System.out.println("Dataset shuffled successfully!");

// Split into features and labels
        int[][] features = Utils.extractFeatures(combinedData);
        int[] labels = Utils.extractLabels(combinedData);

// Split into two folds
        System.out.println("Splitting dataset into two folds...");
        Fold[] folds = Utils.splitIntoFolds(features, labels, 2); // 2 folds
        System.out.println("Dataset split into folds successfully!");


        // Prepare features and labels
        int[][] trainingFeatures = Utils.extractFeatures(dataSet1);
        int[] trainingLabels = Utils.extractLabels(dataSet1);
        int[][] testFeatures = Utils.extractFeatures(dataSet2);
        int[] testLabels = Utils.extractLabels(dataSet2);

        // Number of classes
        int numClasses = Utils.getMaxLabel(trainingLabels) + 1; // Assuming labels start from 0

        // Initialize classifiers
        System.out.println("Initializing classifiers...");
        MulticlassSVMClassifier svm = new MulticlassSVMClassifier(0.001, 0.01, 1000, trainingFeatures[0].length, numClasses);
        MulticlassPerceptronClassifier perceptron = new MulticlassPerceptronClassifier(1000, trainingFeatures[0].length, numClasses);
        KNearestNeighborsClassifier knn = new KNearestNeighborsClassifier(3, numClasses); // k = 3
        NearestNeighborClassifier nearestNeighbor = new NearestNeighborClassifier();

        // Train and evaluate classifiers
        Classifier[] classifiers = {svm, perceptron, knn, nearestNeighbor};
        String[] classifierNames = {"Multiclass SVM", "Multiclass Perceptron", "k-NN", "Nearest Neighbor"};

        for (int i = 0; i < classifiers.length; i++) {
            System.out.println("\nTraining " + classifierNames[i] + "...");
            classifiers[i].train(trainingFeatures, trainingLabels);

            System.out.println("Evaluating " + classifierNames[i] + "...");
            int correctPredictions = 0;
            for (int j = 0; j < testFeatures.length; j++) {
                int predictedLabel = classifiers[i].predict(testFeatures[j]);
                if (predictedLabel == testLabels[j]) {
                    correctPredictions++;
                }
            }
            double accuracy = (double) correctPredictions / testFeatures.length * 100;
            System.out.println(classifierNames[i] + " Accuracy: " + accuracy + "%");
        }

        System.out.println("All classifiers evaluated successfully!");
    }
}
