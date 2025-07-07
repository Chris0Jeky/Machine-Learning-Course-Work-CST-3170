import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

public class Main {
    private static final String RESULTS_DIR = "results";
    private static PrintWriter resultWriter;
    private static List<ExperimentResult> allResults = new ArrayList<>();
    
    static class ExperimentResult {
        String name;
        double avgAccuracy;
        long avgTrainingTime;
        long avgEvaluationTime;
        
        ExperimentResult(String name, double avgAccuracy, long avgTrainingTime, long avgEvaluationTime) {
            this.name = name;
            this.avgAccuracy = avgAccuracy;
            this.avgTrainingTime = avgTrainingTime;
            this.avgEvaluationTime = avgEvaluationTime;
        }
    }
    
    public static void main(String[] args) {
        createResultsDirectory();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String resultFile = RESULTS_DIR + "/experiment_results_" + timestamp + ".txt";
        
        try {
            resultWriter = new PrintWriter(new FileWriter(resultFile));
            
            System.out.println("==============================================");
            System.out.println("    Machine Learning Classifier Comparison    ");
            System.out.println("==============================================");
            System.out.println("Dataset: Handwritten Digit Recognition (8x8)");
            System.out.println("Evaluation: 2-Fold Cross Validation");
            System.out.println("Results saved to: " + resultFile);
            System.out.println("==============================================\n");
            
            printAndLog("==============================================");
            printAndLog("    Machine Learning Classifier Comparison    ");
            printAndLog("==============================================");
            printAndLog("Dataset: Handwritten Digit Recognition (8x8)");
            printAndLog("Evaluation: 2-Fold Cross Validation");
            printAndLog("Date: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()));
            printAndLog("==============================================\n");
            
            // Run all experiments
            runNearestNeighborExperiment();
            runKNNExperiments();
            runNeuralNetworkExperiments();
            runSVMExperiments();
            runTreeBasedExperiments();
            runEnsembleExperiments();
            
            // Print summary
            printSummary();
            
            resultWriter.close();
            System.out.println("\nResults saved to: " + resultFile);
            
        } catch (IOException e) {
            System.err.println("Error writing results: " + e.getMessage());
        }
    }
    
    private static void createResultsDirectory() {
        File dir = new File(RESULTS_DIR);
        if (!dir.exists()) {
            dir.mkdir();
        }
    }
    
    private static void printAndLog(String message) {
        System.out.println(message);
        if (resultWriter != null) {
            resultWriter.println(message);
            resultWriter.flush();
        }
    }
    
    private static void runNearestNeighborExperiment() {
        printAndLog("\n========== NEAREST NEIGHBOR CLASSIFIERS ==========");
        
        // Basic Nearest Neighbor
        runExperiment("1-Nearest Neighbor", 
            (features, labels, numClasses) -> new NearestNeighborClassifier());
    }
    
    private static void runKNNExperiments() {
        printAndLog("\n========== K-NEAREST NEIGHBORS CLASSIFIERS ==========");
        
        // Weighted k-NN with different k values
        int[] kValues = {3, 4, 5};
        for (int k : kValues) {
            runExperiment("Weighted " + k + "-NN", 
                (features, labels, numClasses) -> new WeightedKNearestNeighborsClassifier(k, numClasses));
        }
    }
    
    private static void runNeuralNetworkExperiments() {
        printAndLog("\n========== NEURAL NETWORK CLASSIFIERS ==========");
        
        // Multi-layer Perceptron
        runExperiment("MLP (100 hidden units)", 
            (features, labels, numClasses) -> new MLPClassifier(features[0].length, 100, numClasses, 0.002, 100));
    }
    
    private static void runSVMExperiments() {
        printAndLog("\n========== SUPPORT VECTOR MACHINES ==========");
        
        // Linear SVM
        runExperiment("Linear SVM", 
            (features, labels, numClasses) -> new MulticlassSVMClassifier(0.001, 0.01, 1000, features[0].length, numClasses));
        
        // SVM with centroid features
        runExperimentWithCentroidFeatures("SVM with Centroid Features");
        
        // Kernel SVM with RBF
        runExperiment("RBF Kernel SVM", 
            (features, labels, numClasses) -> new MulticlassKernelSVMClassifier(new RBFKernel(0.1), 0.001, 0.01, 500, features[0].length, numClasses));
    }
    
    private static void runTreeBasedExperiments() {
        printAndLog("\n========== TREE-BASED CLASSIFIERS ==========");
        
        // Decision Tree
        runExperiment("Decision Tree", 
            (features, labels, numClasses) -> new DecisionTreeClassifier(numClasses, 10, 5, 1, features[0].length, new Random(42)));
        
        // Random Forest
        runExperiment("Random Forest (10 trees)", 
            (features, labels, numClasses) -> new RandomForestClassifier(10, numClasses, 5, 2, 1, 0.5, 1.0));
        
        // Gradient Boosted Trees
        runExperiment("Gradient Boosted Trees", 
            (features, labels, numClasses) -> new MultiClassGradientBoostedTreesClassifier(numClasses, 50, 0.1, 3, 5, 10, 1.0));
    }
    
    private static void runEnsembleExperiments() {
        printAndLog("\n========== ENSEMBLE CLASSIFIERS ==========");
        
        // Voting Classifier
        runVotingClassifierExperiment();
        
        // Hybrid Classifier
        runHybridClassifierExperiment();
    }
    
    private static void runExperiment(String experimentName, ClassifierFactory factory) {
        printAndLog("\n--- " + experimentName + " ---");
        
        // Load datasets
        int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
        int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");
        
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);
        
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
        double[] accuracies = new double[2];
        long[] trainingTimes = new long[2];
        long[] evaluationTimes = new long[2];
        
        // Two-fold cross-validation
        for (int fold = 0; fold < 2; fold++) {
            printAndLog("\nFold " + (fold + 1) + ":");
            
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
            
            // Create classifier
            Classifier classifier = factory.create(trainFeatures, trainLabels, numClasses);
            
            // Train
            printAndLog("  Training...");
            long startTime = System.currentTimeMillis();
            classifier.train(trainFeatures, trainLabels);
            long endTime = System.currentTimeMillis();
            trainingTimes[fold] = endTime - startTime;
            printAndLog("  Training time: " + trainingTimes[fold] + " ms");
            
            // Evaluate
            printAndLog("  Evaluating...");
            int[][] confusionMatrix = new int[numClasses][numClasses];
            int correctPredictions = 0;
            
            startTime = System.currentTimeMillis();
            for (int i = 0; i < testFeatures.length; i++) {
                int predicted = classifier.predict(testFeatures[i]);
                int actual = testLabels[i];
                if (predicted == actual) {
                    correctPredictions++;
                }
                confusionMatrix[actual][predicted]++;
            }
            endTime = System.currentTimeMillis();
            evaluationTimes[fold] = endTime - startTime;
            
            accuracies[fold] = (double) correctPredictions / testFeatures.length * 100;
            printAndLog("  Accuracy: " + String.format("%.2f%%", accuracies[fold]));
            printAndLog("  Evaluation time: " + evaluationTimes[fold] + " ms");
        }
        
        // Calculate averages
        double avgAccuracy = (accuracies[0] + accuracies[1]) / 2;
        long avgTrainingTime = (trainingTimes[0] + trainingTimes[1]) / 2;
        long avgEvaluationTime = (evaluationTimes[0] + evaluationTimes[1]) / 2;
        
        printAndLog("\nAverage Results:");
        printAndLog("  Average Accuracy: " + String.format("%.2f%%", avgAccuracy));
        printAndLog("  Average Training Time: " + avgTrainingTime + " ms");
        printAndLog("  Average Evaluation Time: " + avgEvaluationTime + " ms");
        
        allResults.add(new ExperimentResult(experimentName, avgAccuracy, avgTrainingTime, avgEvaluationTime));
    }
    
    private static void runExperimentWithCentroidFeatures(String experimentName) {
        printAndLog("\n--- " + experimentName + " ---");
        
        // Load datasets
        int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
        int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");
        
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);
        
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
        double[] accuracies = new double[2];
        long[] trainingTimes = new long[2];
        long[] evaluationTimes = new long[2];
        
        for (int fold = 0; fold < 2; fold++) {
            printAndLog("\nFold " + (fold + 1) + ":");
            
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
            
            // Add centroid features
            double[][] centroids = Utils.computeCentroids(trainFeatures, trainLabels, numClasses);
            trainFeatures = Utils.addCentroidFeatures(trainFeatures, centroids);
            testFeatures = Utils.addCentroidFeatures(testFeatures, centroids);
            
            // Create classifier
            Classifier classifier = new MulticlassSVMClassifier(0.001, 0.01, 1000, trainFeatures[0].length, numClasses);
            
            // Train
            printAndLog("  Training...");
            long startTime = System.currentTimeMillis();
            classifier.train(trainFeatures, trainLabels);
            long endTime = System.currentTimeMillis();
            trainingTimes[fold] = endTime - startTime;
            printAndLog("  Training time: " + trainingTimes[fold] + " ms");
            
            // Evaluate
            printAndLog("  Evaluating...");
            int correctPredictions = 0;
            
            startTime = System.currentTimeMillis();
            for (int i = 0; i < testFeatures.length; i++) {
                int predicted = classifier.predict(testFeatures[i]);
                if (predicted == testLabels[i]) {
                    correctPredictions++;
                }
            }
            endTime = System.currentTimeMillis();
            evaluationTimes[fold] = endTime - startTime;
            
            accuracies[fold] = (double) correctPredictions / testFeatures.length * 100;
            printAndLog("  Accuracy: " + String.format("%.2f%%", accuracies[fold]));
            printAndLog("  Evaluation time: " + evaluationTimes[fold] + " ms");
        }
        
        // Calculate averages
        double avgAccuracy = (accuracies[0] + accuracies[1]) / 2;
        long avgTrainingTime = (trainingTimes[0] + trainingTimes[1]) / 2;
        long avgEvaluationTime = (evaluationTimes[0] + evaluationTimes[1]) / 2;
        
        printAndLog("\nAverage Results:");
        printAndLog("  Average Accuracy: " + String.format("%.2f%%", avgAccuracy));
        printAndLog("  Average Training Time: " + avgTrainingTime + " ms");
        printAndLog("  Average Evaluation Time: " + avgEvaluationTime + " ms");
        
        allResults.add(new ExperimentResult(experimentName, avgAccuracy, avgTrainingTime, avgEvaluationTime));
    }
    
    private static void runVotingClassifierExperiment() {
        printAndLog("\n--- Voting Classifier (NN + MLP + Weighted k-NN) ---");
        
        // Load datasets
        int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
        int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");
        
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);
        
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
        double[] accuracies = new double[2];
        long[] trainingTimes = new long[2];
        long[] evaluationTimes = new long[2];
        
        for (int fold = 0; fold < 2; fold++) {
            printAndLog("\nFold " + (fold + 1) + ":");
            
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
            
            // Create base classifiers
            Classifier nn = new NearestNeighborClassifier();
            Classifier mlp = new MLPClassifier(trainFeatures[0].length, 100, numClasses, 0.002, 100);
            Classifier weightedKnn = new WeightedKNearestNeighborsClassifier(3, numClasses);
            
            // Train base classifiers
            printAndLog("  Training base classifiers...");
            long startTime = System.currentTimeMillis();
            nn.train(trainFeatures, trainLabels);
            mlp.train(trainFeatures, trainLabels);
            weightedKnn.train(trainFeatures, trainLabels);
            long endTime = System.currentTimeMillis();
            trainingTimes[fold] = endTime - startTime;
            printAndLog("  Training time: " + trainingTimes[fold] + " ms");
            
            // Create voting classifier
            Classifier voting = new SimpleVotingClassifier(new Classifier[]{nn, mlp, weightedKnn});
            
            // Evaluate
            printAndLog("  Evaluating...");
            int correctPredictions = 0;
            
            startTime = System.currentTimeMillis();
            for (int i = 0; i < testFeatures.length; i++) {
                int predicted = voting.predict(testFeatures[i]);
                if (predicted == testLabels[i]) {
                    correctPredictions++;
                }
            }
            endTime = System.currentTimeMillis();
            evaluationTimes[fold] = endTime - startTime;
            
            accuracies[fold] = (double) correctPredictions / testFeatures.length * 100;
            printAndLog("  Accuracy: " + String.format("%.2f%%", accuracies[fold]));
            printAndLog("  Evaluation time: " + evaluationTimes[fold] + " ms");
        }
        
        // Calculate averages
        double avgAccuracy = (accuracies[0] + accuracies[1]) / 2;
        long avgTrainingTime = (trainingTimes[0] + trainingTimes[1]) / 2;
        long avgEvaluationTime = (evaluationTimes[0] + evaluationTimes[1]) / 2;
        
        printAndLog("\nAverage Results:");
        printAndLog("  Average Accuracy: " + String.format("%.2f%%", avgAccuracy));
        printAndLog("  Average Training Time: " + avgTrainingTime + " ms");
        printAndLog("  Average Evaluation Time: " + avgEvaluationTime + " ms");
        
        allResults.add(new ExperimentResult("Voting Classifier", avgAccuracy, avgTrainingTime, avgEvaluationTime));
    }
    
    private static void runHybridClassifierExperiment() {
        printAndLog("\n--- Hybrid Classifier (NN + MLP with distance threshold) ---");
        
        // Load datasets
        int[][] dataSet1 = DataLoader.loadData("datasets/dataSet1.csv");
        int[][] dataSet2 = DataLoader.loadData("datasets/dataSet2.csv");
        
        int[][] features1 = Utils.extractFeatures(dataSet1);
        int[] labels1 = Utils.extractLabels(dataSet1);
        int[][] features2 = Utils.extractFeatures(dataSet2);
        int[] labels2 = Utils.extractLabels(dataSet2);
        
        int numClasses = Utils.getMaxLabel(labels1, labels2) + 1;
        double[] accuracies = new double[2];
        long[] evaluationTimes = new long[2];
        
        for (int fold = 0; fold < 2; fold++) {
            printAndLog("\nFold " + (fold + 1) + ":");
            
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
            
            // Create and train base classifiers
            NearestNeighborClassifier nn = new NearestNeighborClassifier();
            Classifier mlp = new MLPClassifier(trainFeatures[0].length, 100, numClasses, 0.002, 100);
            
            printAndLog("  Training base classifiers...");
            nn.train(trainFeatures, trainLabels);
            mlp.train(trainFeatures, trainLabels);
            
            // Create hybrid classifier
            double threshold = 20.0;
            Classifier hybrid = new HybridClassifier(nn, mlp, threshold);
            
            // Evaluate
            printAndLog("  Evaluating...");
            int correctPredictions = 0;
            
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < testFeatures.length; i++) {
                int predicted = hybrid.predict(testFeatures[i]);
                if (predicted == testLabels[i]) {
                    correctPredictions++;
                }
            }
            long endTime = System.currentTimeMillis();
            evaluationTimes[fold] = endTime - startTime;
            
            accuracies[fold] = (double) correctPredictions / testFeatures.length * 100;
            printAndLog("  Accuracy: " + String.format("%.2f%%", accuracies[fold]));
            printAndLog("  Evaluation time: " + evaluationTimes[fold] + " ms");
        }
        
        // Calculate averages
        double avgAccuracy = (accuracies[0] + accuracies[1]) / 2;
        long avgEvaluationTime = (evaluationTimes[0] + evaluationTimes[1]) / 2;
        
        printAndLog("\nAverage Results:");
        printAndLog("  Average Accuracy: " + String.format("%.2f%%", avgAccuracy));
        printAndLog("  Average Evaluation Time: " + avgEvaluationTime + " ms");
        
        allResults.add(new ExperimentResult("Hybrid Classifier", avgAccuracy, 0, avgEvaluationTime));
    }
    
    private static void printSummary() {
        printAndLog("\n==============================================");
        printAndLog("              FINAL SUMMARY                   ");
        printAndLog("==============================================");
        
        // Sort by accuracy
        allResults.sort((a, b) -> Double.compare(b.avgAccuracy, a.avgAccuracy));
        
        printAndLog("\nClassifier Rankings by Accuracy:");
        printAndLog("------------------------------------------------");
        int rank = 1;
        for (ExperimentResult result : allResults) {
            printAndLog(String.format("%2d. %-35s %.2f%%", 
                rank++, result.name, result.avgAccuracy));
        }
        
        // Find best and worst
        ExperimentResult best = allResults.get(0);
        ExperimentResult worst = allResults.get(allResults.size() - 1);
        
        printAndLog("\n------------------------------------------------");
        printAndLog("Best Performer:  " + best.name + " (" + String.format("%.2f%%", best.avgAccuracy) + ")");
        printAndLog("Worst Performer: " + worst.name + " (" + String.format("%.2f%%", worst.avgAccuracy) + ")");
        printAndLog("Performance Gap: " + String.format("%.2f%%", best.avgAccuracy - worst.avgAccuracy));
        
        // Find fastest
        ExperimentResult fastest = allResults.stream()
            .filter(r -> r.avgTrainingTime > 0)
            .min((a, b) -> Long.compare(a.avgTrainingTime, b.avgTrainingTime))
            .orElse(null);
        
        if (fastest != null) {
            printAndLog("\nFastest Training: " + fastest.name + " (" + fastest.avgTrainingTime + " ms)");
        }
        
        printAndLog("==============================================");
    }
    
    interface ClassifierFactory {
        Classifier create(int[][] features, int[] labels, int numClasses);
    }
}