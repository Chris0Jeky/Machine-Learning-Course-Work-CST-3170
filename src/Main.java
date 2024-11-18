import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static final String COMMA_DELIMITER = ",";

    public static void main(String[] args) {
        // Path to CSV files
        String csvFileName1 = "src/datasets/dataSet1.csv";
        String csvFileName2 = "src/datasets/dataSet2.csv";

        // Read data from CSV files
        int[][] dataSet1 = initializeScanning(csvFileName1);
        int[][] dataSet2 = initializeScanning(csvFileName2);

        // Prepare features and labels
        int[][] features1 = new int[dataSet1.length][64];
        int[] labels1 = new int[dataSet1.length];

        for (int i = 0; i < dataSet1.length; i++) {
            System.arraycopy(dataSet1[i], 0, features1[i], 0, 64);
            labels1[i] = dataSet1[i][64];
        }

        int[][] features2 = new int[dataSet2.length][64];
        int[] labels2 = new int[dataSet2.length];

        for (int i = 0; i < dataSet2.length; i++) {
            System.arraycopy(dataSet2[i], 0, features2[i], 0, 64);
            labels2[i] = dataSet2[i][64];
        }

        // No normalization

        // Experiment with different k values
        int[] kValues = {1, 3, 5, 7};

        for (int k : kValues) {
            int correctPredictions = 0;
            for (int i = 0; i < features2.length; i++) {
                int predictedLabel = classify(features2[i], features1, labels1, k);
                int actualLabel = labels2[i];
                if (predictedLabel == actualLabel) {
                    correctPredictions++;
                }
            }
            double accuracy = (double) correctPredictions / features2.length * 100;
            System.out.println("Accuracy with k=" + k + ": " + accuracy + "%");
        }
    }

    // Function to initialize the 2D array by reading the CSV file
    private static int[][] initializeScanning(String csvFileName) {
        List<int[]> dataList = new ArrayList<>();

        // Create a File object
        File file = new File(csvFileName);
        System.out.println("Attempting to open file at path: " + file.getAbsolutePath());

        try (Scanner scanner = new Scanner(file)) {
            // Read each line from the file
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] values = line.split(COMMA_DELIMITER);

                int[] row = new int[values.length];
                for (int col = 0; col < values.length; col++) {
                    try {
                        row[col] = Integer.parseInt(values[col]);
                    } catch (NumberFormatException e) {
                        System.out.println("Error: Invalid number format at row " + (dataList.size() + 1) + ", column " + (col + 1));
                        row[col] = 0; // Assign a default value (0) in case of invalid format
                    }
                }

                dataList.add(row);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found: " + csvFileName);
            return new int[0][];
        }

        // Convert the list to a 2D array
        return dataList.toArray(new int[0][]);
    }

    // Euclidean distance function
    public static double euclideanDistance(int[] vector1, int[] vector2) {
        double sum = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            double diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // k-NN classification function without extra libraries
    public static int classify(int[] testImage, int[][] trainingFeatures, int[] trainingLabels, int k) {
        int n = trainingFeatures.length;
        double[] distances = new double[n];
        int[] labels = new int[n];

        // Compute distances
        for (int i = 0; i < n; i++) {
            distances[i] = euclideanDistance(testImage, trainingFeatures[i]);
            labels[i] = trainingLabels[i];
        }

        // Sort distances and labels together using a simple selection sort
        for (int i = 0; i < n - 1; i++) {
            // Find the index of the minimum distance
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (distances[j] < distances[minIndex]) {
                    minIndex = j;
                }
            }
            // Swap distances[i] and distances[minIndex]
            double tempDist = distances[i];
            distances[i] = distances[minIndex];
            distances[minIndex] = tempDist;

            // Swap labels[i] and labels[minIndex]
            int tempLabel = labels[i];
            labels[i] = labels[minIndex];
            labels[minIndex] = tempLabel;
        }

        // Count labels in the first k entries
        int[] labelCounts = new int[10]; // Assuming labels are digits 0-9
        for (int i = 0; i < k; i++) {
            int label = labels[i];
            labelCounts[label]++;
        }

        // Find the label with the highest count
        int predictedLabel = 0;
        int maxCount = labelCounts[0];
        for (int i = 1; i < 10; i++) {
            if (labelCounts[i] > maxCount) {
                maxCount = labelCounts[i];
                predictedLabel = i;
            }
        }

        // Handle ties (optional)
        // Check if multiple labels have the same max count
        int numMaxLabels = 0;
        for (int i = 0; i < 10; i++) {
            if (labelCounts[i] == maxCount) {
                numMaxLabels++;
            }
        }

        if (numMaxLabels == 1) {
            // Unique label with max count
            return predictedLabel;
        } else {
            // Tie occurred
            // Among the tied labels, choose the one with the smallest cumulative distance
            double[] cumulativeDistances = new double[10];
            for (int i = 0; i < k; i++) {
                int label = labels[i];
                if (labelCounts[label] == maxCount) {
                    cumulativeDistances[label] += distances[i];
                }
            }

            double minCumulativeDistance = Double.MAX_VALUE;
            for (int i = 0; i < 10; i++) {
                if (labelCounts[i] == maxCount && cumulativeDistances[i] < minCumulativeDistance) {
                    minCumulativeDistance = cumulativeDistances[i];
                    predictedLabel = i;
                }
            }
            return predictedLabel;
        }
    }
}
