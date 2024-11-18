import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List; // Make sure to import List
import java.util.Scanner;

public class Main {
    private static final String COMMA_DELIMITER = ",";

    public static void main(String[] args) {
        // Path to CSV file
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

        // Classify test images and evaluate
        int correctPredictions = 0;
        for (int i = 0; i < features2.length; i++) {
            int predictedLabel = classify(features2[i], features1, labels1);
            int actualLabel = labels2[i];
            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
            System.out.println("Test Image " + (i + 1) + ": Predicted Label = " + predictedLabel + ", Actual Label = " + actualLabel);
        }

        double accuracy = (double) correctPredictions / features2.length * 100;
        System.out.println("Accuracy: " + accuracy + "%");
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

    // Nearest Neighbor classification function
    public static int classify(int[] testImage, int[][] trainingFeatures, int[] trainingLabels) {
        double minDistance = Double.MAX_VALUE;
        int predictedLabel = -1;

        for (int i = 0; i < trainingFeatures.length; i++) {
            double distance = euclideanDistance(testImage, trainingFeatures[i]);
            if (distance < minDistance) {
                minDistance = distance;
                predictedLabel = trainingLabels[i];
            }
        }
        return predictedLabel;
    }
}
