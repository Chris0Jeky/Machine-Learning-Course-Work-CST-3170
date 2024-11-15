import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static final String COMMA_DELIMITER = ",";

    public static void main(String[] args) {
        // Path to CSV file
        String csvFileName1 = "src/datasets/dataSet1.csv";
        String csvFileName2 = "src/datasets/dataSet2.csv";

        // Determine dimensions from the file and initialize the array
        int[][] dataSet1 = initializeScanning(csvFileName1);
        int[][] dataSet2 = initializeScanning(csvFileName2);

        // Optionally, print the dataset to verify correctness
        //print2DArr(dataSet1);
        //print2DArr(dataSet2);
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

    // Function to print the 2D array (optional for debugging purposes)
    private static void print2DArr(int[][] arr) {
        for (int[] ints : arr) {
            for (int anInt : ints) {
                System.out.print(anInt + " ");
            }
            System.out.println(); // Move to the next line after printing a row
        }
    }

    public static void printArray(int[] arr) {
        for (int element : arr) {
            System.out.println(element);
        }
    }
}
