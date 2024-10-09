import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Main {
    // Constants for dataset dimensions
    private static final int NUM_ROWS = 2810;
    private static final int NUM_COLUMNS = 65;
    private static final String COMMA_DELIMITER = ",";

    public static void main(String[] args) {
        // Initialize a 2D array with 2810 rows and 65 columns
        int[][] dataSet1 = new int[NUM_ROWS][NUM_COLUMNS];

        // Path to your CSV file
        String csvFileName1 = "C:\\Users\\Chris\\Desktop\\Minecfraft\\UTIL\\ThirdYear\\AIGradle\\Lab001v2\\src\\main\\java\\dataSet1.csv";

        // Fill the 2D array with data from the CSV file
        dataSet1 = initializeScanning(csvFileName1, dataSet1);

        // Optionally, print the dataset to verify correctness
        print2DArr(dataSet1);
    }

    // Function to initialize the 2D array by reading the CSV file
    private static int[][] initializeScanning(String csvFileName, int[][] dataSet) {

        int row = 0;
        try (Scanner scanner = new Scanner(new File(csvFileName))) {
            // Open the CSV file

            // Read each line from the file
            while (scanner.hasNextLine() && row < dataSet.length) {
                String line = scanner.nextLine();

                // Split the line by commas to get the values
                String[] values = line.split(COMMA_DELIMITER);

                // Check that the line has exactly 65 values
                if (values.length != NUM_COLUMNS) {
                    System.out.println("Error: Row " + (row + 1) + " does not have exactly 65 values.");
                    continue;
                }

                // Parse each value and store it in the corresponding position in the 2D array
                for (int col = 0; col < values.length; col++) {
                    try {
                        dataSet[row][col] = Integer.parseInt(values[col]);
                    } catch (NumberFormatException e) {
                        System.out.println("Error: Invalid number format at row " + (row + 1) + ", column " + (col + 1));
                        dataSet[row][col] = 0; // Assign a default value (0) in case of invalid format
                    }
                }

                row++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found: " + csvFileName);
        }

        return dataSet;
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
}