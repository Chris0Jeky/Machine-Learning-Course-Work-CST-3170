import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataLoader {
    private static final String COMMA_DELIMITER = ",";

    public static int[][] loadData(String csvFileName) {
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
}
