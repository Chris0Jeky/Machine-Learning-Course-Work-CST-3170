import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataLoader {
    private static final String COMMA_DELIMITER = ",";

    // Loads a CSV file of integers into a 2D array.
    // Each row of the CSV is converted to an int[].
    // Any parsing errors yield a default value of 0 for that cell.
    public static int[][] loadData(String csvFileName) {
        List<int[]> dataList = new ArrayList<>();
        File file = new File(csvFileName);
        System.out.println("Attempting to open file at path: " + file.getAbsolutePath());

        try (Scanner scanner = new Scanner(file)) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] values = line.split(COMMA_DELIMITER);
                int[] row = new int[values.length];
                for (int col = 0; col < values.length; col++) {
                    try {
                        row[col] = Integer.parseInt(values[col]);
                    } catch (NumberFormatException e) {
                        System.out.println("Error parsing value at row " + (dataList.size() + 1) + ", column " + (col + 1));
                        row[col] = 0;
                    }
                }
                dataList.add(row);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found: " + csvFileName);
            return new int[0][];
        }

        return dataList.toArray(new int[0][]);
    }
}