import java.util.List;
import java.util.ArrayList;

public class Utils {
    // Extract features (first 64 columns)
    public static int[][] extractFeatures(int[][] data) {
        int[][] features = new int[data.length][64];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, features[i], 0, 64);
        }
        return features;
    }

    // Extract labels (last column)
    public static int[] extractLabels(int[][] data) {
        int[] labels = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            labels[i] = data[i][64];
        }
        return labels;
    }

    public static int getMaxLabel(int[] labels) {
        int max = Integer.MIN_VALUE;
        for (int label : labels) {
            if (label > max) {
                max = label;
            }
        }
        return max;
    }
}
