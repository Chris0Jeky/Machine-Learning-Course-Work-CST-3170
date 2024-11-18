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

    // Filter features for binary classes
    public static int[][] filterBinaryClasses(int[][] features, int[] labels, int class1, int class2) {
        List<int[]> filteredFeatures = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == class1 || labels[i] == class2) {
                filteredFeatures.add(features[i]);
            }
        }
        return filteredFeatures.toArray(new int[0][]);
    }

    // Filter labels for binary classes
    public static int[] filterBinaryLabels(int[] labels, int class1, int class2) {
        List<Integer> filteredLabels = new ArrayList<>();
        for (int label : labels) {
            if (label == class1 || label == class2) {
                filteredLabels.add(label);
            }
        }
        int[] result = new int[filteredLabels.size()];
        for (int i = 0; i < filteredLabels.size(); i++) {
            result[i] = filteredLabels.get(i);
        }
        return result;
    }
}
