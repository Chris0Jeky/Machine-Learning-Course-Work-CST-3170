public class SimpleVotingClassifier implements Classifier {
    private Classifier[] classifiers;

    public SimpleVotingClassifier(Classifier[] classifiers) {
        this.classifiers = classifiers;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        // Train all classifiers
        for (Classifier c : classifiers) {
            c.train(features, labels);
        }
    }

    @Override
    public int predict(int[] sample) {
        // Count votes for each predicted class
        // First, we need to know how many classes we have, or we can just map predictions to counts
        // If we know numClasses from outside, we could create an array of counts, but it's easier just to store predictions and pick majority.

        // If we don't know numClasses here, let's just use a hashmap-like approach with arrays since we know labels range from 0..numClasses-1.
        // Alternatively, we can assume a reasonable max number of classes or find max label after predictions.
        // For simplicity, let's just store predictions and find the majority by counting.

        // Gather predictions
        int[] predictions = new int[classifiers.length];
        for (int i = 0; i < classifiers.length; i++) {
            predictions[i] = classifiers[i].predict(sample);
        }

        // Find majority vote
        // Since we don't know the range of classes directly here, we can map out each prediction count.
        // A simple approach: find the class that occurs most often in predictions.

        int majorityVote = majorityElement(predictions);
        return majorityVote;
    }

    // Helper method to find majority element. If tie occurs, returns one of them.
    private int majorityElement(int[] arr) {
        // A simple way: count frequencies in a map
        // Since classes are likely small integers, we can do a simple count:
        // Let's assume class labels won't exceed a certain limit. For large range, you'd use a HashMap.
        // For safety, use a HashMap:

        java.util.Map<Integer, Integer> counts = new java.util.HashMap<>();
        for (int val : arr) {
            counts.put(val, counts.getOrDefault(val, 0) + 1);
        }

        int maxCount = -1;
        int majority = arr[0];
        for (java.util.Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majority = entry.getKey();
            }
        }
        return majority;
    }
}
