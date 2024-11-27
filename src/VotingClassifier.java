import java.util.HashMap;
import java.util.Map;

public class VotingClassifier implements Classifier {
    private Classifier[] classifiers;

    public VotingClassifier(Classifier[] classifiers) {
        this.classifiers = classifiers;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        for (Classifier classifier : classifiers) {
            classifier.train(features, labels);
        }
    }

    @Override
    public int predict(int[] sample) {
        Map<Integer, Integer> votes = new HashMap<>();
        for (Classifier classifier : classifiers) {
            int prediction = classifier.predict(sample);
            votes.put(prediction, votes.getOrDefault(prediction, 0) + 1);
        }

        // Find the class with the most votes
        int maxVotes = 0;
        int predictedClass = -1;
        for (Map.Entry<Integer, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                predictedClass = entry.getKey();
            }
        }
        return predictedClass;
    }
}
