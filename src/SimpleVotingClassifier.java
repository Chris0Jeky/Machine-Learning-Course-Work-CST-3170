public class SimpleVotingClassifier implements Classifier {
    private Classifier c1;
    private Classifier c2;

    public SimpleVotingClassifier(Classifier c1, Classifier c2) {
        this.c1 = c1;
        this.c2 = c2;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        c1.train(features, labels);
        c2.train(features, labels);
    }

    @Override
    public int predict(int[] sample) {
        int p1 = c1.predict(sample);
        int p2 = c2.predict(sample);
        // Simple majority voting for two classifiers:
        // If they disagree, just pick one (or add a third classifier for tie-break)
        return (p1 == p2) ? p1 : p1; // or p2, or use distance-based tie-breaking
    }
}
