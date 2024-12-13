public class KernelSVMClassifier implements Classifier {
    private int[][] trainingFeatures;
    private int[] trainingLabels;
    private double[] alphas;
    private double b;
    private double C;
    private double tol;
    private int maxPasses;
    private Kernel kernel;
    private int n; // number of samples

    public KernelSVMClassifier(double C, double tol, int maxPasses, Kernel kernel, int numClasses) {
        this.C = C;
        this.tol = tol;
        this.maxPasses = maxPasses;
        this.kernel = kernel;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        this.trainingFeatures = features;
        this.trainingLabels = labels;
        this.n = features.length;
        alphas = new double[n];
        b = 0.0;

        // SMO variables
        int passes = 0;

        // Convert labels from {0,...} to {+1, -1} if needed
        // Assume labels are already +1 or -1. If not, you must convert them before training.

        // Check if labels contain {1,-1}, if not, try to convert:
        boolean needConversion = false;
        for (int l : labels) {
            if (l != 1 && l != -1) {
                needConversion = true;
                break;
            }
        }
        int[] convertedLabels = labels;
        if (needConversion) {
            // If your code expects binary problem {1,-1}, you must have done that before calling train.
            // For multiclass, MulticlassKernelSVMClassifier does this conversion.
            // Here, we assume the caller already converted.
            convertedLabels = labels; // If we reached here, just assume labels are {1,-1}.
        }

        // Simple SMO Implementation
        while (passes < maxPasses) {
            int numChangedAlphas = 0;
            for (int i = 0; i < n; i++) {
                double Ei = computeError(i, convertedLabels);

                if ((convertedLabels[i] * Ei < -tol && alphas[i] < C) ||
                        (convertedLabels[i] * Ei > tol && alphas[i] > 0)) {

                    // Select j != i randomly
                    int j = i;
                    java.util.Random rand = new java.util.Random();
                    while (j == i) {
                        j = rand.nextInt(n);
                    }

                    double Ej = computeError(j, convertedLabels);

                    double alphaIold = alphas[i];
                    double alphaJold = alphas[j];

                    // Compute L and H
                    double L, H;
                    if (convertedLabels[i] != convertedLabels[j]) {
                        L = Math.max(0, alphas[j] - alphas[i]);
                        H = Math.min(C, C + alphas[j] - alphas[i]);
                    } else {
                        L = Math.max(0, alphas[i] + alphas[j] - C);
                        H = Math.min(C, alphas[i] + alphas[j]);
                    }

                    if (L == H) continue;

                    double eta = 2 * kernel.compute(trainingFeatures[i], trainingFeatures[j])
                            - kernel.compute(trainingFeatures[i], trainingFeatures[i])
                            - kernel.compute(trainingFeatures[j], trainingFeatures[j]);

                    if (eta >= 0) continue;

                    // Update alphas[j]
                    alphas[j] = alphas[j] - (convertedLabels[j] * (Ei - Ej)) / eta;

                    // Clip alphas[j]
                    if (alphas[j] > H) alphas[j] = H;
                    else if (alphas[j] < L) alphas[j] = L;

                    if (Math.abs(alphas[j] - alphaJold) < 1e-5) continue;

                    // Update alphas[i]
                    alphas[i] = alphas[i] + convertedLabels[i] * convertedLabels[j] * (alphaJold - alphas[j]);

                    // Compute b1 and b2
                    double b1 = b - Ei
                            - convertedLabels[i] * (alphas[i] - alphaIold) * kernel.compute(trainingFeatures[i], trainingFeatures[i])
                            - convertedLabels[j] * (alphas[j] - alphaJold) * kernel.compute(trainingFeatures[i], trainingFeatures[j]);

                    double b2 = b - Ej
                            - convertedLabels[i] * (alphas[i] - alphaIold) * kernel.compute(trainingFeatures[i], trainingFeatures[j])
                            - convertedLabels[j] * (alphas[j] - alphaJold) * kernel.compute(trainingFeatures[j], trainingFeatures[j]);

                    if (0 < alphas[i] && alphas[i] < C) {
                        b = b1;
                    } else if (0 < alphas[j] && alphas[j] < C) {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2;
                    }

                    numChangedAlphas++;
                }
            }
            if (numChangedAlphas == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }
    }

    @Override
    public int predict(int[] sample) {
        double val = decisionFunction(sample);
        return val >= 0 ? 1 : -1;
    }

    public double decisionFunction(int[] sample) {
        double sumVal = 0.0;
        for (int i = 0; i < n; i++) {
            if (alphas[i] > 0) {
                sumVal += alphas[i] * trainingLabels[i] * kernel.compute(trainingFeatures[i], sample);
            }
        }
        sumVal += b;
        return sumVal;
    }

    private double computeError(int i, int[] convertedLabels) {
        double fx_i = 0.0;
        for (int j = 0; j < n; j++) {
            if (alphas[j] > 0) {
                fx_i += alphas[j] * convertedLabels[j] * kernel.compute(trainingFeatures[j], trainingFeatures[i]);
            }
        }
        fx_i += b;
        return fx_i - convertedLabels[i];
    }
}
