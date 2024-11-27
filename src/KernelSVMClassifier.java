public class KernelSVMClassifier implements Classifier {
    private int[][] trainingFeatures;
    private int[] trainingLabels;
    private double[] alphas;
    private double b;
    private double C;
    private double tol;
    private int maxPasses;
    private Kernel kernel;
    private int numClasses;

    public KernelSVMClassifier(double C, double tol, int maxPasses, Kernel kernel, int numClasses) {
        this.C = C;
        this.tol = tol;
        this.maxPasses = maxPasses;
        this.kernel = kernel;
        this.numClasses = numClasses;
    }

    @Override
    public void train(int[][] features, int[] labels) {
        this.trainingFeatures = features;
        this.trainingLabels = labels;
        int n = features.length;
        alphas = new double[n];
        b = 0.0;

        int passes = 0;
        while (passes < maxPasses) {
            int numChangedAlphas = 0;
            for (int i = 0; i < n; i++) {
                double Ei = computeError(i);
                if ((trainingLabels[i] * Ei < -tol && alphas[i] < C) || (trainingLabels[i] * Ei > tol && alphas[i] > 0)) {
                    int j = selectJ(i, n);
                    double Ej = computeError(j);

                    double alphaIold = alphas[i];
                    double alphaJold = alphas[j];

                    // Compute L and H
                    double L, H;
                    if (trainingLabels[i] != trainingLabels[j]) {
                        L = Math.max(0, alphas[j] - alphas[i]);
                        H = Math.min(C, C + alphas[j] - alphas[i]);
                    } else {
                        L = Math.max(0, alphas[i] + alphas[j] - C);
                        H = Math.min(C, alphas[i] + alphas[j]);
                    }

                    if (L == H) continue;

                    // Compute eta
                    double eta = 2 * kernel.compute(trainingFeatures[i], trainingFeatures[j])
                            - kernel.compute(trainingFeatures[i], trainingFeatures[i])
                            - kernel.compute(trainingFeatures[j], trainingFeatures[j]);

                    if (eta >= 0) continue;

                    // Update alphas[j]
                    alphas[j] = alphas[j] - (trainingLabels[j] * (Ei - Ej)) / eta;

                    // Clip alphas[j]
                    if (alphas[j] > H) alphas[j] = H;
                    else if (alphas[j] < L) alphas[j] = L;

                    if (Math.abs(alphas[j] - alphaJold) < 1e-5) continue;

                    // Update alphas[i]
                    alphas[i] = alphas[i] + trainingLabels[i] * trainingLabels[j] * (alphaJold - alphas[j]);

                    // Compute b1 and b2
                    double b1 = b - Ei
                            - trainingLabels[i] * (alphas[i] - alphaIold) * kernel.compute(trainingFeatures[i], trainingFeatures[i])
                            - trainingLabels[j] * (alphas[j] - alphaJold) * kernel.compute(trainingFeatures[i], trainingFeatures[j]);

                    double b2 = b - Ej
                            - trainingLabels[i] * (alphas[i] - alphaIold) * kernel.compute(trainingFeatures[i], trainingFeatures[j])
                            - trainingLabels[j] * (alphas[j] - alphaJold) * kernel.compute(trainingFeatures[j], trainingFeatures[j]);

                    // Update b
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
        double sum = 0.0;
        for (int i = 0; i < trainingFeatures.length; i++) {
            if (alphas[i] > 0) {
                sum += alphas[i] * trainingLabels[i] * kernel.compute(trainingFeatures[i], sample);
            }
        }
        sum += b;
        return sum >= 0 ? 1 : -1;
    }

    private double computeError(int i) {
        double fx_i = 0.0;
        for (int j = 0; j < trainingFeatures.length; j++) {
            if (alphas[j] > 0) {
                fx_i += alphas[j] * trainingLabels[j] * kernel.compute(trainingFeatures[j], trainingFeatures[i]);
            }
        }
        fx_i += b;
        return fx_i - trainingLabels[i];
    }

    private int selectJ(int i, int n) {
        int j = i;
        while (j == i) {
            j = (int) (Math.random() * n);
        }
        return j;
    }
}
