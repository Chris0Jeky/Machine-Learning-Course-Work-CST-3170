public class PolynomialKernel implements Kernel {
    private int degree;
    private double coef0;
    private double gamma;

    public PolynomialKernel(int degree, double coef0, double gamma) {
        this.degree = degree;
        this.coef0 = coef0;
        this.gamma = gamma;
    }

    @Override
    public double compute(int[] x1, int[] x2) {
        double dotProduct = 0.0;
        for (int i = 0; i < x1.length; i++) {
            dotProduct += x1[i] * x2[i];
        }
        return Math.pow(gamma * dotProduct + coef0, degree);
    }
}
