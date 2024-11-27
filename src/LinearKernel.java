public class LinearKernel implements Kernel {
    @Override
    public double compute(int[] x1, int[] x2) {
        double result = 0.0;
        for (int i = 0; i < x1.length; i++) {
            result += x1[i] * x2[i];
        }
        return result;
    }
}
