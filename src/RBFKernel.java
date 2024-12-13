//public class RBFKernel implements Kernel {
//    private double gamma;
//
//    public RBFKernel(double gamma) {
//        this.gamma = gamma;
//    }
//
//    @Override
//    public double compute(int[] x1, int[] x2) {
//        double sum = 0.0;
//        for (int i = 0; i < x1.length; i++) {
//            double diff = x1[i] - x2[i];
//            sum += diff * diff;
//        }
//        return Math.exp(-gamma * sum);
//    }
//}