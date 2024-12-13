public interface Kernel {
    // Represents a kernel function used by kernel-based SVMs.
    // Given two feature vectors x1 and x2, compute returns their kernelized similarity.
    double compute(int[] x1, int[] x2);
}
