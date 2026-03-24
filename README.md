# nnls_eigen
Simple implementation of the Lawson-Hanson non-negative least squares linear system solver using the C++ Eigen library. Functions the same as the Scipy nnls function and has been tested to give the same results.

### Example usage 

```
int main()
{
    Eigen::MatrixXd A(3, 3);
    A << 2, -1, 0,
        -1, 2, -1,
         0, -1, 2;

    Eigen::VectorXd b(3);
    b << 1, 0, 1;

    NNLSResult result = nnls(A, b);

    std::cout << "Solution x:\n" << result.x << std::endl;
    std::cout << "Residual norm ||Ax - b||_2:\n" << result.rnorm << std::endl;

    return 0;
}
```
