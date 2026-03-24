#include <iostream>
#include <vector>
#include "eigen/Eigen/Dense"
#include <limits>
#include <algorithm>

struct NNLSResult {
    Eigen::VectorXd x;
    double rnorm;
};

// Lawson-Hanson NNLS Solver
NNLSResult nnls(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                int maxIter = 1000, double tol = 1e-10)
{
    int m = A.rows();
    int n = A.cols();

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd w = A.transpose() * (b - A * x);
    std::vector<int> passiveSet;
    std::vector<int> activeSet(n);
    for (int i = 0; i < n; ++i) activeSet[i] = i;

    int iter = 0;
    while (!activeSet.empty() && w.maxCoeff() > tol && iter++ < maxIter) {
        // Find index with largest w
        int t = -1;
        double maxW = -std::numeric_limits<double>::infinity();
        for (int i : activeSet) {
            if (w(i) > maxW) {
                maxW = w(i);
                t = i;
            }
        }

        passiveSet.push_back(t);
        activeSet.erase(std::remove(activeSet.begin(), activeSet.end(), t), activeSet.end());

        Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd Ap(m, passiveSet.size());
        for (size_t i = 0; i < passiveSet.size(); ++i) {
            Ap.col(i) = A.col(passiveSet[i]);
        }

        Eigen::VectorXd zp = Ap.colPivHouseholderQr().solve(b);

        for (size_t i = 0; i < passiveSet.size(); ++i) {
            z(passiveSet[i]) = zp(i);
        }

        while ((z.array() < 0).any()) {
            double alpha = std::numeric_limits<double>::infinity();
            for (int i : passiveSet) {
                if (z(i) < 0) {
                    double alpha_i = x(i) / (x(i) - z(i));
                    if (alpha_i < alpha) alpha = alpha_i;
                }
            }

            x = x + alpha * (z - x);

            for (size_t i = 0; i < passiveSet.size();) {
                if (x(passiveSet[i]) < tol) {
                    activeSet.push_back(passiveSet[i]);
                    passiveSet.erase(passiveSet.begin() + i);
                } else {
                    ++i;
                }
            }

            Ap.resize(m, passiveSet.size());
            for (size_t i = 0; i < passiveSet.size(); ++i) {
                Ap.col(i) = A.col(passiveSet[i]);
            }

            zp = Ap.colPivHouseholderQr().solve(b);
            z = Eigen::VectorXd::Zero(n);
            for (size_t i = 0; i < passiveSet.size(); ++i) {
                z(passiveSet[i]) = zp(i);
            }
        }

        x = z;
        w = A.transpose() * (b - A * x);
    }

    // Compute residual norm ||Ax - b||_2
    Eigen::VectorXd r = A * x - b;
    double rnorm = r.norm();

    return {x, rnorm};
}