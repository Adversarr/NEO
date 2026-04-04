#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/SelectionRule.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace
{
using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using SolverOp = Spectra::SymShiftInvert<
    double,
    Eigen::Sparse,
    Eigen::Sparse,
    Eigen::Lower,
    Eigen::Lower,
    Eigen::RowMajor,
    Eigen::RowMajor>;
using MassOp = Spectra::SparseSymMatProd<double, Eigen::Lower, Eigen::RowMajor>;
using Solver = Spectra::SymGEigsShiftSolver<SolverOp, MassOp, Spectra::GEigsMode::ShiftInvert>;

std::string comp_info_to_string(Spectra::CompInfo info);
int configure_threads_from_env();
void set_num_threads(int num_threads);
int get_num_threads();

double max_abs_coeff(const SparseMatrixd& mat)
{
    double out = 0.0;
    for(int outer = 0; outer < mat.outerSize(); ++outer)
    {
        for(SparseMatrixd::InnerIterator it(mat, outer); it; ++it)
            out = std::max(out, std::abs(it.value()));
    }
    return out;
}

double fallback_shift(double sigma, const SparseMatrixd& A, const SparseMatrixd& B)
{
    const double scale = std::max({1.0, std::abs(sigma), max_abs_coeff(A), max_abs_coeff(B)});
    const double delta = std::sqrt(std::numeric_limits<double>::epsilon()) * scale;
    return sigma >= 0.0 ? sigma - delta : sigma + delta;
}

py::dict run_solver(
    const SparseMatrixd& A,
    const SparseMatrixd& B,
    Eigen::Index k,
    Eigen::Index ncv,
    double sigma,
    Eigen::Index maxit,
    double tol)
{
    SolverOp op(A, B);
    MassOp bop(B);
    Solver solver(op, bop, k, ncv, sigma);

    solver.init();
    const Eigen::Index nconv = solver.compute(Spectra::SortRule::LargestMagn, maxit, tol);
    const auto info = solver.info();

    if(info == Spectra::CompInfo::NumericalIssue)
        throw std::runtime_error("Spectra failed with NumericalIssue");

    py::dict out;
    out["nconv"] = nconv;
    out["status"] = comp_info_to_string(info);
    out["eigenvalues"] = solver.eigenvalues();
    out["eigenvectors"] = solver.eigenvectors();
    out["used_sigma"] = sigma;
    return out;
}

void set_num_threads(int num_threads)
{
    if(num_threads <= 0)
        throw std::invalid_argument("num_threads must be positive");

    Eigen::setNbThreads(num_threads);
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
}

int get_num_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return Eigen::nbThreads();
#endif
}

int configure_threads_from_env()
{
    const char* env = std::getenv("OMP_NUM_THREADS");
    if(env == nullptr || *env == '\0')
        return get_num_threads();

    int parsed = 0;
    const char* end = env + std::char_traits<char>::length(env);
    const auto result = std::from_chars(env, end, parsed);
    if(result.ec != std::errc{} || result.ptr != end || parsed <= 0)
        throw std::invalid_argument("OMP_NUM_THREADS must be a positive integer");

    set_num_threads(parsed);
    return parsed;
}

std::string comp_info_to_string(Spectra::CompInfo info)
{
    switch(info)
    {
    case Spectra::CompInfo::Successful:
        return "Successful";
    case Spectra::CompInfo::NotComputed:
        return "NotComputed";
    case Spectra::CompInfo::NotConverging:
        return "NotConverging";
    case Spectra::CompInfo::NumericalIssue:
        return "NumericalIssue";
    }

    return "Unknown";
}

py::dict solve_sym_shift_invert_generalized(
    const SparseMatrixd& A, const SparseMatrixd& B, Eigen::Index k,
    Eigen::Index ncv = -1, double sigma = 0.0, Eigen::Index maxit = 1000,
    double tol = 1e-10)
{
    if(A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
        throw std::invalid_argument("A and B must be square sparse matrices with the same shape");

    const Eigen::Index n = A.rows();
    if(n < 2)
        throw std::invalid_argument("Matrix dimension must be at least 2");
    if(k < 1 || k >= n)
        throw std::invalid_argument("k must satisfy 1 <= k < n");

    if(ncv < 0)
        ncv = std::min<Eigen::Index>(n, std::max<Eigen::Index>(2 * k + 1, 20));
    if(ncv <= k || ncv > n)
        throw std::invalid_argument("ncv must satisfy k < ncv <= n");

    try
    {
        return run_solver(A, B, k, ncv, sigma, maxit, tol);
    }
    catch(const std::exception&)
    {
        const double sigma_retry = fallback_shift(sigma, A, B);
        if(sigma_retry == sigma)
            throw;

        try
        {
            return run_solver(A, B, k, ncv, sigma_retry, maxit, tol);
        }
        catch(const std::exception&)
        {
            throw std::runtime_error(
                "Spectra shift-invert solve failed; B must be positive definite and A - sigma * B "
                "must be factorizable. The retry with a perturbed shift also failed.");
        }
    }
}
}  // namespace

PYBIND11_MODULE(_pyspec, m)
{
    m.doc() = "Minimal Spectra-based bindings for symmetric generalized shift-invert eigensolves";
    Eigen::initParallel();
    const int configured_threads = configure_threads_from_env();

    m.def("set_num_threads", &set_num_threads, py::arg("num_threads"));
    m.def("get_num_threads", &get_num_threads);
    m.attr("openmp_enabled") =
#ifdef _OPENMP
        true;
#else
        false;
#endif
    m.attr("configured_num_threads") = configured_threads;

    m.def(
        "solve_sym_shift_invert_generalized",
        &solve_sym_shift_invert_generalized,
        py::arg("A"),
        py::arg("B"),
        py::arg("k"),
        py::arg("ncv") = -1,
        py::arg("sigma") = 0.0,
        py::arg("maxit") = 1000,
        py::arg("tol") = 1e-10,
        R"pbdoc(
Solve A x = lambda B x for the eigenpairs nearest sigma using shift-invert.

`A` and `B` must be SciPy sparse matrices accepted by pybind11's Eigen sparse casters.
`B` must be symmetric positive definite. The solver returns the eigenpairs nearest
`sigma` by running Spectra with `SortRule::LargestMagn` in shift-invert mode.
Internally the sparse matrices are stored in row-major CSR format so Eigen can
use OpenMP for sparse matrix-vector and matrix-matrix products.
)pbdoc");
}
