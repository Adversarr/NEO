#include <stdexcept>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "FastSpectrum.h"

namespace py = pybind11;

namespace {

Eigen::MatrixXd copyVertices(const py::array_t<double, py::array::c_style | py::array::forcecast> &vertices)
{
	if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
		throw std::invalid_argument("V must be a 2D float array with shape (n, 3).");
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mapped(
		static_cast<Eigen::MatrixXd::Index>(vertices.shape(0)),
		static_cast<Eigen::MatrixXd::Index>(vertices.shape(1)));
	const auto view = vertices.unchecked<2>();
	for (py::ssize_t row = 0; row < vertices.shape(0); ++row) {
		for (py::ssize_t col = 0; col < vertices.shape(1); ++col) {
			mapped(static_cast<Eigen::MatrixXd::Index>(row), static_cast<Eigen::MatrixXd::Index>(col)) = view(row, col);
		}
	}
	return Eigen::MatrixXd(mapped);
}

Eigen::MatrixXi copyFaces(const py::array_t<int, py::array::c_style | py::array::forcecast> &faces)
{
	if (faces.ndim() != 2 || faces.shape(1) != 3) {
		throw std::invalid_argument("F must be a 2D integer array with shape (m, 3).");
	}

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mapped(
		static_cast<Eigen::MatrixXi::Index>(faces.shape(0)),
		static_cast<Eigen::MatrixXi::Index>(faces.shape(1)));
	const auto view = faces.unchecked<2>();
	for (py::ssize_t row = 0; row < faces.shape(0); ++row) {
		for (py::ssize_t col = 0; col < faces.shape(1); ++col) {
			mapped(static_cast<Eigen::MatrixXi::Index>(row), static_cast<Eigen::MatrixXi::Index>(col)) = view(row, col);
		}
	}
	return Eigen::MatrixXi(mapped);
}

}  // namespace

PYBIND11_MODULE(pyfastspectrum, module)
{
	module.doc() = "Python bindings for FastSpectrum";

	py::enum_<SamplingType>(module, "SamplingType")
		.value("Sample_Poisson_Disk", Sample_Poisson_Disk)
		.value("Sample_Farthest_Point", Sample_Farthest_Point)
		.value("Sample_Random", Sample_Random)
		.export_values();

	py::enum_<SolverBackend>(module, "SolverBackend")
		.value("Auto", SolverBackend::Auto)
		.value("CPU", SolverBackend::CPU)
		.value("CUDA", SolverBackend::CUDA);

	module.def("cuda_compiled", &isCudaCompiled, "Return True when CUDA support is compiled into the extension.");
	module.def("cuda_available", &isCudaRuntimeAvailable, "Return True when a CUDA device is available at runtime.");

	py::class_<FastSpectrum>(module, "FastSpectrum")
		.def(py::init<>())
		.def(
			"assemble_laplacian",
			[](FastSpectrum &self,
			   const py::array_t<double, py::array::c_style | py::array::forcecast> &vertices,
			   const py::array_t<int, py::array::c_style | py::array::forcecast> &faces) {
				Eigen::MatrixXd V = copyVertices(vertices);
				Eigen::MatrixXi F = copyFaces(faces);
				Eigen::SparseMatrix<double> stiffness;
				Eigen::SparseMatrix<double> mass;

				self.setMesh(V, F);
				{
					py::gil_scoped_release release;
					self.constructLaplacianMatrix();
					self.getLaplacian(stiffness, mass);
				}

				return py::make_tuple(std::move(stiffness), std::move(mass));
			},
			py::arg("V"),
			py::arg("F"),
			"Assemble the stiffness and mass matrices for the provided mesh.")
		.def(
			"compute_eigenpairs",
			[](FastSpectrum &self,
			   const py::array_t<double, py::array::c_style | py::array::forcecast> &vertices,
			   const py::array_t<int, py::array::c_style | py::array::forcecast> &faces,
			   const int numSamples,
			   const SamplingType sampleType,
			   const SolverBackend backend) {
				if (numSamples < 1) {
					throw std::invalid_argument("num_samples must be at least 1.");
				}

				Eigen::MatrixXd V = copyVertices(vertices);
				Eigen::MatrixXi F = copyFaces(faces);
				if (numSamples > V.rows()) {
					throw std::invalid_argument("num_samples cannot exceed the number of vertices.");
				}

				Eigen::SparseMatrix<double> basis;
				Eigen::MatrixXd reducedEigVects;
				Eigen::VectorXd reducedEigVals;

				self.setSolverBackend(backend);
				{
					py::gil_scoped_release release;
					self.computeEigenPairs(V, F, numSamples, sampleType, basis, reducedEigVects, reducedEigVals);
				}
				return py::make_tuple(std::move(basis), std::move(reducedEigVects), std::move(reducedEigVals));
			},
			py::arg("V"),
			py::arg("F"),
			py::arg("num_samples"),
			py::arg("sample_type") = Sample_Poisson_Disk,
			py::arg("backend") = SolverBackend::Auto,
			"Compute the reduced basis and eigenpairs from vertex and face arrays.");
}
