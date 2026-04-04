#include <string>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "lpmatrix.h"

namespace py = pybind11;

namespace {

py::tuple
compute_triplets(py::array_t<double, py::array::c_style | py::array::forcecast> points,
                 size_t tdim,
                 unsigned int nn,
                 double hs,
                 double rho,
                 const std::string& htype)
{
  py::buffer_info info = points.request();
  if (info.ndim != 2)
    {
      throw std::invalid_argument("points must be a 2D array with shape (n_points, ambient_dim)");
    }

  const size_t np = static_cast<size_t>(info.shape[0]);
  const size_t dim = static_cast<size_t>(info.shape[1]);

  pcdlaplace::Options options;
  options.nn = nn;
  options.hs = hs;
  options.rho = rho;
  options.htype = pcdlaplace::parse_htype(htype);

  const auto result = pcdlaplace::compute_pcdlaplace_matrix_sparse(
      static_cast<const double *>(info.ptr), np, dim, tdim, options);

  py::array_t<long long> rows(result.rows.size());
  py::array_t<long long> cols(result.cols.size());
  py::array_t<double> values(result.values.size());

  auto rows_view = rows.mutable_unchecked<1>();
  auto cols_view = cols.mutable_unchecked<1>();
  auto values_view = values.mutable_unchecked<1>();
  for (size_t i = 0; i < result.values.size(); ++i)
    {
      rows_view(i) = static_cast<long long>(result.rows[i]);
      cols_view(i) = static_cast<long long>(result.cols[i]);
      values_view(i) = result.values[i];
    }

  return py::make_tuple(std::move(rows),
                        std::move(cols),
                        std::move(values),
                        py::make_tuple(result.nrows, result.ncols));
}

}  // namespace

PYBIND11_MODULE(_core, m)
{
  m.doc() = "pybind11 bindings for the point-cloud Laplace matrix core";
  m.def("compute_triplets",
        &compute_triplets,
        py::arg("points"),
        py::arg("tdim"),
        py::arg("nn") = 10,
        py::arg("hs") = 2.0,
        py::arg("rho") = 3.0,
        py::arg("htype") = "ddr");
}
