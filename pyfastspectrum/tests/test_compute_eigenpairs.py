from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import pyfastspectrum


MODEL_PATH = Path(__file__).resolve().parents[1] / "suzanne.obj"
NUM_SAMPLES = 16


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.split()[1:]
                polygon = []
                for part in parts:
                    vertex_index = int(part.split("/")[0]) - 1
                    polygon.append(vertex_index)
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def assert_result_shapes(basis, reduced_eigvecs, reduced_eigvals, num_vertices: int) -> None:
    assert sp.isspmatrix(basis)
    assert basis.shape == (num_vertices, NUM_SAMPLES)
    assert basis.nnz > 0
    assert reduced_eigvecs.shape == (NUM_SAMPLES, NUM_SAMPLES)
    assert reduced_eigvals.shape == (NUM_SAMPLES,)
    assert np.isfinite(basis.data).all()
    assert np.isfinite(reduced_eigvecs).all()
    assert np.isfinite(reduced_eigvals).all()
    assert np.all(np.diff(reduced_eigvals) >= -1e-8)


@pytest.fixture(scope="module")
def suzanne() -> tuple[np.ndarray, np.ndarray]:
    return load_obj(MODEL_PATH)


def test_compute_eigenpairs_cpu(suzanne: tuple[np.ndarray, np.ndarray]) -> None:
    vertices, faces = suzanne
    fastspectrum = pyfastspectrum.FastSpectrum()

    basis, reduced_eigvecs, reduced_eigvals = fastspectrum.compute_eigenpairs(
        vertices,
        faces,
        NUM_SAMPLES,
        sample_type=pyfastspectrum.SamplingType.Sample_Farthest_Point,
        backend=pyfastspectrum.SolverBackend.CPU,
    )

    assert_result_shapes(basis, reduced_eigvecs, reduced_eigvals, vertices.shape[0])


@pytest.mark.skipif(
    not (pyfastspectrum.cuda_compiled() and pyfastspectrum.cuda_available()),
    reason="CUDA backend is not compiled or not available.",
)
def test_compute_eigenpairs_cuda(suzanne: tuple[np.ndarray, np.ndarray]) -> None:
    vertices, faces = suzanne
    fastspectrum = pyfastspectrum.FastSpectrum()

    basis, reduced_eigvecs, reduced_eigvals = fastspectrum.compute_eigenpairs(
        vertices,
        faces,
        NUM_SAMPLES,
        sample_type=pyfastspectrum.SamplingType.Sample_Farthest_Point,
        backend=pyfastspectrum.SolverBackend.CUDA,
    )

    assert_result_shapes(basis, reduced_eigvecs, reduced_eigvals, vertices.shape[0])


def test_compute_eigenpairs_rejects_invalid_input(suzanne: tuple[np.ndarray, np.ndarray]) -> None:
    vertices, faces = suzanne
    fastspectrum = pyfastspectrum.FastSpectrum()

    with pytest.raises(ValueError, match="shape"):
        fastspectrum.compute_eigenpairs(vertices[:, :2], faces, NUM_SAMPLES)

    with pytest.raises(ValueError, match="num_samples"):
        fastspectrum.compute_eigenpairs(vertices, faces, vertices.shape[0] + 1)
