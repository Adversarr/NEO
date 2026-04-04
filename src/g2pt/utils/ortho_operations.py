import torch
import numpy as np

def gram_schmidt_orthogonalization(vectors: torch.Tensor, mass: torch.Tensor | None) -> torch.Tensor:
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors.

    Args:
        vectors (torch.Tensor): A tensor of shape (B, N, C) where B is the batch size,
                                C is the number of vectors, and N is the dimension of each vector.
        mass (torch.Tensor | None): Optional tensor of shape (B, N, 1) representing the mass of each point.

    Returns:
        torch.Tensor: A tensor of the same shape as input, with orthogonalized vectors.
    """
    with torch.autocast(vectors.device.type, enabled=False):
        # Ensure vectors are in float32 format
        vectors = vectors.float()
        m = mass.float().squeeze(-1) if mass is not None else None  # (B, N)

        batch_size, vector_dim, num_vectors = vectors.shape
        orthogonal_vectors = [] # [(B, N), ...]

        for i in range(num_vectors):
            v = vectors[:, :, i].clone()  # Shape: (B, N)
            if i > 0:
                # Compute projections onto all previous orthogonal vectors at once
                prev_ortho = torch.stack(orthogonal_vectors, dim=1).detach()  # Shape: (B, i, N)
                if m is not None:
                    # If mass is provided, apply it to the dots
                    dots = torch.sum((v.unsqueeze(1) * prev_ortho) * m.unsqueeze(1), dim=2)  # Shape: (B, i)
                else:
                    dots = torch.sum(v.unsqueeze(1) * prev_ortho, dim=2)  # Shape: (B, i)
                projections = torch.sum(dots.unsqueeze(2) * prev_ortho, dim=1)  # Shape: (B, N)
                v = v - projections

            # Normalize
            if m is not None:
                # If mass is provided, normalize with respect to the mass
                norms = torch.sqrt(torch.sum(v * v * m, dim=1, keepdim=True)) # Shape: (B, 1)
            else:
                norms = torch.norm(v, dim=1, keepdim=True)  # Shape: (B, 1)
            orthogonal_vectors.append(v / (norms + 1e-6))  # Avoid division by zero

    return torch.stack(orthogonal_vectors, dim=2)  # Shape: (B, N, C)

def modified_gram_schmidt_orthogonalization(vectors: torch.Tensor, mass: torch.Tensor | None) -> torch.Tensor:
    """
    Perform modified Gram-Schmidt orthogonalization on a set of vectors.

    Args:
        vectors (torch.Tensor): A tensor of shape (B, N, C) where B is the batch size,
                                C is the number of vectors, and N is the dimension of each vector.
        mass (torch.Tensor | None): Optional tensor of shape (B, N, 1) representing the mass of each point.

    Returns:
        torch.Tensor: A tensor of the same shape as input, with orthogonalized vectors.
    """
    with torch.autocast(vectors.device.type, enabled=False):
        # Ensure vectors are in float32 format
        vectors = vectors.float()
        m = mass.float().squeeze(-1) if mass is not None else None  # (B, N)

        batch_size, vector_dim, num_vectors = vectors.shape
        orthogonal_vectors = []  # List to store orthogonalized vectors

        for i in range(num_vectors):
            v = vectors[:, :, i].clone()  # Shape: (B, N)
            
            # Orthogonalize against all previously computed orthogonal vectors
            for j in range(i):
                u_j = orthogonal_vectors[j]  # Shape: (B, N)
                if m is not None:
                    # If mass is provided, compute weighted dot product
                    dot = torch.sum(v * u_j * m, dim=1, keepdim=True)  # Shape: (B, 1)
                else:
                    dot = torch.sum(v * u_j, dim=1, keepdim=True)  # Shape: (B, 1)
                v = v - dot * u_j  # Remove projection

            # Normalize the orthogonalized vector
            if m is not None:
                # If mass is provided, normalize with respect to the mass
                norms = torch.sqrt(torch.sum(v * v * m, dim=1, keepdim=True))  # Shape: (B, 1)
            else:
                norms = torch.norm(v, dim=1, keepdim=True)  # Shape: (B, 1)
            orthogonal_vectors.append(v / (norms + 1e-6))  # Avoid division by zero

    return torch.stack(orthogonal_vectors, dim=2)  # Shape: (B, N, C)

def qr_orthogonalization(vectors, mass, epsilon=1e-8):
    if isinstance(vectors, torch.Tensor):
        return qr_orthogonalization_torch(vectors, mass, epsilon)
    elif isinstance(vectors, np.ndarray):
        return qr_orthogonalization_numpy(vectors, mass, epsilon)
    else:
        raise TypeError("Unsupported vector type")

def qr_orthogonalization_torch(vectors: torch.Tensor, mass: torch.Tensor | None, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Perform QR orthogonalization on a set of vectors using the Gram-Schmidt process.

    Args:
        vectors (torch.Tensor): A tensor of shape (B, N, C) where B is the batch size,
                                C is the number of vectors, and N is the dimension of each vector.
        mass (torch.Tensor | None): Optional tensor of shape (B, N, 1) representing the mass of each point.

    Returns:
        torch.Tensor: A tensor of the same shape as input, with orthogonalized vectors.
    """
    with torch.autocast(vectors.device.type, enabled=False):
        # Ensure vectors are in float32 format
        vectors = vectors.float()
        if mass is not None:
            # Mass-weighted orthogonalization via diagonal sqrt(M) transform
            # Compute sqrt and rsqrt of mass for stable scaling
            massf = mass.float()
            sqrt_mass = torch.sqrt(massf + epsilon)
            rsqrt_mass = torch.rsqrt(massf + epsilon)
            # Apply sqrt(M) along the row dimension
            weighted = vectors * sqrt_mass
            Q, R = torch.linalg.qr(weighted, mode='reduced')  # QR decomposition
            # Map back to original space to get M-orthonormal columns
            Q = Q * rsqrt_mass
        else:
            Q, R = torch.linalg.qr(vectors, mode='reduced')

    return Q  # Return orthogonalized vectors (Q matrix from QR decomposition)

def qr_orthogonalization_numpy(vectors: np.ndarray, mass: np.ndarray | None, epsilon: float = 1e-8) -> np.ndarray:
    """
    Perform QR orthogonalization on a set of vectors using the Gram-Schmidt process.

    Args:
        vectors (np.ndarray): A tensor of shape (B, N, C) where B is the batch size,
                                C is the number of vectors, and N is the dimension of each vector.
        mass (np.ndarray | None): Optional tensor of shape (B, N, 1) representing the mass of each point.

    Returns:
        np.ndarray: A tensor of the same shape as input, with orthogonalized vectors.
    """
    # Ensure vectors are in float32 format
    vectors = vectors.astype(np.float32)
    if mass is not None:
        # Mass-weighted orthogonalization via diagonal sqrt(M) transform
        mass = mass.astype(np.float32)
        sqrt_mass = np.sqrt(mass + epsilon)
        rsqrt_mass = np.reciprocal(sqrt_mass)
        weighted = vectors * sqrt_mass
        Q, R = np.linalg.qr(weighted, mode='reduced')  # QR decomposition
        Q = Q * rsqrt_mass  # Map back to original space
    else:
        Q, R = np.linalg.qr(vectors, mode='reduced')

    return Q  # Return orthogonalized vectors (Q matrix from QR decomposition)


def newton_schulz(A: torch.Tensor, mass: torch.Tensor | None, num_iterations: int = 5) -> torch.Tensor:
    """
    Perform Newton-Schulz iteration for matrix square root.

    Args:
        A (torch.Tensor): Input matrix of shape (B, N, N).
        mass (torch.Tensor | None): Optional mass tensor of shape (B, N, 1).
        num_iterations (int): Number of iterations to perform.

    Returns:
        torch.Tensor: Approximated matrix square root of A.
    """
    # the iterative solver is stable that bf16 is ok.
    with torch.autocast(A.device.type, enabled=False):
        X = A.mT.float()
        # Initialize the approximation to the identity matrix
        Af = A.float()  # Ensure A is in float32 format
        a, b, c = (3.445, -4.7750, 2.0315)
        if mass is not None:
            norm_x = torch.sqrt(torch.sum(Af * Af * mass.float(), dim=(-2, -1), keepdim=True))
        else:
            norm_x = torch.linalg.norm(Af, dim=(-2, -1), keepdim=True)
        X = X / (norm_x + 1e-6)  # Normalize X to avoid numerical instability

    for _ in range(num_iterations):
        # Compute intermediate Gram matrix; do not overwrite input A
        G = torch.bmm(X, X.mT)
        B = b * G + c * torch.bmm(G, G)
        X = a * X + torch.bmm(B, X)  # Update X using the Newton-Schulz iteration
    X = X.mT  # Transpose back to original shape
    if mass is not None:
        # If mass is provided, apply it to the result
        rsqrt_mass = torch.rsqrt(mass + 1e-6)
        X = X * rsqrt_mass
    return X  # Return the approximated square root of A