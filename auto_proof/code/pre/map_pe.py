# Core logic from: https://github.com/PKU-ML/LaplacianCanonization/blob/master/data/molecules.py#L314

from auto_proof.code.pre import data_utils

import numpy as np
import networkx as nx
import torch
import glob
from typing import Any, Tuple # Import Any for generic types

def map_pe_wrapper(pos_enc_dim: int, edges: np.ndarray, num_vertices: int) -> Tuple[bool, Any, np.ndarray]:
    """Creates Maximal Axis Projection (MAP) Positional Encodings.

    This is a wrapper function to handle the creation of a NetworkX graph from
    edges and vertices, and then compute the MAP positional encoding. It
    includes error handling for robustness.

    Args:
        pos_enc_dim: The desired dimension of the positional encoding.
        edges: A NumPy array of shape (N, 2) representing the graph edges.
        num_vertices: The total number of vertices in the graph.

    Returns:
        A tuple:
        - bool: True if MAP PE was successfully computed, False otherwise.
        - Any: An Exception object if an error occurred, None otherwise.
        - np.ndarray: The computed MAP positional encoding as a NumPy array,
                      or None if computation failed.
    """
    try:
        g = nx.Graph()
        g.add_nodes_from(range(num_vertices))
        # Ensure edges are valid (e.g., not empty and contain valid indices)
        if edges.size > 0:
            g.add_edges_from(edges)
        
        # Parameters for map_positional_encoding are set to True for all ambiguity resolutions
        map_pe = map_positional_encoding(g, use_unique_sign=True, use_unique_basis=True, use_eig_val=True, pos_enc_dim=pos_enc_dim)
        return True, None, map_pe.numpy()
            
    except Exception as e:
        # Catch any exception during graph creation or PE computation
        return False, e, None
    try:
        g = nx.Graph()
        g.add_nodes_from(range(num_vertices))
        g.add_edges_from(edges)
        map_pe = map_positional_encoding(g, True, True, True, pos_enc_dim)
        return True, None, map_pe.numpy()
            
    except Exception as e:
        return False, e, None

def map_positional_encoding(g, use_unique_sign=True, use_unique_basis=True, use_eig_val=True, pos_enc_dim=32):
    """Creates Maximal Axis Projection (MAP) Positional Encodings.

    This function computes Maximal Axis Projection (MAP) Positional Encodings
    for a given graph. It involves constructing and normalizing the adjacency matrix,
    computing its eigenvectors and eigenvalues, and applying various transformations
    to ensure uniqueness and desired properties of the positional encoding.

    Core logic from:
    https://github.com/PKU-ML/LaplacianCanonization/blob/master/data/molecules.py#L314

    Args:
        g: A NetworkX graph object.
        use_unique_sign: If True, resolves sign ambiguity of eigenvectors.
        use_unique_basis: If True, resolves basis ambiguity for repeated eigenvalues.
        use_eig_val: If True, scales eigenvectors by the square root of eigenvalues.
        pos_enc_dim: The desired dimension (k) of the positional encoding.

    Returns:
        torch.Tensor: A tensor of shape [n, pos_enc_dim] representing the
                      MAP positional encoding for the graph, where n is the number of nodes.
    """
    A = nx.adjacency_matrix(g).astype(np.double)
    A = torch.from_numpy(A.toarray()).double()

    A = normalize_adjacency(A)

    n, k = A.shape[0], pos_enc_dim
    E, U = torch.linalg.eigh(A)
    E = E.round(decimals=14)

    dim = min(n, k)
    _, mult = torch.unique(E[-dim:], return_counts=True)
    ind = torch.cat([torch.LongTensor([0]), torch.cumsum(mult, dim=0)])
    
    ind += max(n - k, 0)
    if use_unique_sign:
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                U[:, ind[i]:ind[i + 1]] = unique_sign(U[:, ind[i]:ind[i + 1]])  # eliminate sign ambiguity
    if use_unique_basis:
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                continue  # single eigenvector, no basis ambiguity
            try:
                U[:, ind[i]:ind[i + 1]] = unique_basis(U[:, ind[i]:ind[i + 1]])  # eliminate basis ambiguity
            except AssertionError:
                continue  # assumption violated, skip
    if use_eig_val:
        Lambda = torch.nn.ReLU()(torch.diag(E))
        U = U @ torch.sqrt(Lambda)
    if n < k:
        zeros = torch.zeros([n, k - n])
        U = torch.cat([U, zeros], dim=-1)
    pos_enc = U[:, -k:]
    return pos_enc

def normalize_adjacency(A):
    """Normalizes the adjacency matrix using symmetric normalization ($D^{-1/2} A D^{-1/2} + I$).

    This normalization is commonly used in Graph Neural Networks. It first
    computes the inverse square root of the degree matrix and then applies
    it to the adjacency matrix. Finally, the identity matrix is added.

    Args:
        A: A square adjacency matrix as a torch.Tensor.

    Returns:
        torch.Tensor: The symmetrically normalized adjacency matrix.
    """
    n = A.shape[0]
    assert list(A.shape) == [n, n]

    d = torch.sum(A, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    D_inv_sqrt[D_inv_sqrt == float("inf")] = 0.
    A = D_inv_sqrt @ A @ D_inv_sqrt
    A += torch.eye(n)
    return A

def unique_sign(U):
    """
    Eliminating sign ambiguity of the input eigenvectors.

    >>> U = Tensor([[1, -1, 4], [2, -2, 5], [3, -3, -6]])
    >>> unique_sign(U)
    tensor([[ 1.,  1.,  4.],
            [ 2.,  2.,  5.],
            [ 3.,  3., -6.]])
    >>> U = Tensor([[2, -2, 5], [3, -3, -6], [1, -1, 4]])
    >>> unique_sign(U)
    tensor([[ 2.,  2.,  5.],
            [ 3.,  3., -6.],
            [ 1.,  1.,  4.]])

    Args: 
        Tensor of shape [n, d]. Each column of U is an eigenvector.
    Return: 
        Tensor of shape [n, d].
    """
    n, d = U.shape
    for i in range(d):
        u = U[:, i].view(n, 1)
        P = u @ u.T.view(1, n)
        E = torch.eye(n)
        J = torch.ones(n)
        Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
        Pe = [i for i in enumerate(Pe)]
        Pe.sort(key=lambda x: x[1])
        indices = [i[0] for i in Pe]
        lengths = [i[1] for i in Pe]
        _, counts = np.unique(lengths, return_counts=True)
        step = 0
        X = torch.zeros([len(counts), n]).double()
        for j in range(len(counts)):
            for _ in range(counts[j]):
                X[j] += E[indices[step]]
                step += 1
            X[j] += 10 * J
        u_0, x = torch.zeros(n), torch.zeros(n)
        flag = True
        for j in range(len(counts)):
            u_0 = P @ X[j]
            if torch.linalg.vector_norm(u_0).round(decimals=12) != 0:
                x = X[j]
                flag = False
                break
        if flag:  # violates sign assumption, skip
            continue
        u = u.view(n)
        u_0 /= torch.abs(u @ x)
        U[:, i] = u_0
    return U

def unique_basis(U_i):
    """Eliminating basis ambiguity of the input eigenvectors.

    Args:
        U_i: Tensor of shape [n, d]. Each column of U is an eigenvector.
    Return:
        Tensor of shape [n, d].
    """
    n, d = U_i.shape
    E = torch.eye(n)
    J = torch.ones(n)
    P = U_i @ U_i.T
    Pe = [torch.linalg.vector_norm(P[:, i]).round(decimals=14).item() for i in range(n)]
    Pe = [i for i in enumerate(Pe)]
    Pe.sort(key=lambda x: x[1])
    indices = [i[0] for i in Pe]
    lengths = [i[1] for i in Pe]
    _, counts = np.unique(lengths, return_counts=True)
    assert len(counts) >= d  # basis assumption 1
    X = torch.zeros([d, n]).double()  # [x_1, ..., x_d]
    step = -1
    for i in range(1, d + 1):
        x = torch.zeros(n)
        for _ in range(counts[-i]):
            x += E[indices[step]]
            step -= 1
        X[i - 1] = x + 10 * J
    U_0 = torch.zeros([n, d])  # the unique basis
    u_span = torch.empty([n, 0])  # span(u_1, ..., u_{i-1})
    u_perp = U_i.clone()  # orthogonal complementary space
    for i in range(d):
        P_perp = u_perp @ u_perp.T
        u_i = P_perp @ X[i]
        assert torch.linalg.vector_norm(u_i) != 0  # basis assumption 2
        u_i = torch.nn.functional.normalize(u_i, dim=0)
        U_0[:, i] = u_i
        u_span = torch.cat([u_span, u_i.unsqueeze(dim=1)], dim=1)
        u_perp = find_complementary_space(U_i, u_span)
    return U_0

def find_complementary_space(U, u_span):
    """
    Find the orthogonal complementary space of u_span in the linear space U.

    >>> U = Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    >>> u_span = Tensor([[0, 0], [1, 0], [0, 0], [0, 0], [0, 1]])
    >>> find_complementary_space(U, u_span)
    tensor([[1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.],
            [0., 0.]])

    Args:
        U: Tensor of shape [n, d].
        u_span: Tensor of shape [n, s], where s <= d.
    Return: 
        Tensor of shape [n, d - s].
    """
    n, d = U.shape
    s = u_span.shape[1]
    u_base = u_span.clone()
    for j in range(d):
        i = u_base.shape[1]
        u_j = U[:, j].unsqueeze(dim=1)  # shape [n, 1]
        u_temp = torch.cat([u_base, u_j], dim=1)  # shape [n, d'] where i <= d' <= d
        if torch.linalg.matrix_rank(u_temp) == i + 1:  # u_temp are linear independent
            u_base = u_temp
        if u_base.shape[1] == d:
            break
    u_base = orthogonalize(u_base)
    u_perp = u_base[:, s:d]
    return u_perp

def orthogonalize(U):
    """
    Orthogonalize a set of linear independent vectors using Gram–Schmidt process.

    >>> U = torch.nn.functional.normalize(torch.randn(5, 3), dim=0)
    >>> U = orthogonalize(U)
    >>> torch.allclose(U.T @ U, torch.eye(3), atol=1e-06)
    True

    Args:
        U: Tensor of shape [n, d], d <= n.
    Return:
        Tensor of shape [n, d].
    """
    Q, R = torch.linalg.qr(U)
    return Q