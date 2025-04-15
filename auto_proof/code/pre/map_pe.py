from auto_proof.code.pre import data_utils

import numpy as np
import networkx as nx
# import graph_tool.all as gt
import torch
import glob

def map_pe_wrapper(pos_enc_dim, edges, num_vertices):
    """Creates Maximal Axis Projection (MAP) Positional Encodings

    Copied from, make sure to cite:
    https://github.com/PKU-ML/LaplacianCanonization/blob/master/data/molecules.py#L314
    """
    try:
        g = nx.Graph()
        g.add_nodes_from(range(num_vertices))
        g.add_edges_from(edges)
        # print("nodes order nx", g.nodes())
        # g = gt.Graph(edges, directed=False)
        # print("nodes order gt", list(g.vertices()))
        map_pe = map_positional_encoding(g, True, True, True, pos_enc_dim)
        return True, None, map_pe.numpy()
            
    except Exception as e:
        return False, e, None

# There are different versions of the normalized adjency matrix that I could use
def map_positional_encoding(g, use_unique_sign=True, use_unique_basis=True, use_eig_val=True, pos_enc_dim=32):
    """Creates Maximal Axis Projection (MAP) Positional Encodings

    Copied from, make sure to cite:
    https://github.com/PKU-ML/LaplacianCanonization/blob/master/data/molecules.py#L314
    """
    # A = gt.adjacency(g).astype(np.double)
    A = nx.adjacency_matrix(g).astype(np.double)
    A = torch.from_numpy(A.toarray()).double()
    # print("A", A)
    A = normalize_adjacency(A)
    # print("A normalized", A)
    # A = np.array([[0, 1, 0, 1],
    #               [1, 0, 1, 0],
    #               [0, 1, 0, 1],
    #               [1, 0, 1, 0]]).astype(np.double)
    # A = gt.laplacian(g, norm=True)
    # A = torch.from_numpy(A.toarray()).double()
    # print("A normalized", A)
    n, k = A.shape[0], pos_enc_dim
    E, U = torch.linalg.eigh(A)
    E = E.round(decimals=14)
    # print("E", E)
    # print("U", U)
    dim = min(n, k)
    # If I reverse I have to do it here also
    _, mult = torch.unique(E[-dim:], return_counts=True)
    # print("mult", mult)
    ind = torch.cat([torch.LongTensor([0]), torch.cumsum(mult, dim=0)])
    
    # If freq == 'high'
    ind += max(n - k, 0)
    # To do the freq low thing we need to do more stuff
    # if freq == 'low':
    #     ind += 1
    # print("ind", ind)
    if use_unique_sign:
        # print("entering use_unique_sign")
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                U[:, ind[i]:ind[i + 1]] = unique_sign(U[:, ind[i]:ind[i + 1]])  # eliminate sign ambiguity
    # print("U after sign", U)
    if use_unique_basis:
        # print("entering use_unique_basis")
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                continue  # single eigenvector, no basis ambiguity
            try:
                U[:, ind[i]:ind[i + 1]] = unique_basis(U[:, ind[i]:ind[i + 1]])  # eliminate basis ambiguity
            except AssertionError:
                continue  # assumption violated, skip
    # print("U after basis", U)
    if use_eig_val:
        Lambda = torch.nn.ReLU()(torch.diag(E))
        U = U @ torch.sqrt(Lambda)
    # print("U after eig val", U)
    if n < k:
        zeros = torch.zeros([n, k - n])
        U = torch.cat([U, zeros], dim=-1)
    # print("U at the end", U)
    # print("U shape before taking last k eigenvectors", U.shape)
    # if freq == 'high':
        # print('high')
    pos_enc = U[:, -k:]  # last k eigenvectors
    # elif freq == 'low':
    #     print('low')
    #     pos_enc = U[:, 1:k+1] # first k eigenvectors except first
    # else: # freq == 'mix'
    #     print('mix')
    #     pos_enc = torch.cat(U[:, 1: (k // 2) + 1], U[:, -(k // 2):])
    # Make sure that the dim stuff is the same since we're using gt instead of networkx
    # Can run it with a root thats smaller than 32
    # verify that pos_enc is the smallest eigenvectors
    return pos_enc

# There are different versions of the normalized adjency matrix that I could use
# Normalized Laplacian, Normalized Adjacency, Normalized Adjacency with pre self loop/post self loop
# Currenty this does the normalized adjacency with post self loop like in
def normalize_adjacency(A):
    """Normalize adjacency matrix

    https://github.com/PKU-ML/LaplacianCanonization/blob/4f40a2236707fb6f604aede8c3a8f12813c7db92/data/map.py#L91
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
    # print("Entering unique basis")
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
    # print("d is equal to", d)
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

def save_map_pe_roots(map_pes_dir, post_map_pes_roots_file):
    files = glob.glob(f'{map_pes_dir}*')
    roots = [files[i][-27:-5] for i in range(len(files))]
    print(roots[0])
    data_utils.save_txt(post_map_pes_roots_file, roots)

if __name__ == "__main__":
    config = data_utils.get_config()
    config['data']['root_path'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/root_ids/proofread_roots_in_train_roots_369502_913_conv.txt"
    map_pes_dir = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/map_pes/"
    # freq = 'high'
    config['data']['features_dir'] = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/features/"
    print("creating map pes")
    create_map_pe(config, map_pes_dir)

    post_map_pes_roots_file = "/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/post_map_pe_roots.txt"
    save_map_pe_roots(map_pes_dir, post_map_pes_roots_file)

    diff_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shawn.stanley/auto_proof/auto_proof/auto_proof/data/debugging_data/missing_map_pe_roots.txt'
    data_utils.compare_roots(config['data']['root_path'], post_map_pes_roots_file, diff_path)