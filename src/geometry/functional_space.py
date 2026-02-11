import torch

@torch.no_grad()
def get_unembedding_weight(model, device="cpu"):
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        raise ValueError("model.get_output_embeddings() returned None; can't find unembedding.")
    W = out_emb.weight.detach().to(device)

    # Ensure shape is (vocab, d_model)
    # (Most models store as (vocab, d_model); some store transposed.)
    if W.ndim != 2:
        raise ValueError(f"Unexpected unembedding weight shape: {tuple(W.shape)}")
    return W

@torch.no_grad()
def get_functional_basis_from_unembedding(model, Df: int, device="cpu"):
    """
    Returns U_f: (d_model, Df) top right-singular directions of W_U.
    evals_top: (Df,)    corresponding eigenvalues of (W_U^T W_U)
    """
    WU = get_unembedding_weight(model, device=device).float()  # (vocab, d)

    # C = WU^T WU  (d_model x d_model)
    C = WU.T @ WU

    # eigh gives ascending eigenvalues
    evals, evecs = torch.linalg.eigh(C)

    # take top Df
    U_f = evecs[:, -Df:]  # (d_model, Df)
    evals_top = evals[-Df:]

    # sort descending by eigenvalue
    idx = torch.argsort(evals_top, descending=True)
    U_f = U_f[:, idx]
    evals_top = evals_top[idx]

    return U_f, evals_top

def project_functional(x_cpu, U_f_cpu):
    # x_cpu: (..., d_model)
    # returns (..., Df)
    return x_cpu @ U_f_cpu