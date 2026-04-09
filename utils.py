import torch
import torch.nn.functional as F


def log_batch_similarity_stats(user_emb, global_step, log_every=200, prefix="train_src"):
    if global_step % log_every != 0:
        return

    with torch.no_grad():
        # -------------------------
        # user-user cosine (off-diagonal)
        # -------------------------
        user_norm = F.normalize(user_emb, p=2, dim=1)
        user_sim = torch.matmul(user_norm, user_norm.t())  # [B, B]

        user_mask = ~torch.eye(user_sim.size(0), dtype=torch.bool, device=user_sim.device)
        user_offdiag = user_sim[user_mask]

        print(
            f"[{prefix}] [step {global_step}] "
            f"user offdiag cosine mean={user_offdiag.mean().item():.4f}, "
            f"std={user_offdiag.std().item():.4f}, "
            f"min={user_offdiag.min().item():.4f}, "
            f"max={user_offdiag.max().item():.4f}"
        )
