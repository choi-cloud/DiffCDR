import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    def __init__(self, code_dim: int, num_levels: int = 4, codebook_size=None, level_loss_weights=None, recon_lambda=1.0):
        super().__init__()

        self.code_dim = code_dim
        self.num_levels = num_levels
        self.recon_lambda = recon_lambda

        # -------------------------------------------------
        # codebook size parsing
        # -------------------------------------------------
        if codebook_size is None:
            codebook_size = [4, 16, 16, 16]

        # argparse string 처리
        if isinstance(codebook_size, str):
            codebook_size = eval(codebook_size)

        # int면 모든 level 동일하게
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_levels

        assert len(codebook_size) == num_levels

        self.codebook_sizes = codebook_size

        # -------------------------------------------------
        # level weights
        # -------------------------------------------------
        if level_loss_weights is None:
            level_loss_weights = [0.25, 1.0, 1.0, 1.0]

        if isinstance(level_loss_weights, str):
            level_loss_weights = eval(level_loss_weights)

        assert len(level_loss_weights) == num_levels

        self.register_buffer(
            "level_loss_weights",
            torch.tensor(level_loss_weights, dtype=torch.float32),
        )

        # -------------------------------------------------
        # codebooks
        # -------------------------------------------------
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(k, code_dim) * 0.1) for k in codebook_size])

    def forward(self, z: torch.Tensor):
        """
        z: [B, D]

        returns:
            quantized: [B, D]
            all_level_vectors: [L, B, D]
            total_loss: scalar
        """
        B, D = z.shape

        z = z.detach()

        residual = z
        all_level_vectors = []

        rq_loss = z.new_tensor(0.0)

        for l in range(self.num_levels):
            codebook_l = self.codebooks[l]  # [K_l, D]

            dist = (
                torch.cdist(
                    residual.unsqueeze(0),
                    codebook_l.unsqueeze(0),
                    p=2,
                ).squeeze(0)
                ** 2
            )  # [B, K_l]

            idx = torch.argmin(dist, dim=-1)  # [B]
            chosen = codebook_l[idx]  # [B, D]

            level_loss = F.mse_loss(chosen, residual.detach())
            rq_loss = rq_loss + self.level_loss_weights[l] * level_loss

            all_level_vectors.append(chosen)

            residual = residual - chosen.detach()

        all_level_vectors = torch.stack(all_level_vectors, dim=0)  # [L, B, D]
        quantized = all_level_vectors.sum(dim=0)  # [B, D]

        recon_loss = F.mse_loss(quantized, z)

        total_loss = rq_loss + self.recon_lambda * recon_loss

        return quantized, all_level_vectors, total_loss
