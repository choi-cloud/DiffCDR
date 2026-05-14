# rqvae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualQuantizer(nn.Module):
    """
    아주 간단한 RQ-VAE 스타일 residual quantizer.
    - 입력: z [B, D] (여기서는 src uid embedding)
    - 출력:
        quantized: [B, D]        (모든 레벨 합친 최종 벡터)
        all_level_vectors: [L, B, D]  (레벨별 코드벡터)
        rq_loss: scalar (코드북 학습용 loss)
    """

    def __init__(self, code_dim: int, num_levels: int = 4, codebook_size: int = 256):
        super().__init__()
        self.code_dim = code_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size

        # [num_levels, codebook_size, code_dim]
        self.codebooks = nn.Parameter(torch.randn(num_levels, codebook_size, code_dim) * 0.1)

    def forward(self, z: torch.Tensor):
        """
        z: [B, D] (fixed embedding)

        returns:
            quantized: [B, D]
            all_level_vectors: [L, B, D]
            total_loss: scalar
        """

        B, D = z.shape

        # encoder 학습 안 하므로 detach
        z = z.detach()

        residual = z
        all_level_vectors = []

        rq_loss = 0.0

        for l in range(self.num_levels):

            # [K, D]
            codebook_l = self.codebooks[l]

            # ------------------------
            # nearest code search
            # ------------------------
            residual_expanded = residual.unsqueeze(1)  # [B, 1, D]
            codebook_expanded = codebook_l.unsqueeze(0)  # [1, K, D]

            dist = torch.sum((residual_expanded - codebook_expanded) ** 2, dim=-1)  # [B, K]

            idx = torch.argmin(dist, dim=-1)  # [B]

            # [B, D]
            chosen = codebook_l[idx]

            # ------------------------
            # codebook fitting loss
            # ------------------------
            rq_loss = rq_loss + F.mse_loss(chosen, residual)

            # 저장
            all_level_vectors.append(chosen)

            # ------------------------
            # residual update
            # ------------------------
            residual = residual - chosen.detach()

        # ------------------------
        # stack
        # ------------------------
        all_level_vectors = torch.stack(all_level_vectors, dim=0)  # [L, B, D]

        # additive reconstruction
        quantized = all_level_vectors.sum(dim=0)  # [B, D]

        # ------------------------
        # global reconstruction loss
        # ------------------------
        recon_loss = F.mse_loss(quantized, z)

        recon_lambda = 1.0

        total_loss = rq_loss + recon_lambda * recon_loss

        return quantized, all_level_vectors, total_loss
