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
        z: [B, D]
        returns:
            quantized: [B, D]
            all_level_vectors: [L, B, D]
            rq_loss: scalar
        """
        B, D = z.shape
        device = z.device

        residual = z
        all_level_vectors = []
        rq_loss = 0.0

        for l in range(self.num_levels):
            # [K, D]
            codebook_l = self.codebooks[l]  # [codebook_size, code_dim]

            # 거리 계산: [B, K]
            residual_expanded = residual.unsqueeze(1)  # [B, 1, D]
            codebook_expanded = codebook_l.unsqueeze(0)  # [1, K, D]
            dist = torch.sum((residual_expanded - codebook_expanded) ** 2, dim=-1)  # [B, K]

            # 가장 가까운 코드 선택
            idx = torch.argmin(dist, dim=-1)  # [B]
            chosen = codebook_l[idx]  # [B, D]

            rq_loss = rq_loss +  F.mse_loss(chosen, residual.detach())
            
            # Commitment loss: 입력 임베딩(residual)을 선택된 코드북 벡터 근처로 끌어당겨
            # 양자화 공간에 안정적으로 정착시키기 위한 loss
            # rq_loss = rq_loss + 0.25*F.mse_loss(residual, chosen.detach())

            # STE (Straight-Through Estimator): forward에서는 양자화된 벡터를 사용하되,
            # backward에서는 gradient가 residual(z)로 그대로 흐르도록 만드는 장치
            chosen_ste = residual + (chosen - residual).detach()

            all_level_vectors.append(chosen_ste)

            # residual 업데이트
            residual = residual - chosen

        # [L, B, D]
        all_level_vectors = torch.stack(all_level_vectors, dim=0)
        quantized = all_level_vectors.sum(dim=0)  # [B, D]

        return quantized, all_level_vectors, rq_loss


class SingleVQ(nn.Module):
    """
    코드북 1개를 가진 독립 VQ
    - 입력 z를 직접 quantize
    - residual 없음
    """
    def __init__(self, code_dim: int, codebook_size: int):
        super().__init__()
        self.code_dim = code_dim
        self.codebook_size = codebook_size

        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim) * 0.1)

    def forward(self, z: torch.Tensor):
        """
        z: [B, D]
        returns:
            chosen_ste: [B, D]
            rq_loss: scalar
            idx: [B]
        """
        z_expanded = z.unsqueeze(1)                    # [B, 1, D]
        codebook_expanded = self.codebook.unsqueeze(0)  # [1, K, D]
        dist = torch.sum((z_expanded - codebook_expanded) ** 2, dim=-1)  # [B, K]

        idx = torch.argmin(dist, dim=-1)   # [B]
        chosen = self.codebook[idx]        # [B, D]

        # 각 VQ는 원본 z를 직접 복원하도록 독립 학습
        rq_loss = F.mse_loss(chosen, z.detach())

        # STE
        chosen_ste = z + (chosen - z).detach()

        return chosen_ste, rq_loss, idx


class MultiLevelVQ(nn.Module):
    """
    SingleVQ 10개를 가진 multi-level VQ
    - 각 레벨은 독립 VQ
    - 모든 레벨이 같은 원본 z를 입력으로 받음
    - 반환 형식은 기존 ResidualQuantizer와 동일
    """
    def __init__(self, code_dim: int, num_levels: int = 4, codebook_sizes=None):
        super().__init__()

        if codebook_sizes is None:
            codebook_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        self.code_dim = code_dim
        self.codebook_sizes = codebook_sizes
        self.total_num_levels = len(codebook_sizes)
        self.num_levels = num_levels

        assert 1 <= self.num_levels <= self.total_num_levels

        self.vqs = nn.ModuleList([
            SingleVQ(code_dim=code_dim, codebook_size=k)
            for k in self.codebook_sizes
        ])

    def get_pretrain_num_levels(self):
        return self.total_num_levels

    def forward(self, z: torch.Tensor, num_levels: int = None):
        """
        z: [B, D]
        returns:
            quantized: [B, D]
            all_level_vectors: [L, B, D]
            rq_loss: scalar
        """
        if num_levels is None:
            num_levels = self.num_levels

        assert 1 <= num_levels <= self.total_num_levels

        all_level_vectors = []
        rq_loss = z.new_tensor(0.0)

        for l in range(num_levels):
            chosen_ste, loss_l, _ = self.vqs[l](z)   # 모든 레벨이 같은 z 사용
            all_level_vectors.append(chosen_ste)
            rq_loss = rq_loss + loss_l

        all_level_vectors = torch.stack(all_level_vectors, dim=0)  # [L, B, D]
        quantized = all_level_vectors.sum(dim=0)                   # [B, D]

        return quantized, all_level_vectors, rq_loss