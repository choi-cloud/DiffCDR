import torch
import torch.nn.functional as F

import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR

from rqvae import ResidualQuantizer


class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, meta_dim_0, codebook_level, codebook_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)

        self.rq = ResidualQuantizer(code_dim=emb_dim, num_levels=codebook_level, codebook_size=codebook_size)

    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False):
        if stage == "train_src":
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ["train_tgt", "test_tgt"]:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage == "train_diff":
            tgt_uid, iid_input, y_input = x  # [B], [B,1], [B,1]

            # x_0 역할: 타깃 도메인 유저 임베딩
            tgt_emb = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # [B, emb_dim]

            # RQ-VAE 입력: 소스 도메인 uid 임베딩
            src_uid_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # [B, emb_dim]

            # 🔥 RQ-VAE 통과: 코드북 레벨 벡터 학습
            quantized, all_level_vectors, rq_loss = self.rq(src_uid_emb)
            # all_level_vectors: [L, B, emb_dim]  (다음 단계에서 diffusion cond로 쓸 예정)

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()  # [B, emb_dim]

            # 아직은 DiffCDR는 원래대로 src_uid_emb를 cond_emb로 사용
            diff_loss = Diff.diffusion_loss_fn(
                diff_model,
                tgt_emb,  # x_0
                src_uid_emb,  # cond_emb (is_task=False에서 사용)
                iid_emb,
                y_input,
                device,
                is_task,
                all_level_vectors=all_level_vectors,  # 🔥 추가
            )

            # 🔥 코드북도 같이 학습
            alpha_rq = 1e-2  # 튜닝 가능
            total_loss = diff_loss + alpha_rq * rq_loss

            return total_loss
        elif stage == "test_diff":
            tgt_uid, iid_input, _ = x

            src_uid_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # [B, D]
            quantized, all_level_vectors, _ = self.rq(src_uid_emb)  # [L, B, D]

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()  # [B, D]

            # 🔥 RQ 기반 샘플링 사용
            trans_emb, iid_emb_out = Diff.p_sample_loop_with_rq(diff_model, all_level_vectors, iid_emb, device)

            x = torch.sum(trans_emb * iid_emb_out, dim=1)
            return x
