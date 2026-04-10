import torch
import torch.nn.functional as F
import torch.nn as nn

import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR

from utils import log_batch_similarity_stats


class LookupEmbedding(nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim, hidden_dim=64, out_dim=10, uni_lambda=0.1):
        super().__init__()
        self.uid_embedding = nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = nn.Embedding(iid_all + 1, emb_dim)

        # user mlp
        self.user_mlp = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

        # item mlp
        self.item_mlp = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

        self.uni_lambda = uni_lambda

    def uniformity_loss(self, z, t=2.0):
        z = F.normalize(z, dim=1)
        sq_pdist = torch.pdist(z, p=2).pow(2)
        return torch.log(torch.exp(-t * sq_pdist).mean() + 1e-8)

    def forward(self, x, return_loss=False):
        uid_emb = self.uid_embedding(x[:, 0])  # [B, D]
        iid_emb = self.iid_embedding(x[:, 1])  # [B, D]

        user_vec = self.user_mlp(uid_emb)  # [B, d]
        item_vec = self.item_mlp(iid_emb)  # [B, d]

        emb = torch.stack([user_vec, item_vec], dim=1)  # [B, 2, d]

        if return_loss:
            user_uni = self.uniformity_loss(user_vec)
            item_uni = self.uniformity_loss(item_vec)

            uni_loss = self.uni_lambda * (user_uni + item_uni)
            return emb, uni_loss

        return emb


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(), torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)


class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, meta_dim_0):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)

        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.global_step = 0

    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False):
        # self.global_step += 1

        if stage == "train_src":
            emb, uni_loss = self.src_model.forward(x, return_loss=True)  # [B, 2, d]

            user_emb = emb[:, 0, :]  # [B, d]
            item_emb = emb[:, 1, :]  # [B, d]

            log_batch_similarity_stats(user_emb, global_step=self.global_step, log_every=600, prefix="train_src")

            x = torch.sum(user_emb * item_emb, dim=1)
            return x, uni_loss

        elif stage == "train_tgt":
            emb, uni_loss = self.tgt_model.forward(x, return_loss=True)

            user_emb = emb[:, 0, :]  # [B, d]
            item_emb = emb[:, 1, :]  # [B, d]

            log_batch_similarity_stats(user_emb, global_step=self.global_step, log_every=600, prefix="train_tgt")

            x = torch.sum(user_emb * item_emb, dim=1)
            return x, uni_loss

        elif stage == "test_tgt":
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x

        elif stage in ["train_aug", "test_aug"]:
            emb = self.aug_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x

        elif stage in ["train_meta", "test_meta"]:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output

        elif stage == "train_map":
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb

        elif stage == "test_map":
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x

        elif stage == "train_diff":

            tgt_uid, iid_input, y_input = x

            tgt_emb = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            tgt_emb = self.tgt_model.user_mlp(tgt_emb)

            cond_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            cond_emb = self.src_model.user_mlp(cond_emb)

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()
            iid_emb = self.tgt_model.item_mlp(iid_emb)

            if is_task == False:
                loss = Diff.diffusion_loss_fn(diff_model, tgt_emb, cond_emb, iid_emb, y_input, device, is_task)
                return loss
            else:
                align_loss, task_loss, uni_loss = Diff.diffusion_loss_fn(diff_model, tgt_emb, cond_emb, iid_emb, y_input, device, is_task)
                return align_loss, task_loss, uni_loss

        elif stage == "test_diff":

            tgt_uid, iid_input, _ = x

            tgt_emb = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            tgt_emb = self.tgt_model.user_mlp(tgt_emb)

            cond_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            cond_emb = self.src_model.user_mlp(cond_emb)

            # mu = cond_emb.mean(dim=0, keepdim=True)  # [1, D]
            # std = cond_emb.std(dim=0, keepdim=True)  # [1, D]
            # cond_emb = torch.randn_like(cond_emb) * std + mu

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()
            iid_emb = self.tgt_model.item_mlp(iid_emb)

            # final_output_raw, iid_emb_out = Diff.p_sample_loop(diff_model, cond_emb, iid_emb, device)
            # final_output_raw, iid_emb = Diff.p_sample_loop_naive(diff_model, cond_emb, iid_emb, device, start_mode="x0_forward", x0_ref=tgt_emb)
            # final_output_raw, iid_emb = Diff.p_sample_loop_naive(diff_model, cond_emb, iid_emb, device, start_mode="cond")
            # final_output_raw, iid_emb = Diff.p_sample_loop_naive(diff_model, cond_emb, iid_emb, device, start_mode="noise")
            # final_output_raw, iid_emb = Diff.p_sample_loop_x0(diff_model, cond_emb, iid_emb, device, start_mode="cond")
            final_output_raw, iid_emb = Diff.p_sample_loop_x0_solver(
                model=diff_model, cond_emb=cond_emb, iid_emb=iid_emb, device=device, start_mode="cond", sample_steps=20, eta=0.0
            )

            final_output_proj = diff_model.al_linear(final_output_raw)

            x = torch.sum(final_output_proj * iid_emb, dim=1)

            return x

        elif stage == "train_ss":
            x_u, x_p_i, x_n_i, x_t_u = x

            x_u_emb = self.src_model.uid_embedding(x_u.unsqueeze(1)).squeeze()
            x_p_i_emb = self.src_model.iid_embedding(x_p_i.unsqueeze(1)).squeeze()
            x_n_i_emb = self.src_model.iid_embedding(x_n_i.unsqueeze(1)).squeeze()
            x_t_u_emb = self.tgt_model.uid_embedding(x_t_u.unsqueeze(1)).squeeze()

            loss = SSCDR.sscdr_loss_fn(ss_model.forward(x_u_emb), ss_model.forward(x_p_i_emb), ss_model.forward(x_n_i_emb), x_t_u_emb)
            return loss

        elif stage == "test_ss":
            uid_emb = ss_model.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x

        elif stage == "train_la":
            x_uid, x_mask_src, x_mask_tgt = x

            x_u_emb_s = self.src_model.uid_embedding(x_uid.unsqueeze(1)).squeeze()
            x_u_emb_t = self.tgt_model.uid_embedding(x_uid.unsqueeze(1)).squeeze()
            x_mask_src = x_mask_src.unsqueeze(1)
            x_mask_tgt = x_mask_tgt.unsqueeze(1)

            loss = LACDR.lacdr_loss_fn(la_model, x_u_emb_s, x_mask_src, x_u_emb_t, x_mask_tgt)
            return loss

        elif stage == "test_la":
            uid_emb = la_model.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
