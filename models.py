import torch
import torch.nn.functional as F
import torch.nn as nn

import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR
from rqvae import ResidualQuantizer
from utils import AttentionLayer

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

        self.graph_emb_cache = {}  # 🔥 Cache for graph embeddings

    def clear_graph_cache(self):
        self.graph_emb_cache = {}

    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False, item_cond=False, style_src=None):
        if stage == "train_src":
            emb, uni_loss = self.src_model.forward(x, return_loss=True)  # [B, 2, d]

            user_emb = emb[:, 0, :]  # [B, d]
            item_emb = emb[:, 1, :]  # [B, d]

            # log_batch_similarity_stats(user_emb, global_step=self.global_step, log_every=600, prefix="train_src")

            x = torch.sum(user_emb * item_emb, dim=1)
            return x, uni_loss

        elif stage == "train_tgt":
            emb, uni_loss = self.tgt_model.forward(x, return_loss=True)

            user_emb = emb[:, 0, :]  # [B, d]
            item_emb = emb[:, 1, :]  # [B, d]

            # log_batch_similarity_stats(user_emb, global_step=self.global_step, log_every=600, prefix="train_tgt")

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

        elif stage == "train_diff":  # DiffCDR - train

            tgt_uid, iid_input, y_input = x

            tgt_emb = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            cond_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            loss = Diff.diffusion_loss_fn(diff_model, tgt_emb, cond_emb, iid_emb, y_input, device, is_task)
            return loss  # is_task=False: 노이즈 예측 , is_task=True: ALS, pred 로스

        elif stage == "test_diff":  # DiffCDR - test

            tgt_uid, iid_input, _ = x

            cond_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            trans_emb, iid_emb_out = Diff.p_sample_loop(diff_model, cond_emb, iid_emb, device)

            x = torch.sum(trans_emb * iid_emb_out, dim=1)
            return x

        elif stage == "train_diff_parallel":  # DiffParallel - train

            tgt_uid, iid_input, y_input = x
            tgt_emb1 = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF
            tgt_emb1 = self.tgt_model.user_mlp(tgt_emb1)

            src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF
            src_uid_emb1 = self.src_model.user_mlp(src_uid_emb1)

            tgt_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=True)  # Aggr
            src_uid_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)  # Aggr

            cond_emb1 = src_uid_emb1
            cond_emb2 = src_uid_emb2

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()
            iid_emb = self.tgt_model.item_mlp(iid_emb)

            # ! mf 임베딩과 aggr 임베딩 양자화
            if diff_model.rqvae["RQVAE"] == True:
                quantized1, all_level_vectors1, rq_loss1 = diff_model.rq_mf(cond_emb1)  # [L, B, D]
                quantized2, all_level_vectors2, rq_loss2 = diff_model.rq_aggr(cond_emb2)

            else:
                all_level_vectors1 = cond_emb1
                all_level_vectors2 = cond_emb2
                quantized1, quantized2 = None, None

            # is_task=False: 노이즈 예측 , is_task=True: ALS + task 로스
            if is_task == False:
                loss = Diff.diffusion_loss_fn_parallel(
                    diff_model,
                    tgt_emb1,
                    tgt_emb2,
                    src_uid_emb1,
                    src_uid_emb2,
                    iid_emb,
                    y_input,
                    device,
                    is_task,
                    q_embs1=all_level_vectors1,
                    q_embs2=all_level_vectors2,
                    style_src=style_src,
                    uid=tgt_uid,
                    iid=iid_input,
                    Q_emb1=quantized1,
                    Q_emb2=quantized2,
                )
                return loss
            else:
                task_loss, uni_loss = Diff.diffusion_loss_fn_parallel(
                    diff_model,
                    tgt_emb1,
                    tgt_emb2,
                    src_uid_emb1,
                    src_uid_emb2,
                    iid_emb,
                    y_input,
                    device,
                    is_task,
                    q_embs1=all_level_vectors1,
                    q_embs2=all_level_vectors2,
                    style_src=style_src,
                    uid=tgt_uid,
                    iid=iid_input,
                    Q_emb1=quantized1,
                    Q_emb2=quantized2,
                )
                return task_loss, uni_loss

        elif stage == "test_diff_parallel":  # DiffParallel - test

            tgt_uid, iid_input, _ = x

            src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF
            src_uid_emb1 = self.src_model.user_mlp(src_uid_emb1)

            src_uid_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)

            cond_emb1 = src_uid_emb1
            cond_emb2 = src_uid_emb2
            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()
            iid_emb = self.tgt_model.item_mlp(iid_emb)

            if diff_model.rqvae["RQVAE"] == True:
                quantized1, all_level_vectors1, _ = diff_model.rq_mf(cond_emb1)  # [L, B, D]
                quantized2, all_level_vectors2, _ = diff_model.rq_aggr(cond_emb2)

                cond1, cond2 = all_level_vectors1, all_level_vectors2
                p_sample = Diff.p_sample_loop_x0_solver

            else:
                cond1, cond2 = src_uid_emb1, src_uid_emb2
                p_sample = Diff.p_sample_loop_x0_solver

            if diff_model.rqvae["start_point"] == "src_u":
                start1, start2 = src_uid_emb1, src_uid_emb2
            elif diff_model.rqvae["start_point"] == "quant_u":
                start1, start2 = quantized1, quantized2
            elif diff_model.rqvae["start_point"] == "noise":
                start1, start2 = torch.randn_like(src_uid_emb1), torch.randn_like(src_uid_emb2)
            if diff_model.parallel["batch_norm"]:
                iid_emb = diff_model.ln_iid(iid_emb)

            if diff_model.aggregation == "aggregation":
                final_output_m, iid_emb = p_sample(diff_model, cond1, iid_emb, device, diff_id=0)
                final_output_g, iid_emb = p_sample(diff_model, cond2, iid_emb, device, diff_id=1)

                # final_output_m = diff_model.ln_m(diff_model.linear_m(final_output_m))
                # final_output_g = diff_model.ln_g(diff_model.linear_g(final_output_g))
                if diff_model.parallel["batch_norm"]:
                    final_output_m = diff_model.ln_m(final_output_m)
                    final_output_g = diff_model.ln_g(final_output_g)
                base_tokens = torch.stack([final_output_m, final_output_g], dim=1)

            elif diff_model.aggregation == "aggregation_ab1":
                final_output_m, iid_emb = p_sample(diff_model, cond1, iid_emb, device, diff_id=0)
                if diff_model.parallel["batch_norm"]:
                    # final_output_m, iid_emb = p_sample(diff_model, start1, cond1, iid_emb, device, diff_id=0)
                    # final_output_m = diff_model.ln_m(diff_model.linear_m(final_output_m))
                    final_output_m = diff_model.ln_m(final_output_m)
                base_tokens = torch.stack([final_output_m], dim=1)

            elif diff_model.aggregation == "aggregation_ab2":
                final_output_g, iid_emb = p_sample(diff_model, start2, cond2, iid_emb, device, diff_id=0)
                final_output_g = diff_model.ln_g(diff_model.linear_g(final_output_g))
                base_tokens = torch.stack([final_output_g], dim=1)

            if diff_model.parallel["set_aggr"] == "item":
                tokens = base_tokens

            elif diff_model.parallel["set_aggr"] == "item_i":
                iid = iid_input.squeeze(1)
                style_tgt_item = diff_model.style_tgt_item.to(start1.device)  # [I_total, F_item]
                style_i = style_tgt_item[iid][:, :2]  # (B, F_item)
                item_style_tok = diff_model.item_style_encoder(style_i)  # (B, D)
                item_style_tok = diff_model.item_style_ln(item_style_tok)  # (B, D)
                item_style_tok = diff_model.item_style_scale * item_style_tok  # (B, D)

                tokens = torch.cat([base_tokens, item_style_tok.unsqueeze(1)], dim=1)

            elif diff_model.parallel["set_aggr"] == "item_iu":
                uid = tgt_uid.long()  # (B,)
                iid = iid_input.squeeze(1)
                style_src = style_src.to(start1.device)
                style_u = style_src[uid][:, :2]  # (B, F)

                if diff_model.parallel["bias_mapping"] == "user":
                    style_u = diff_model.user_style_mapper(style_u)
                    style_u = style_u.detach()

                style_tok = diff_model.style_encoder(style_u)  # (B, D)
                style_tok = diff_model.style_ln(style_tok)  # (B, D)
                style_tok_u = diff_model.style_scale * style_tok  # (B, D)

                style_tgt_item = diff_model.style_tgt_item.to(start1.device)  # [I_total, F_item]
                style_i = style_tgt_item[iid][:, :2]  # (B, F_item)
                item_style_tok = diff_model.item_style_encoder(style_i)  # (B, D)
                item_style_tok = diff_model.item_style_ln(item_style_tok)  # (B, D)
                item_style_tok = diff_model.item_style_scale * item_style_tok  # (B, D)

                tokens = torch.cat([base_tokens, style_tok_u.unsqueeze(1), item_style_tok.unsqueeze(1)], dim=1)

            elif diff_model.parallel["set_aggr"] == "item_u":
                uid = tgt_uid.long()  # (B,)
                style_src = style_src.to(start1.device)
                style_u = style_src[uid][:, :2]  # (B, F)

                if diff_model.parallel["bias_mapping"] == "user":
                    style_u = diff_model.user_style_mapper(style_u)
                    style_u = style_u.detach()

                style_tok = diff_model.style_encoder(style_u)  # (B, D)
                style_tok = diff_model.style_ln(style_tok)  # (B, D)
                style_tok_u = diff_model.style_scale * style_tok  # (B, D)

                tokens = torch.cat([base_tokens, style_tok_u.unsqueeze(1)], dim=1)

            elif diff_model.parallel["set_aggr"] == "item":
                tokens = base_tokens

            out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
            final_output = out[:, 0, :]  # (B, D)
            y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측

            return y_pred

    def _fetch_vbge_user_embedding(self, diff_model, tgt_uid, use_target=False):

        attr = "smooth_user_emb_tgt" if use_target else "smooth_user_emb_src"
        vbge_cache = getattr(diff_model, attr, None)
        if vbge_cache is None:
            return None
        indices = tgt_uid.long()
        return vbge_cache[indices]

    def _fetch_vbge_item_embedding(self, diff_model, tgt_iid):
        if not hasattr(diff_model, "smooth_item_emb"):
            return None
        indices = tgt_iid.long()
        return diff_model.smooth_item_emb[indices].squeeze()
