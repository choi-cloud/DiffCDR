import torch
import torch.nn.functional as F
import torch.nn as nn

import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR
from rqvae import ResidualQuantizer
from utils import AttentionLayer

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

        self.graph_emb_cache = {} # 🔥 Cache for graph embeddings

    def clear_graph_cache(self):
        self.graph_emb_cache = {}

    @torch.no_grad()
    def build_user_prototype_cache(
        self,
        device,
        top_p=0.01,  # 상위 p%
        bottom_p=0.005,  # 하위 p%
        user_batch=256,
    ):
        """
        Memory-safe prototype cache builder
        """
        uid_emb_all = self.src_model.uid_embedding.weight.detach().to(device)
        iid_emb = self.src_model.iid_embedding.weight.detach().to(device)

        num_users = uid_emb_all.size(0)
        num_items = iid_emb.size(0)
        d = uid_emb_all.size(1)

        # percentage → k (NO clamp)
        topk = int(num_items * top_p)
        bottomk = int(num_items * bottom_p)

        # prototype cache
        top_proto = torch.zeros((num_users, d), device=device)
        bot_proto = torch.zeros((num_users, d), device=device)

        for start in range(0, num_users, user_batch):
            end = min(start + user_batch, num_users)

            u_emb = uid_emb_all[start:end]  # [B, d]

            # score matrix for this batch only
            scores = torch.matmul(u_emb, iid_emb.t())  # [B, I]

            # ---------- TOP PROTOTYPE ----------
            if topk > 0:
                top_idx = torch.topk(scores, k=topk, dim=1).indices  # [B, topk]
                top_proto[start:end] = iid_emb[top_idx].mean(dim=1)
            else:
                # clean skip
                top_proto[start:end] = torch.zeros_like(u_emb)

            # ---------- BOTTOM PROTOTYPE (REPULSION) ----------
            if bottomk > 0:
                bot_idx = torch.topk(scores, k=bottomk, dim=1, largest=False).indices  # [B, bottomk]

                bottom_mean = iid_emb[bot_idx].mean(dim=1)  # [B, d]
                # bot_proto[start:end] = F.normalize(
                #     u_emb - bottom_mean, dim=1
                # )
                bot_proto[start:end] = iid_emb[bot_idx].mean(dim=1)
            else:
                # clean skip
                bot_proto[start:end] = torch.zeros_like(u_emb)

            # very important to free memory
            del scores

            if start % (user_batch * 20) == 0:
                torch.cuda.empty_cache()

        # store cache
        self.user_proto_cache = {"top": top_proto.detach(), "bottom": bot_proto.detach()}

    @torch.no_grad()
    def get_top_bottom_item_prototypes(self, uid):
        top = self.user_proto_cache["top"][uid]
        bottom = self.user_proto_cache["bottom"][uid]
        return top, bottom

    def compute_user_graph_embeddings(self, graph_data, use_target=False, device='cuda'):
        if graph_data is None:
            return None, None
        
        # Check cache first
        cache_key = "tgt" if use_target else "src"
        if cache_key in self.graph_emb_cache:
            return self.graph_emb_cache[cache_key]

        # 단순 2홉 aggr
        # simple 2-hop aggregation (user -> items -> users) excluding 1-hop self contribution
        uv_adj = graph_data["uv_adj"].to(device)
        vu_adj = graph_data["vu_adj"].to(device)
        if use_target:
            user_feat = self.tgt_model.uid_embedding.weight#.detach().to(self.device)
        else:
            user_feat = self.src_model.uid_embedding.weight#.detach().to(self.device)
        # with torch.no_grad():
        # 1-hop: items aggregate from users
        item_msg = torch.sparse.mm(vu_adj, user_feat)  # [num_items, d]

        # 2-hop: users aggregate from items
        user_2hop = torch.sparse.mm(uv_adj, item_msg)  # [num_users, d]

        # remove self 1-hop contribution (user -> item -> user)
        user_deg = torch.sparse.sum(uv_adj, dim=1).to_dense().unsqueeze(1)
        user_2hop = user_2hop - user_deg * user_feat  # self-removal

        # count real 2-hop neighbors: user -> item -> other_users
        item_deg = torch.sparse.sum(vu_adj, dim=1).to_dense()
        item_other = torch.relu(item_deg - 1)  # max(deg-1, 0)
        two_hop_counts = torch.sparse.mm(uv_adj, item_other[:, None]).to_dense()

        # normalization (avoid division by zero)
        norm = torch.where(two_hop_counts == 0, torch.ones_like(two_hop_counts), two_hop_counts)

        # final 2-hop embedding
        user_emb = user_2hop / norm

        # fallback: if no 2-hop neighbors, keep original embedding
        zero_mask = two_hop_counts.squeeze(1) == 0
        user_emb[zero_mask] = user_feat[zero_mask]
        
        # Store in cache (Detached to avoid graph memory explosion)
        self.graph_emb_cache[cache_key] = user_emb.detach()

        return user_emb

    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False, item_cond=False, style_src=None):
        if stage == "train_src":
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x

        elif stage in ["train_tgt", "test_tgt"]:
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
            # tgt_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=True)  # Aggr
            tgt_emb2 = self.compute_user_graph_embeddings(self.graph_tgt, use_target=True, device=device)[tgt_uid]
            
            # Diff1: MF 유저 임베딩, Diff2: Aggr 유저 임베딩
            src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF
            # src_uid_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)  # Aggr
            src_uid_emb2 = self.compute_user_graph_embeddings(self.graph_src, use_target=False, device=device)[tgt_uid]
            
            if item_cond == True:
                top_proto, bottom_proto = self.get_top_bottom_item_prototypes(tgt_uid)
                cond_emb1 = top_proto
                cond_emb2 = bottom_proto

            else:
                cond_emb1 = src_uid_emb1
                cond_emb2 = src_uid_emb2

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            # ! mf 임베딩과 aggr 임베딩 양자화
            if diff_model.rqvae["RQVAE"] == True:
                quantized1, all_level_vectors1, rq_loss1 = diff_model.rq_mf(cond_emb1)  # [L, B, D]
                quantized2, all_level_vectors2, rq_loss2 = diff_model.rq_aggr(cond_emb2)  # [L, B, D]
            else:
                all_level_vectors1 = cond_emb1
                all_level_vectors2 = cond_emb2

            # is_task=False: 노이즈 예측 , is_task=True: ALS + task 로스
            loss = Diff.diffusion_loss_fn_parallel(
                diff_model,
                tgt_emb1,
                tgt_emb2,
                # ! diff_loss 계산 시에는 양자화하지 않은 기존 소스 임베딩을 컨디션으로 이용
                src_uid_emb1,   # 시작점
                src_uid_emb2,
                iid_emb,
                y_input,
                device,
                is_task,
                # ! is_taks가 True일 때만 양자화된 컨디션을 시간축에 따라 이용
                q_embs1=all_level_vectors1,
                q_embs2=all_level_vectors2,
                style_src=style_src,
                uid=tgt_uid,
                iid=iid_input,
                Q_emb1=quantized1,
                Q_emb2=quantized2,                
            )

            if diff_model.rqvae["RQVAE"] == True:
                total_loss = loss + diff_model.rqvae["alpha_rq"] * (rq_loss1 + rq_loss2)
            else:
                total_loss = loss

            return total_loss

        elif stage == "test_diff_parallel":  # DiffParallel - test

            tgt_uid, iid_input, _ = x

            src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF
            # src_uid_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)  # Aggr
            src_uid_emb2 = self.compute_user_graph_embeddings(self.graph_src, use_target=False)[tgt_uid]

            cond_emb1 = src_uid_emb1
            cond_emb2 = src_uid_emb2
            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            # ! mf 임베딩과 aggr 임베딩 양자화
            if diff_model.rqvae["RQVAE"] == True:
                quantized, all_level_vectors1, _ = diff_model.rq_mf(cond_emb1)  # [L, B, D]
                quantized, all_level_vectors2, _ = diff_model.rq_aggr(cond_emb2)  # [L, B, D]
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(diff_model, src_uid_emb1, all_level_vectors1, iid_emb, device, diff_id=0)
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(diff_model, src_uid_emb2, all_level_vectors2, iid_emb, device, diff_id=1)
            else:
                all_level_vectors1 = cond_emb1
                all_level_vectors2 = cond_emb2
                trans_emb_m, iid_emb = Diff.p_sample_loop(diff_model, src_uid_emb1, all_level_vectors1, iid_emb, device, diff_id=0)
                trans_emb_g, iid_emb = Diff.p_sample_loop(diff_model, src_uid_emb2, all_level_vectors2, iid_emb, device, diff_id=1)



            ### [TEST] 2. Diff1, Diff2 결과 aggregation
            if diff_model.parallel["set_aggr"] == "attn":
                # ! 어텐션으로 최종 임베딩 종합
                trans_emb = diff_model.attn_layer(torch.cat([trans_emb_m, trans_emb_g], dim=1))

            elif diff_model.parallel["set_aggr"] == "item_attn":
                # 아이템을 쿼리로 사용
                trans_emb = diff_model.attn_layer(torch.cat([trans_emb_m, trans_emb_g], dim=1), query=torch.cat([iid_emb, iid_emb], dim=1))

            elif diff_model.parallel["set_aggr"] == "item_diu":
                # 아이템 포함해서 self attn -> 아이템 출력만 사용
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                uid = tgt_uid.long()

                # style token
                style_src = style_src.to(trans_emb_g.device)
                style_u = style_src[uid]  # (B, F)
                style_tok = diff_model.style_encoder(style_u)  # (B, D)
                style_tok = diff_model.style_ln(style_tok)  # (B, D)
                style_tok_u = diff_model.style_scale * style_tok  # (B, D)

                style_tgt_item = diff_model.style_tgt_item.to(trans_emb_g.device)  # [I_total, F_item]
                style_i = style_tgt_item[iid_input.squeeze(1)]  # (B, F_item)
                item_style_tok = diff_model.item_style_encoder(style_i)  # (B, D)
                item_style_tok = diff_model.item_style_ln(item_style_tok)  # (B, D)
                item_style_tok = diff_model.item_style_scale * item_style_tok  # (B, D)

                tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u, item_style_tok], dim=1)
                out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))
                final_output = out[:, 0, :]

                y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측
                mu_t = diff_model.tgt_global_bias
                y_pred = y_pred + mu_t
                    
            elif diff_model.parallel["set_aggr"] == "item_d":
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                tokens = torch.stack([iid_emb, final_output_m, final_output_g], dim=1)
                out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
                final_output = out[:, 0, :]  # (B, D)

                y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측
                mu_t = diff_model.tgt_global_bias
                y_pred = y_pred + mu_t

            elif diff_model.parallel["set_aggr"] == "item_di":
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                style_tgt_item = diff_model.style_tgt_item.to(trans_emb_m.device)  # [I_total, F_item]
                style_i = style_tgt_item[iid_input.squeeze(1)]  # (B, F_item)
                item_style_tok = diff_model.item_style_encoder(style_i)  # (B, D)
                item_style_tok = diff_model.item_style_ln(item_style_tok)  # (B, D)
                item_style_tok = diff_model.item_style_scale * item_style_tok  # (B, D)

                tokens = torch.stack([iid_emb, final_output_m, final_output_g, item_style_tok], dim=1)
                out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
                final_output = out[:, 0, :]  # (B, D)

                y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측
                mu_t = diff_model.tgt_global_bias
                y_pred = y_pred + mu_t

            elif diff_model.parallel["set_aggr"] == "item_du":
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                uid = tgt_uid.long()  # (B,)

                style_src = style_src.to(trans_emb_m.device)
                style_u = style_src[uid]  # (B, F)
                style_tok = diff_model.style_encoder(style_u)  # (B, D)
                style_tok = diff_model.style_ln(style_tok)  # (B, D)
                style_tok_u = diff_model.style_scale * style_tok  # (B, D)

                tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u], dim=1)
                out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
                final_output = out[:, 0, :]  # (B, D)

                y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측
                mu_t = diff_model.tgt_global_bias
                y_pred = y_pred + mu_t


            elif diff_model.parallel["set_aggr"] == "item_i":
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                uid = tgt_uid.long()  # (B,)

                style_tgt_item = diff_model.style_tgt_item.to(trans_emb_m.device)  # [I_total, F_item]
                style_i = style_tgt_item[iid_input.squeeze(1)]  # (B, F_item)
                item_style_tok = diff_model.item_style_encoder(style_i)  # (B, D)
                item_style_tok = diff_model.item_style_ln(item_style_tok)  # (B, D)
                item_style_tok = diff_model.item_style_scale * item_style_tok  # (B, D)

                tokens = torch.stack([iid_emb, final_output_m, final_output_g, item_style_tok], dim=1)
                out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
                final_output = out[:, 0, :]  # (B, D)

                y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측


            elif diff_model.parallel["set_aggr"] == "item_iu":
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                uid = tgt_uid.long()  # (B,)

                style_src = style_src.to(trans_emb_m.device)
                style_u = style_src[uid]  # (B, F)
                style_tok = diff_model.style_encoder(style_u)  # (B, D)
                style_tok = diff_model.style_ln(style_tok)  # (B, D)
                style_tok_u = diff_model.style_scale * style_tok  # (B, D)

                style_tgt_item = diff_model.style_tgt_item.to(trans_emb_m.device)  # [I_total, F_item]
                style_i = style_tgt_item[iid_input.squeeze(1)]  # (B, F_item)
                item_style_tok = diff_model.item_style_encoder(style_i)  # (B, D)
                item_style_tok = diff_model.item_style_ln(item_style_tok)  # (B, D)
                item_style_tok = diff_model.item_style_scale * item_style_tok  # (B, D)

                tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u, item_style_tok], dim=1)
                out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))  # (B, 1, D)
                final_output = out[:, 0, :]  # (B, D)

                y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측


            elif diff_model.parallel["set_aggr"] == "item_u":
                iid_emb = diff_model.ln_iid(iid_emb)
                final_output_m = diff_model.ln_m(diff_model.linear_m(trans_emb_m))
                final_output_g = diff_model.ln_g(diff_model.linear_g(trans_emb_g))

                uid = tgt_uid.long()  # (B,)

                style_src = style_src.to(trans_emb_m.device)
                style_u = style_src[uid]  # (B, F)
                style_tok = diff_model.style_encoder(style_u)  # (B, D)
                style_tok = diff_model.style_ln(style_tok)  # (B, D)
                style_tok_u = diff_model.style_scale * style_tok  # (B, D)

                tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u], dim=1)
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
