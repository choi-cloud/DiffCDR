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


class uidEmbedding(torch.nn.Module):

    def __init__(self, uid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.linear_1 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_2 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_3 = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x)
        uid_emb = self.linear_1(uid_emb)
        uid_emb = F.relu(uid_emb)
        uid_emb = self.linear_2(uid_emb)
        uid_emb = F.relu(uid_emb)
        uid_emb = self.linear_3(uid_emb)
        return F.relu(uid_emb)
    
class iidEmbedding(torch.nn.Module):

    def __init__(self, iid_all, emb_dim):
        super().__init__() 
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)  
        self.linear_1 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_2 = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_3 = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        iid_emb = self.iid_embedding(x)
        iid_emb = self.linear_1(iid_emb)
        iid_emb = F.relu(iid_emb)
        iid_emb = self.linear_2(iid_emb)
        iid_emb = F.relu(iid_emb)
        iid_emb = self.linear_3(iid_emb)
        return F.relu(iid_emb)
    

class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, meta_dim_0):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)

        self.user_embedding = uidEmbedding(uid_all, emb_dim)
        self.item_embedding = iidEmbedding(iid_all, emb_dim)
        self.num_embeddings = uid_all
        self.num_embeddings_i = iid_all

        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def encode_all_users(self, device='cuda'):
        all_uid = torch.arange(self.num_embeddings, device=device)
        user_feat = self.user_embedding(all_uid)   # ⭐ forward 통과
        return user_feat   
    
    def encode_all_item(self, device='cuda'):
        all_iid = torch.arange(self.num_embeddings_i, device=device)
        item_feat = self.item_embedding(all_iid.unsqueeze(1))   # ⭐ forward 통과
        return item_feat

    def compute_user_graph_embeddings(self, is_train=False, device='cuda'):
        if is_train: 
            graph_data = self.graph_shared_train
        else:
            graph_data = self.graph_shared_test
        # graph_data = self.graph_src
        # 단순 2홉 aggr
        # simple 2-hop aggregation (user -> items -> users) excluding 1-hop self contribution
        uv_adj = graph_data["uv_adj"].to(device)
        vu_adj = graph_data["vu_adj"].to(device)

        user_feat = self.encode_all_users().to(device)

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

        return user_emb
    
    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False, item_cond=False, style_src=None):
        if stage == "train_src":
            # emb = self.src_model.forward(x)
            # x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            # return x
            user_emb = self.user_embedding(x[:, 0])
            item_emb = self.item_embedding(x[:, 1])
            x = torch.sum(user_emb * item_emb, dim=1)
            return x 


        elif stage in ["train_tgt", "test_tgt"]:
            # emb = self.tgt_model.forward(x)
            # x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            # return x
            user_emb = self.user_embedding(x[:, 0])
            item_emb = self.item_embedding(x[:, 1])
            x = torch.sum(user_emb * item_emb, dim=1)
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

            # Diff1: MF 유저 임베딩, Diff2: Aggr 유저 임베딩
            # src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF
            uid_emb1 = self.user_embedding(tgt_uid.unsqueeze(1)).squeeze()
            uid_emb2 = self.compute_user_graph_embeddings(is_train=True)[tgt_uid]  # Aggr

            cond_emb1 = uid_emb1
            cond_emb2 = uid_emb2

            if item_cond==True:
                # attention 
                pos_items = diff_model.user_src_items_pad[tgt_uid]     # (B, L)
                pos_mask  = diff_model.user_src_items_mask[tgt_uid]    # (B, L)

                # emb1 = diff_model.ln_m(uid_emb1)
                # emb2 = diff_model.ln_g(uid_emb2)

                # iid_emb1 = self.encode_all_item()[tgt_uid]  
                # print(iid_emb1.shape, uid_emb1.shape, emb1.shape)
                # user_z_src_mf = diff_model.pool_user_z_attention_batch(pos_items, pos_mask, emb1, is_mf=True)
                # user_z_src_aggr = diff_model.pool_user_z_attention_batch(pos_items, pos_mask, emb2, is_mf=False)

                # cond_emb1 = user_z_src_mf
                # cond_emb2 = user_z_src_aggr
                # potential_items = diff_model.potential_items_pad[tgt_uid]
                # potential_mask  = diff_model.potential_items_mask[tgt_uid]

                user_z_src1 = diff_model.pool_user_z_attention_batch(
                    pos_items=pos_items,
                    pos_mask=pos_mask,
                )

                # user_z_src2 = diff_model.pool_user_z_attention_batch(
                #     pos_items=potential_items,
                #     pos_mask=potential_mask,
                # )
                user_z_src2 = user_z_src1

                cond_emb1 = user_z_src1
                cond_emb2 = user_z_src2
                         
            iid_emb = self.item_embedding(iid_input.unsqueeze(1)).squeeze()

            # ! mf 임베딩과 aggr 임베딩 양자화
            quantized1, all_level_vectors1, rq_loss1 = diff_model.rq_mf(cond_emb1)
            quantized2, all_level_vectors2, rq_loss2 = diff_model.rq_aggr(cond_emb2)

            # is_task=False: 노이즈 예측 , is_task=True: ALS + task 로스
            loss = Diff.diffusion_loss_fn_parallel(
                diff_model,
                uid_emb1,
                uid_emb2,
                # ! diff_loss 계산 시에는 양자화하지 않은 기존 소스 임베딩을 컨디션으로 이용
                cond_emb1,
                cond_emb2,
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
                iid_input = iid_input,
                src_item_m=cond_emb1,
                src_item_g=cond_emb2,
            )

            total_loss = loss + diff_model.rqvae["alpha_rq"] * (rq_loss1 + rq_loss2)
            return total_loss

        elif stage == "test_diff_parallel":  # DiffParallel - test

            tgt_uid, iid_input, _ = x

            uid_emb1 = self.user_embedding(tgt_uid.unsqueeze(1)).squeeze()
            uid_emb2 = self.compute_user_graph_embeddings(is_train=False)[tgt_uid]  # Aggr

            cond_emb1 = uid_emb1
            cond_emb2 = uid_emb2

            iid_emb = self.item_embedding(iid_input.unsqueeze(1)).squeeze()

            if item_cond==True:
                # attention 
                pos_items = diff_model.user_src_items_pad[tgt_uid]     # (B, L)
                pos_mask  = diff_model.user_src_items_mask[tgt_uid]    # (B, L)

                # emb1 = diff_model.ln_m(uid_emb1)
                # emb2 = diff_model.ln_g(uid_emb2)

                # # iid_emb = self.encode_all_item()[tgt_uid]  

                # user_z_src_mf = diff_model.pool_user_z_attention_batch(pos_items, pos_mask, emb1, is_mf=True)
                # user_z_src_aggr = diff_model.pool_user_z_attention_batch(pos_items, pos_mask, emb2, is_mf=False)

                # cond_emb1 = user_z_src_mf
                # cond_emb2 = user_z_src_aggr
                # potential_items = diff_model.potential_items_pad[tgt_uid]
                # potential_mask  = diff_model.potential_items_mask[tgt_uid]

                user_z_src1 = diff_model.pool_user_z_attention_batch(
                    pos_items=pos_items,
                    pos_mask=pos_mask,
                )

                # user_z_src2 = diff_model.pool_user_z_attention_batch(
                #     pos_items=potential_items,
                #     pos_mask=potential_mask,
                # )
                user_z_src2 = user_z_src1

                cond_emb1 = user_z_src1
                cond_emb2 = user_z_src2

            # ! mf 임베딩과 aggr 임베딩 양자화
            quantized, all_level_vectors1, _ = diff_model.rq_mf(cond_emb1)  # [L, B, D]
            quantized, all_level_vectors2, _ = diff_model.rq_aggr(cond_emb2)  # [L, B, D]

            ### [TEST] 1️. Diff1, Diff2 noised x_0 설정에 따라 denoising
            trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(diff_model, uid_emb1, all_level_vectors1, iid_emb, device, diff_id=0)
            trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(diff_model, uid_emb2, all_level_vectors2, iid_emb, device, diff_id=1)

            ### [TEST] 2. Diff1, Diff2 결과 aggregation
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

            if diff_model.parallel["set_aggr"] in ["item_cls"]:
                tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u, item_style_tok], dim=1)
            elif diff_model.parallel["set_aggr"] == "item_cls1": 
                item_z = diff_model.item_Z_tgt[iid_input]   
                item_z = diff_model.ln_z(item_z.squeeze(1))
                tokens = torch.stack([iid_emb, final_output_m, final_output_g, style_tok_u, item_style_tok, item_z], dim=1)
            elif diff_model.parallel["set_aggr"] == "item_cls2": 
                item_z = diff_model.item_Z_tgt[iid_input]   
                item_z = diff_model.ln_z(item_z.squeeze(1))
                tokens = torch.stack([final_output_m, final_output_g, style_tok_u, item_style_tok, item_z], dim=1)
            
            out = diff_model.attn_layer(tokens, query=iid_emb.unsqueeze(1))
            final_output = out[:, 0, :]
            
            y_pred = torch.sum(final_output * iid_emb, dim=1)  # user, item emb 내적해서 예측
            mu_t = diff_model.tgt_global_bias
            y_pred = y_pred + mu_t

            return y_pred
