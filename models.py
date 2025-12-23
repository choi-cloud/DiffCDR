import torch
import torch.nn.functional as F

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

        # ! mf 소스 임베딩과 aggr 소스 임베딩 각각을 양자화하기 위한 모듈
        self.rq_mf = ResidualQuantizer(code_dim=emb_dim, num_levels=4, codebook_size=256)
        self.rq_aggr = ResidualQuantizer(code_dim=emb_dim, num_levels=4, codebook_size=256)

    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False):
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

            tgt_emb1 = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()  # MF feature
            tgt_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=True)

            # 조건 1: MF 기반 유저 임베딩, 조건 2: VBGE 기반 유저 임베딩
            src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            src_uid_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()
            # iid_emb = self._fetch_vbge_item_embedding(diff_model, iid_input)
            # print(iid_emb)

            # ! mf 임베딩과 aggr 임베딩 양자화
            if diff_model.parallel["set_aggr"] == "pop_attn":
                int_item_aggr = diff_model.int_item_aggr[tgt_uid.unsqueeze(1)].squeeze()
                conf_item_aggr = diff_model.conf_item_aggr[tgt_uid.unsqueeze(1)].squeeze()
                quantized, all_level_vectors1, rq_loss1 = self.rq_mf(int_item_aggr)  # [L, B, D]
                quantized, all_level_vectors2, rq_loss2 = self.rq_aggr(conf_item_aggr)  # [L, B, D]
            else: 
                quantized, all_level_vectors1, rq_loss1 = self.rq_mf(src_uid_emb1)
                quantized, all_level_vectors2, rq_loss2 = self.rq_aggr(src_uid_emb2)

            conf_weight = diff_model.item_popularity[iid_input]
            # int_weight = 1 - conf_weight

            loss = Diff.diffusion_loss_fn_parallel(
                diff_model,
                tgt_emb1,
                tgt_emb2,
                # ! diff_loss 계산 시에는 양자화하지 않은 기존 소스 임베딩을 컨디션으로 이용
                src_uid_emb1,
                src_uid_emb2,
                iid_emb,
                y_input,
                device,
                is_task,
                # ! is_taks가 True일 때만 양자화된 컨디션을 시간축에 따라 이용
                q_embs1=all_level_vectors1,
                q_embs2=all_level_vectors2,
                pop=conf_weight,
            )

            alpha_rq = 1e-2
            total_loss = loss + alpha_rq * (rq_loss1 + rq_loss2)
            return total_loss  # is_task=False: 노이즈 예측 , is_task=True: ALS, pred 로스

        elif stage == "test_diff_parallel":  # DiffParallel - test

            tgt_uid, iid_input, _ = x

            src_uid_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            src_uid_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)

            # ! mf 임베딩과 aggr 임베딩 양자화
            if diff_model.parallel["set_aggr"] == "pop_attn":
                int_item_aggr = diff_model.int_item_aggr[tgt_uid.unsqueeze(1)].squeeze()
                conf_item_aggr = diff_model.conf_item_aggr[tgt_uid.unsqueeze(1)].squeeze()
                quantized, all_level_vectors1, _ = self.rq_mf(int_item_aggr)  # [L, B, D]
                quantized, all_level_vectors2, _ = self.rq_aggr(conf_item_aggr)  # [L, B, D]
            else: 
                quantized, all_level_vectors1, _ = self.rq_mf(src_uid_emb1)  # [L, B, D]
                quantized, all_level_vectors2, _ = self.rq_aggr(src_uid_emb2)  # [L, B, D]

            # TODO item 임베딩은 MF 임베딩을 공유?
            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()
            # iid_emb = self._fetch_vbge_item_embedding(diff_model, iid_input)

            if diff_model.parallel["set_init"] == 0:  # x_0 둘다 MF ui로
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb1, cond_emb1, iid_emb, device, diff_id=0
                )  # 디노이징 된 user emb_m / item emb
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb1, cond_emb2, iid_emb, device, diff_id=1
                )  # 디노이징 된 user emb_g / item emb

            elif diff_model.parallel["set_init"] == 1 or diff_model.parallel["set_init"] == 4:  # 각각 MF, Aggr
                # ! 각각 MF, Aggr인 파트만 수정
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, all_level_vectors1, all_level_vectors1, iid_emb, device, diff_id=0
                )  # 디노이징 된 user emb_m / item emb
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, all_level_vectors2, all_level_vectors2, iid_emb, device, diff_id=1
                )  # 디노이징 된 user emb_g / item emb

            elif diff_model.parallel["set_init"] == 2:  # x_0 둘다 MF + Aggr로
                start = (cond_emb1 + cond_emb2) / 2
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, start, cond_emb1, iid_emb, device, diff_id=0
                )  # 디노이징 된 user emb_m / item emb
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, start, cond_emb2, iid_emb, device, diff_id=1
                )  # 디노이징 된 user emb_g / item emb

            elif diff_model.parallel["set_init"] == 3:  # x_0 둘다 Aggr로
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb2, cond_emb2, iid_emb, device, diff_id=0
                )  # 디노이징 된 user emb_m / item emb
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb2, cond_emb2, iid_emb, device, diff_id=1
                )  # 디노이징 된 user emb_g / item emb

            # final output = x'_c + x'_g
            # TODO 두 디퓨전 모델의 아웃풋 결합 방법 ?
            if diff_model.parallel["set_aggr"] == "avg":
                trans_emb = (trans_emb_m + trans_emb_g) / 2
            elif diff_model.parallel["set_aggr"] == "concat":
                trans_emb = torch.cat([trans_emb_m, trans_emb_g], dim=1)
            elif diff_model.parallel["set_aggr"] == "aggonly":
                trans_emb = trans_emb_g
            elif diff_model.parallel["set_aggr"] == "attn":
                # ! 어텐션으로 최종 임베딩 종합 
                trans_emb = diff_model.attn_layer(torch.cat([trans_emb_m, trans_emb_g], dim=1))
            elif diff_model.parallel["set_aggr"] == "pop": 
                # m -> int, g -> conf 
                conf_weight = diff_model.item_popularity[iid_input]
                int_weight = 1 - conf_weight
                trans_emb = int_weight * trans_emb_m + conf_weight * trans_emb_g
            elif diff_model.parallel["set_aggr"] == "pop_attn":
                # conf_weight = diff_model.item_popularity[iid_input]
                # int_weight = 1 - conf_weight
                
                # trans_emb_m = int_weight * trans_emb_m 
                # trans_emb_g = conf_weight * trans_emb_g
                trans_emb = diff_model.attn_layer(torch.cat([trans_emb_m, trans_emb_g], dim=1)) 

            if diff_model.parallel["set_proj"] == 1:
                trans_emb = diff_model.get_al_emb(trans_emb).to(device)

            x = torch.sum(trans_emb * iid_emb, dim=1)  # user, item emb 곱해서 예측

            return x

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
