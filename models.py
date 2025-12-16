import torch
import torch.nn.functional as F

import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR


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
            cond_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            cond_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)

            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            loss = Diff.diffusion_loss_fn_parallel(diff_model, tgt_emb1, tgt_emb2, cond_emb1, cond_emb2, iid_emb, y_input, device, is_task)
            return loss  # is_task=False: 노이즈 예측 , is_task=True: ALS, pred 로스

        elif stage == "test_diff_parallel":  # DiffParallel - test

            tgt_uid, iid_input, _ = x

            cond_emb1 = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            cond_emb2 = self._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False)

            # TODO item 임베딩은 MF 임베딩을 공유?
            iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            if diff_model.parallel["set_init"] == 0:  # x_0 둘다 MF ui로
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb1, cond_emb1, iid_emb, device, diff_id=0
                )  # 디노이징 된 user emb_m / item emb
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb1, cond_emb2, iid_emb, device, diff_id=1
                )  # 디노이징 된 user emb_g / item emb

            elif diff_model.parallel["set_init"] == 1:  # 각각 MF, Aggr
                trans_emb_m, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb1, cond_emb1, iid_emb, device, diff_id=0
                )  # 디노이징 된 user emb_m / item emb
                trans_emb_g, iid_emb = Diff.p_sample_loop_parallel(
                    diff_model, cond_emb2, cond_emb2, iid_emb, device, diff_id=1
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
