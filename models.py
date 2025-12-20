import torch
import torch.nn.functional as F

import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR


class LookupEmbedding(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, n_layers=2):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all, emb_dim)
        self.n_layers = n_layers

    def aggregate(self, graph_data):
        """
        매 호출마다 LightGCN propagation을 수행해서
        전체 유저/아이템 임베딩을 만들어 반환한다.

        graph_data (dict 권장):
          - 'uv_adj': sparse [num_users, num_items]
          - 'vu_adj': sparse [num_items, num_users]
          - 'n_layers' or 'K': optional (없으면 self.n_layers 사용)
        """
        uv_adj = graph_data["uv_adj"]
        vu_adj = graph_data["vu_adj"]
        K = graph_data.get("n_layers", graph_data.get("K", self.n_layers))

        device = self.uid_embedding.weight.device
        uv_adj = uv_adj.to(device)
        vu_adj = vu_adj.to(device)

        # layer 0
        u = self.uid_embedding.weight  # [U, d]
        i = self.iid_embedding.weight  # [I, d]  (pad 포함이면 그래프와 size 맞는지 확인)

        u_layers = [u]
        i_layers = [i]

        # K-hop propagation
        for _ in range(K):
            i = torch.sparse.mm(vu_adj, u)  # [I, d]
            u = torch.sparse.mm(uv_adj, i)  # [U, d]
            u_layers.append(u)
            i_layers.append(i)

        # LightGCN readout: mean over layers
        u_all = torch.stack(u_layers, dim=0).mean(dim=0)  # [U, d]
        i_all = torch.stack(i_layers, dim=0).mean(dim=0)  # [I, d]
        return u_all, i_all

    def forward(self, x, graph_data=None):
        """
        x: [B, 2] where x[:,0]=uid, x[:,1]=iid
        graph_data가 있으면 매 배치마다 aggregate 수행 후 인덱싱.
        """
        if graph_data is None:
            uid_emb = self.uid_embedding(x[:, 0]).unsqueeze(1)  # [B,1,d]
            iid_emb = self.iid_embedding(x[:, 1]).unsqueeze(1)  # [B,1,d]
            return torch.cat([uid_emb, iid_emb], dim=1)

        u_all, i_all = self.aggregate(graph_data)  # 매 배치마다 전체 계산

        uid = x[:, 0]
        iid = x[:, 1]

        uid_emb = u_all.index_select(0, uid).unsqueeze(1)  # [B,1,d]
        iid_emb = i_all.index_select(0, iid).unsqueeze(1)  # [B,1,d]
        return torch.cat([uid_emb, iid_emb], dim=1)


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

        self.src_model_graph = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model_graph = LookupEmbedding(uid_all, iid_all, emb_dim)

        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage, device, diff_model=None, ss_model=None, la_model=None, is_task=False, graph_data=None):
        if stage == "train_src":
            if graph_data == None:
                emb = self.src_model.forward(x)
                x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                return x
            else:
                emb = self.src_model_graph.forward(x, graph_data)
                x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                return x

        elif stage in ["train_tgt", "test_tgt"]:
            if graph_data == None:
                emb = self.tgt_model.forward(x)
                x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                return x
            else:
                emb = self.tgt_model_graph.forward(x, graph_data)
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

            tgt_users, tgt_items = self.tgt_model_graph.aggregate(graph_data=graph_data["train_tgt"])
            src_users, src_items = self.src_model_graph.aggregate(graph_data=graph_data["train_src"])

            tgt_emb = tgt_users.index_select(0, tgt_uid)
            cond_emb = src_users.index_select(0, tgt_uid)
            iid_emb = tgt_items.index_select(0, iid_input.squeeze(1))

            # tgt_emb = self.tgt_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            # cond_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()

            # iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            loss = Diff.diffusion_loss_fn(diff_model, tgt_emb, cond_emb, iid_emb, y_input, device, is_task)
            return loss

        elif stage == "test_diff":

            tgt_uid, iid_input, _ = x

            tgt_users, tgt_items = self.tgt_model_graph.aggregate(graph_data=graph_data["train_tgt"])
            src_users, src_items = self.src_model_graph.aggregate(graph_data=graph_data["train_src"])

            cond_emb = src_users.index_select(0, tgt_uid)
            iid_emb = tgt_items.index_select(0, iid_input.squeeze(1))

            # cond_emb = self.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()
            # iid_emb = self.tgt_model.iid_embedding(iid_input.unsqueeze(1)).squeeze()

            trans_emb, iid_emb_out = Diff.p_sample_loop(diff_model, cond_emb, iid_emb, device)

            x = torch.sum(trans_emb * iid_emb_out, dim=1)
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
