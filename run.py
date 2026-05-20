import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
from models import MFBasedModel
import DiffModel as Diff
import sscdr_model as SSCDR
import lacdr_model as LACDR

import pickle
import json
import os

from utils import *
import ast


class Run:
    def __init__(self, config):
        self.use_cuda = config["use_cuda"]
        self.base_model = config["base_model"]
        self.root = config["root"]
        self.ratio = config["ratio"]
        self.task = config["task"]
        self.src = config["src_tgt_pairs"][self.task]["src"]
        self.tgt = config["src_tgt_pairs"][self.task]["tgt"]
        self.uid_all = config["src_tgt_pairs"][self.task]["uid"]
        self.iid_all = config["src_tgt_pairs"][self.task]["iid"]
        self.batchsize_src = config["src_tgt_pairs"][self.task]["batchsize_src"]
        self.batchsize_tgt = config["src_tgt_pairs"][self.task]["batchsize_tgt"]
        self.batchsize_meta = config["src_tgt_pairs"][self.task]["batchsize_meta"]
        self.batchsize_map = config["src_tgt_pairs"][self.task]["batchsize_map"]
        self.batchsize_ss = config["src_tgt_pairs"][self.task]["batchsize_ss"]
        self.batchsize_la = config["src_tgt_pairs"][self.task]["batchsize_la"]
        self.batchsize_diff = config["src_tgt_pairs"][self.task]["batchsize_diff"]

        self.batchsize_test = config["src_tgt_pairs"][self.task]["batchsize_test"]
        self.batchsize_diff_test = config["src_tgt_pairs"][self.task]["batchsize_diff_test"]

        self.batchsize_aug = self.batchsize_src

        self.epoch = config["epoch"]
        self.emb_dim = config["emb_dim"]
        self.meta_dim = config["meta_dim"]
        self.lr = config["lr"]
        self.la_lr = config["la_lr"]

        self.wd = config["wd"]

        self.ratio = [float(self.ratio.split(",")[0][1:]), float(self.ratio.split(",")[1][:-1])]

        self.input_root = (
            self.root + "ready/_" + str(int(self.ratio[0] * 10)) + "_" + str(int(self.ratio[1] * 10)) + "/tgt_" + self.tgt + "_src_" + self.src
        )
        self.src_path = self.input_root + "/train_src.csv"
        self.tgt_path = self.input_root + "/train_tgt.csv"

        self.meta_path = self.input_root + "/train_meta.csv"
        self.test_path = self.input_root + "/test.csv"

        self.warm_tgt_train_path = self.input_root + "/warm_start_tgt_train.csv"
        self.warm_train_path = self.input_root + "/warm_start_train.csv"
        self.warm_test_path = self.input_root + "/warm_start_test.csv"

        self.stylecache_root = (
            self.root + "stylecache/_" + str(int(self.ratio[0] * 10)) + "_" + str(int(self.ratio[1] * 10)) + "/tgt_" + self.tgt + "_src_" + self.src
        )

        self.csvfile = config["csvfile"]
        self.seed = config["seed"]

        isResidual = "residual"
        aggregaion_name = str(True)
        self.rqvae_ckpt_root = (
            self.root
            + "rqvae_ckpt_0520/"
            + self.src
            + "_"
            + str(int(self.ratio[0] * 10))
            + "_"
            + str(int(self.ratio[1] * 10))
            + "/"
            + str(config["codebook_num"])
            + "_"
            + str(config["codebook_size"])
            + "_"
            + str(config["pretrain_epochs"])
            + "ep_"
            + str(config["rqvae_lr"])
            + "lr_"
            + aggregaion_name
            + "_"
            + isResidual
            + "_data_src"
            + "_tgt_"
            + self.tgt
            + "MF_MLP"
            +"prog" if self.seed == 1 else (str(self.seed) + "0518")
        )

        self.results = {
            "tgt_mae": 10,
            "tgt_rmse": 10,
            "aug_mae": 10,
            "aug_rmse": 10,
            "emcdr_mae": 10,
            "emcdr_rmse": 10,
            "ptupcdr_mae": 10,
            "ptupcdr_rmse": 10,
            "diff_mae": 10,
            "diff_rmse": 10,
            "diff_parallel_mae": 10,
            "diff_parallel_rmse": 10,
            "sscdr_mae": 10,
            "sscdr_rmse": 10,
            "lacdr_mae": 10,
            "lacdr_rmse": 10,
        }

        self.parallel_setting = {
            "set_aggr": config["set_aggr"],
            "aggregation": config["aggregation"],
            "bias_mapping": config["bias_mapping"],
            "mapping_lambda": config["mapping_lambda"],
            "uniformity_loss": config["uniformity_loss"],
            "zero_cond": config["zero_cond"],
            "batch_norm": config["batch_norm"],
            "recon_loss": config["recon_loss"]
        }

        self.rqvae_setting = {
            "codebook_num": config["codebook_num"],
            "codebook_size": config["codebook_size"],
            "RQVAE": config["RQVAE"],
            "start_point": config["start_point"],
            "pretrain_rq": config["pretrain_rq"],
            "pretrain_epochs": config["pretrain_epochs"],
            "freeze_rq": config["freeze_rq"],
            "rqvae_lr": config["rqvae_lr"],
            "cross_cond": config["cross_cond"],
            "rq_num": config["rq_num"]
        }

        self.device = "cuda" if config["use_cuda"] else "cpu"

        self.diff_lr = config["diff_lr"]
        self.diff_steps = config["diff_steps"]
        self.diff_sample_steps = config["diff_sample_steps"]
        self.diff_scale = config["diff_scale"]
        self.diff_dim = config["diff_dim"]
        self.diff_task_lambda = config["diff_task_lambda"]
        self.diff_mask_rate = config["diff_mask_rate"]
        self.diff_parallel_mode = config.get("diff_parallel_mode", "mean")
        parallel_weights = config.get("diff_parallel_weights", [0.5, 0.5])
        if isinstance(parallel_weights, str):
            try:
                parallel_weights = json.loads(parallel_weights)
            except Exception:
                parallel_weights = [0.5, 0.5]
        if not isinstance(parallel_weights, list) or len(parallel_weights) == 0:
            parallel_weights = [0.5, 0.5]
        total_weight = sum(parallel_weights)
        if total_weight <= 0:
            parallel_weights = [1.0 / len(parallel_weights)] * len(parallel_weights)
        else:
            parallel_weights = [w / total_weight for w in parallel_weights]
        self.diff_parallel_weights = parallel_weights

    def seq_extractor(self, x):
        x = x.rstrip("]").lstrip("[").split(", ")
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False, shuffle=True):
        if not history:
            cols = ["uid", "iid", "y"]
            x_col = ["uid", "iid"]
            y_col = ["y"]
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=shuffle)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ["uid", "iid", "y", "pos_seq"]
            x_col = ["uid", "iid"]
            y_col = ["y"]
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding="post")
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=shuffle)
            print(f"test - target mean: {y.float().mean().item()} +- {y.float().std().item()}")
            return data_iter

    def read_map_data(self, data_path):
        cols = ["uid", "iid", "y", "pos_seq"]
        data = pd.read_csv(data_path, header=None)
        data.columns = cols
        X = torch.tensor(data["uid"].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_diff_data(self, data_path, batch_size, shuffle=True):

        meta_uid_seq = pd.read_csv(data_path, header=None)
        meta_uid_seq.columns = ["meta_uid", "iid", "y", "pos_seq"]
        meta_uid = torch.tensor(meta_uid_seq["meta_uid"].values, dtype=torch.long)

        iid_input = torch.tensor(meta_uid_seq[["iid"]].values, dtype=torch.long)
        y_input = torch.tensor(meta_uid_seq[["y"]].values, dtype=torch.long)

        if self.use_cuda:
            meta_uid = meta_uid.cuda()
            iid_input = iid_input.cuda()
            y_input = y_input.cuda()

        dataset = TensorDataset(meta_uid, iid_input, y_input)
        data_iter = DataLoader(dataset, batch_size, shuffle=shuffle)

        print(f"data_diff - target mean: {y_input.float().mean().item()} +- {y_input.float().std().item()}")
        return data_iter

    def build_graph_inputs(self, data_path, include_users=None, exclude_users=None):
        interactions = pd.read_csv(data_path, header=None, usecols=[0, 1])
        interactions.columns = ["uid", "iid"]

        if include_users is not None:
            include_users = set(include_users)
            interactions = interactions[interactions["uid"].isin(include_users)]
        if exclude_users is not None:
            exclude_users = set(exclude_users)
            interactions = interactions[~interactions["uid"].isin(exclude_users)]

        if interactions.empty:
            return None

        user_ids = torch.tensor(interactions["uid"].values, dtype=torch.long)
        item_ids = torch.tensor(interactions["iid"].values, dtype=torch.long)
        edge_values = torch.ones(user_ids.shape[0], dtype=torch.float32)

        uv_indices = torch.stack([user_ids, item_ids])
        uv_adj = torch.sparse_coo_tensor(uv_indices, edge_values, size=(self.uid_all, self.iid_all + 1)).coalesce()

        vu_indices = torch.stack([item_ids, user_ids])
        vu_adj = torch.sparse_coo_tensor(vu_indices, edge_values, size=(self.iid_all + 1, self.uid_all)).coalesce()

        return {
            "uv_adj": uv_adj,
            "vu_adj": vu_adj,
            "user_ids": torch.unique(user_ids),
            "item_ids": torch.unique(item_ids),
            "num_edges": user_ids.shape[0],
        }

    def read_ss_data(self, data_path):
        """ """
        cols = ["uid", "iid", "y", "pos_seq"]
        meta_data = pd.read_csv(data_path, header=None)
        meta_data.columns = cols
        meta_data.drop(["y"], axis=1, inplace=True)

        # neg sample
        meta_data["pos_seq"] = meta_data["pos_seq"].str[1:-1]
        meta_data["pos_seq"] = meta_data["pos_seq"].str.split(",")
        meta_data["pos_split_len"] = [len(x) for x in meta_data["pos_seq"]]
        meta_data["positive_s_i"] = [np.random.choice(x, 1)[0] for x in meta_data["pos_split_len"]]
        meta_data["positive_s_i"] = [int(x[y]) if x != [""] else 0 for x, y in zip(meta_data["pos_seq"], meta_data["positive_s_i"])]

        # hist item
        all_his_item = set()
        for x_seq in meta_data["pos_seq"]:
            for x in x_seq:
                if x != "":
                    all_his_item.add(int(x))

        all_his_item = list(all_his_item)
        neg_s_i = np.random.choice(len(all_his_item), meta_data.shape[0])

        meta_data["negetive_s_i"] = [all_his_item[x] for x in neg_s_i]

        x_u = torch.tensor(meta_data["uid"], dtype=torch.long)
        x_p_i = torch.tensor(meta_data["positive_s_i"], dtype=torch.long)
        x_n_i = torch.tensor(meta_data["negetive_s_i"], dtype=torch.long)
        x_t_u = torch.tensor(meta_data["uid"], dtype=torch.long)

        del meta_data, all_his_item, neg_s_i

        if self.use_cuda:
            x_u = x_u.cuda()
            x_p_i = x_p_i.cuda()
            x_n_i = x_n_i.cuda()
            x_t_u = x_t_u.cuda()
        dataset = TensorDataset(x_u, x_p_i, x_n_i, x_t_u)
        data_iter = DataLoader(dataset, self.batchsize_ss, shuffle=True)

        return data_iter

    def read_la_data(self):

        # overlap
        cols = ["uid", "iid", "y", "pos_seq"]
        meta_data = pd.read_csv(self.meta_path, header=None)
        meta_data.columns = cols
        meta_data.drop(["y"], axis=1, inplace=True)

        # full_uid = meta_data[['uid']].drop_duplicates()
        full_uid = meta_data[["uid"]]

        full_uid["mask_src"] = 1
        full_uid["mask_tgt"] = 1

        x_uid = torch.tensor(full_uid["uid"], dtype=torch.long)
        x_mask_src = torch.tensor(full_uid["mask_src"], dtype=torch.long)
        x_mask_tgt = torch.tensor(full_uid["mask_tgt"], dtype=torch.long)

        del meta_data, full_uid

        if self.use_cuda:
            x_uid = x_uid.cuda()
            x_mask_src = x_mask_src.cuda()
            x_mask_tgt = x_mask_tgt.cuda()
        dataset = TensorDataset(x_uid, x_mask_src, x_mask_tgt)
        data_iter = DataLoader(dataset, self.batchsize_la, shuffle=True)

        return data_iter

    def read_aug_data(self, tgt_path):
        # merge source train , target train

        cols_train = ["uid", "iid", "y"]
        x_col = ["uid", "iid"]
        y_col = ["y"]
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):  # 데이터로더 생성하고 학습 단계별로 재사용.
        print(f"src: {self.src_path}")
        print("========Reading data========")
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print("src {} iter / batchsize = {} ".format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print("tgt {} iter / batchsize = {} ".format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print("meta {} iter / batchsize = {} ".format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data(self.meta_path)
        print("map {} iter / batchsize = {} ".format(len(data_map), self.batchsize_map))

        data_diff = self.read_diff_data(self.meta_path, batch_size=self.batchsize_diff)
        print("diff {} iter / batchsize = {} ".format(len(data_diff), self.batchsize_diff))

        data_aug = self.read_aug_data(self.tgt_path)
        print("aug {} iter / batchsize = {} ".format(len(data_aug), self.batchsize_aug))

        data_ss = self.read_ss_data(self.meta_path)
        print("ss {} iter / batchsize = {} ".format(len(data_ss), self.batchsize_ss))

        data_la = self.read_la_data()
        print("la {} iter / batchsize = {} ".format(len(data_la), self.batchsize_la))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True, shuffle=False)
        print("test {} iter / batchsize = {} ".format(len(data_test), self.batchsize_test))

        data_diff_test = self.read_diff_data(self.test_path, batch_size=self.batchsize_diff_test, shuffle=False)
        print("diff {} iter / batchsize = {} ".format(len(data_diff_test), self.batchsize_diff_test))

        test_users_df = pd.read_csv(self.test_path, header=None, usecols=[0])
        test_users = test_users_df[0].tolist()

        graph_src_train = self.build_graph_inputs(self.src_path)  # 전체 그래프 생성
        graph_tgt_train = self.build_graph_inputs(self.tgt_path, exclude_users=test_users)

        graph_src_test = graph_src_train  # train, test graph 동일
        graph_tgt_test = graph_tgt_train  # tgt_test는 안 쓰임

        def _print_graph_stats(name, graph):
            if graph is None:
                print("{} edges: 0 , unique users: 0, unique items: 0".format(name))
            else:
                print(
                    "{} edges: {} , unique users: {}, unique items: {}".format(
                        name, graph["num_edges"], graph["user_ids"].shape[0], graph["item_ids"].shape[0]
                    )
                )

        _print_graph_stats("graph src train", graph_src_train)
        _print_graph_stats("graph src test", graph_src_test)
        _print_graph_stats("graph tgt train", graph_tgt_train)
        _print_graph_stats("graph tgt test", graph_tgt_test)

        graph_data = {
            "train": {
                "src": graph_src_train,
                "tgt": graph_tgt_train,
            },
            "test": {
                "src": graph_src_test,
                "tgt": graph_tgt_test,
            },
        }

        return data_src, data_tgt, data_meta, data_map, data_diff, data_aug, data_ss, data_la, data_test, data_diff_test, graph_data

    def compute_user_graph_embeddings(
        self, base_model, diff_model, graph_data, use_target=False, low_deg=5, high_deg=30, min_aggr_ratio=0.1, max_aggr_ratio=0.9
    ):
        if graph_data is None:
            return None, None

        uv_adj = graph_data["uv_adj"].cpu()
        vu_adj = graph_data["vu_adj"].cpu()

        if use_target:
            model = base_model.tgt_model
        else:
            model = base_model.src_model

        with torch.no_grad():
            all_uid = torch.arange(model.uid_embedding.num_embeddings, device=self.device)

            raw_user_feat = model.uid_embedding(all_uid)
            user_feat = model.user_mlp(raw_user_feat).detach().cpu()

            # --------------------------------------------------
            # 2-hop aggregation
            # user -> item -> user
            # --------------------------------------------------
            item_msg = torch.sparse.mm(vu_adj, user_feat)
            user_2hop = torch.sparse.mm(uv_adj, item_msg)

            user_deg = torch.sparse.sum(uv_adj, dim=1).to_dense().unsqueeze(1)

            # remove self contribution
            user_2hop = user_2hop - user_deg * user_feat

            item_deg = torch.sparse.sum(vu_adj, dim=1).to_dense()
            item_other = torch.relu(item_deg - 1)

            two_hop_counts = torch.sparse.mm(uv_adj, item_other[:, None]).to_dense()

            norm = torch.where(
                two_hop_counts == 0,
                torch.ones_like(two_hop_counts),
                two_hop_counts,
            )

            aggr_user_feat = user_2hop / norm

            zero_mask = two_hop_counts.squeeze(1) == 0
            aggr_user_feat[zero_mask] = user_feat[zero_mask]

            # --------------------------------------------------
            # degree-adaptive mixing
            # low degree  -> use aggregation more
            # high degree -> keep own embedding more
            # --------------------------------------------------
            degree_raw = user_deg.squeeze(1).float()

            # degree가 low_deg 이하이면 ratio ≈ max_aggr_ratio
            # degree가 high_deg 이상이면 ratio ≈ min_aggr_ratio
            deg_clamped = torch.clamp(degree_raw, min=low_deg, max=high_deg)

            aggr_ratio = 1.0 - (deg_clamped - low_deg) / (high_deg - low_deg + 1e-8)

            aggr_ratio = min_aggr_ratio + aggr_ratio * (max_aggr_ratio - min_aggr_ratio)

            aggr_ratio = aggr_ratio.unsqueeze(1)

            user_emb = aggr_ratio * aggr_user_feat + (1.0 - aggr_ratio) * user_feat

            # two-hop neighbor 없는 유저는 무조건 자기 embedding 유지
            user_emb[zero_mask] = user_feat[zero_mask]

            # degree info
            if use_target:
                degree_by_uid = None
            else:
                degree_by_uid = torch.log1p(degree_raw).to(self.device)

        return user_emb.to(self.device), degree_by_uid

    def get_model(self):
        if self.base_model == "MF":
            model = MFBasedModel(self.uid_all, self.iid_all, self.emb_dim, self.meta_dim)
        else:
            raise ValueError("Unknown base model: " + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model, diff_model=None, ss_model=None, la_model=None):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)

        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)

        if diff_model is None and ss_model is None and la_model is None:
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map

        elif diff_model is None and ss_model is not None and la_model is None:
            optimizer_ss = torch.optim.Adam(params=ss_model.parameters(), lr=self.lr, weight_decay=self.wd)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_ss, optimizer_map

        elif diff_model is None and la_model is not None:
            optimizer_la = torch.optim.Adam(params=la_model.parameters(), lr=self.la_lr, weight_decay=self.wd)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_la, optimizer_map

        elif diff_model is not None and isinstance(diff_model, Diff.DiffCDR):
            optimizer_diff = torch.optim.Adam(params=diff_model.parameters(), lr=self.diff_lr)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_diff, optimizer_map

        elif diff_model is not None and isinstance(diff_model, Diff.DiffParallel):
            optimizer_diff = torch.optim.Adam(list(diff_model.parameters()), lr=self.diff_lr)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_diff, optimizer_map

    def eval_mae(self, model, data_loader, stage, style_src=None):
        print("Evaluating MAE:")

        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()

        with torch.no_grad():
            if stage in ("test_diff"):
                for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval()
                    model[1].eval()
                    pred = model[0](X, stage, self.device, diff_model=model[1])

                    y_input = X[-1]
                    targets.extend(y_input.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

            elif stage in ("test_diff_parallel"):
                y_all = []
                mae_all = []
                pred_all = []
                uid_all = []
                degree_all = []

                test_users_degree = model[1].test_users_degree  # dict: {uid: degree}
                test_users_pop_group = model[1].test_users_pop_group

                for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval()
                    model[1].eval()

                    pred = model[0](X, stage, self.device, diff_model=model[1], style_src=style_src)

                    meta_uid = X[0]  # [B]
                    y_input = X[-1]  # [B, 1] or [B]

                    meta_uid = meta_uid
                    y_true = y_input.squeeze(1)
                    pred = pred

                    targets.extend(y_true.tolist())
                    predicts.extend(pred.tolist())

                    mae = (pred - y_true).abs()

                    y_all.append(y_true.cpu())
                    mae_all.append(mae.cpu())
                    pred_all.append(pred.detach().cpu())
                    uid_all.append(meta_uid.detach().cpu())

                    batch_degree = torch.tensor([test_users_degree.get(uid.item(), 0) for uid in meta_uid.detach().cpu()], dtype=torch.long)
                    degree_all.append(batch_degree)

                y_all = torch.cat(y_all)
                mae_all = torch.cat(mae_all)
                pred_all = torch.cat(pred_all)
                uid_all = torch.cat(uid_all)
                degree_all = torch.cat(degree_all)

                df_score_summary = mae_summary_by_score(y_all.numpy(), mae_all.numpy())
                print(df_score_summary)

                df_sparsity_summary = mae_rmse_summary_by_sparsity(y_true=y_all, y_pred=pred_all, user_degree=degree_all, user_uid=uid_all, n_bins=5)
                print(df_sparsity_summary)

                df_pop_summary = mae_rmse_summary_by_pop_group(
                    y_true=y_all, y_pred=pred_all, user_uid=uid_all, test_users_pop_group=test_users_pop_group
                )
                print(df_pop_summary)

            elif stage in ("test_ss"):
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval()
                    model[1].eval()
                    pred = model[0](X, stage, self.device, diff_model=None, ss_model=model[1])
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

            elif stage in ("test_la"):
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval()
                    model[1].eval()
                    pred = model[0](X, stage, self.device, diff_model=None, la_model=model[1])
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

            else:
                for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model.eval()
                    pred = model(X, stage, self.device)
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        print(f"Target mean: {targets.mean().item():>10.6f} ± {targets.std().item():<10.6f}")
        print(f"Predic mean: {predicts.mean().item():>10.6f} ± {predicts.std().item():<10.6f}")

        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(
        self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False, diff=False, ss=False, la=False, graph_train=None, style_src=None
    ):
        print("Training Epoch {}:".format(epoch + 1))

        loss_ls = []
        # MAE / RMSE 계산용 변수
        sum_abs_err = 0.0
        sum_sq_err = 0.0
        total_count = 0

        mse_loss_sum = 0.0
        uni_loss_sum = 0.0
        total_loss_sum = 0.0
        num_batches = 0

        if diff == False and ss == False and la == False:
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                if mapping:
                    model.train()

                    src_emb, tgt_emb = model(X, stage, self.device)
                    loss = criterion(src_emb, tgt_emb)

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                else:
                    model.train()

                    pred, uni_loss = model(X, stage, self.device)
                    mse_loss = criterion(pred, y.squeeze().float())
                    total_loss = mse_loss + uni_loss

                    model.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # -----------------------------
                    # accumulate
                    # -----------------------------
                    mse_loss_sum += mse_loss.item()
                    uni_loss_sum += uni_loss.item()
                    total_loss_sum += total_loss.item()
                    num_batches += 1

                loss_ls.append(total_loss.item())

                # ====== MAE / RMSE 계산 ======
                with torch.no_grad():
                    y_true = y.squeeze()
                    abs_err = torch.abs(pred - y_true)
                    sq_err = (pred - y_true) ** 2

                    sum_abs_err += abs_err.sum().item()
                    sum_sq_err += sq_err.sum().item()
                    total_count += y_true.numel()

            # ====== epoch metric 계산 ======
            mae = sum_abs_err / total_count
            rmse = (sum_sq_err / total_count) ** 0.5

            # =====================================
            # epoch logging
            # =====================================
            mse_avg = mse_loss_sum / num_batches
            uni_avg = uni_loss_sum / num_batches
            total_avg = total_loss_sum / num_batches

            print(f"[Epoch {epoch+1}] total: {total_avg:.4f} | mse: {mse_avg:.4f} | uni: {uni_avg:.4f}")

            print(f"[Epoch {epoch+1}] | MAE: {mae:.6f} | RMSE: {rmse:.6f}")

            return torch.tensor(loss_ls).mean()

        elif diff == False and ss == True and la == False:
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()
                loss = model[0](X, stage, self.device, diff_model=None, ss_model=model[1])

                model[1].zero_grad()
                loss.backward()
                optimizer.step()

                loss_ls.append(loss.item())
            return torch.tensor(loss_ls).mean()

        elif diff == False and la == True:
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()
                loss = model[0](X, stage, self.device, diff_model=None, la_model=model[1])

                model[1].zero_grad()
                loss.backward()
                optimizer.step()

                loss_ls.append(loss.item())
            return torch.tensor(loss_ls).mean()

        elif diff == True:
            loss1_list = []
            loss2_list = []

            task_loss_list = []
            uni_loss_list = []

            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()

                # ------------------------
                # diffusion loss
                # ------------------------
                diff_loss = model[0](X, stage, self.device, diff_model=model[1], is_task=False, style_src=style_src)
                loss1 = diff_loss

                optimizer.zero_grad(set_to_none=True)
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(list(model[1].parameters()), 1.0)
                optimizer.step()

                # ------------------------
                # task + uniformity
                # ------------------------
                task_loss, uni_loss = model[0](X, stage, self.device, diff_model=model[1], is_task=True, style_src=style_src)
                loss2 = task_loss + uni_loss

                optimizer.zero_grad(set_to_none=True)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(list(model[1].parameters()), 1.0)
                optimizer.step()

                # ------------------------
                # logging
                # ------------------------
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())

                task_loss_list.append(task_loss.item())
                uni_loss_list.append(uni_loss.item())

            # ------------------------
            # epoch logging
            # ------------------------
            diff_mean = sum(loss1_list) / len(loss1_list)
            total_mean = sum(loss2_list) / len(loss2_list)
            task_mean = sum(task_loss_list) / len(task_loss_list)
            uni_mean = sum(uni_loss_list) / len(uni_loss_list)

            print(f"[Epoch {epoch+1}] diff={diff_mean:.4f} | task={task_mean:.4f} | uni={uni_mean:.4f} | total={total_mean:.4f}")

            return diff_mean, total_mean

    def update_results(self, mae, rmse, phase):

        if mae < self.results[phase + "_mae"]:
            self.results[phase + "_mae"] = mae
        if rmse < self.results[phase + "_rmse"]:
            self.results[phase + "_rmse"] = rmse

    def reset_results(self):
        self.results = {
            "tgt_mae": 10,
            "tgt_rmse": 10,
            "aug_mae": 10,
            "aug_rmse": 10,
            "emcdr_mae": 10,
            "emcdr_rmse": 10,
            "ptupcdr_mae": 10,
            "ptupcdr_rmse": 10,
            "diff_mae": 10,
            "diff_rmse": 10,
            "diff_parallel_mae": 10,
            "diff_parallel_rmse": 10,
            "sscdr_mae": 10,
            "sscdr_rmse": 10,
            "lacdr_mae": 10,
            "lacdr_rmse": 10,
        }

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        write("=========TgtOnly========")
        n_epoch = self.epoch

        for i in range(n_epoch):
            loss = self.train(data_tgt, model, criterion, optimizer, i, stage="train_tgt")
            mae, rmse = self.eval_mae(model, data_test, stage="test_tgt")
            self.update_results(mae, rmse, "tgt")
            write("MAE: {} RMSE: {} ".format(mae, rmse))

    def SrcOnly(self, model, data_src, criterion, optimizer_src):
        write("=====SrcOnly=====")
        for i in range(self.epoch):
            loss = self.train(data_src, model, criterion, optimizer_src, i, stage="train_src")

    def BPRMF(self, data_loader, model, optimizer, stage):
        model.train()
        total_loss = []

        for X, _ in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            uid = X[:, 0]
            pos_iid = X[:, 1]

            # negative sampling
            neg_iid = torch.randint(low=0, high=self.iid_all, size=pos_iid.size(), device=pos_iid.device)

            # embeddings
            if stage == "src":
                user_emb = model.src_model.uid_embedding(uid)
                pos_item_emb = model.src_model.iid_embedding(pos_iid)
                neg_item_emb = model.src_model.iid_embedding(neg_iid)
            else:
                user_emb = model.tgt_model.uid_embedding(uid)
                pos_item_emb = model.tgt_model.iid_embedding(pos_iid)
                neg_item_emb = model.tgt_model.iid_embedding(neg_iid)

            # scores
            pos_score = torch.sum(user_emb * pos_item_emb, dim=1)
            neg_score = torch.sum(user_emb * neg_item_emb, dim=1)

            # BPR loss
            loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        return torch.tensor(total_loss).mean()

    def lightgcn_propagate(self, user_emb, item_emb, uv_adj, vu_adj, num_layers=2):
        """
        user_emb: [num_users, d]
        item_emb: [num_items, d]
        """
        u_list = [user_emb]
        i_list = [item_emb]

        u, i = user_emb, item_emb
        for _ in range(num_layers):
            u = torch.sparse.mm(uv_adj, i)
            i = torch.sparse.mm(vu_adj, u)
            u_list.append(u)
            i_list.append(i)

        u_final = torch.stack(u_list, dim=0).mean(dim=0)
        i_final = torch.stack(i_list, dim=0).mean(dim=0)
        return u_final, i_final

    def LightGCN_BPR(self, model, graph, optimizer, stage, num_layers=2):
        model.train()
        uv_adj = graph["uv_adj"].to(self.device)
        vu_adj = graph["vu_adj"].to(self.device)

        if stage == "src":
            user_emb = model.src_model.uid_embedding.weight
            item_emb = model.src_model.iid_embedding.weight
        else:
            user_emb = model.tgt_model.uid_embedding.weight
            item_emb = model.tgt_model.iid_embedding.weight

        # LightGCN propagation
        u_g, i_g = self.lightgcn_propagate(user_emb, item_emb, uv_adj, vu_adj, num_layers)

        # sample edges
        users = graph["user_ids"].to(self.device)
        idx = torch.randint(0, users.shape[0], (self.batchsize_src,), device=self.device)
        u = users[idx]

        # positive items
        edges = uv_adj.indices()
        mask = torch.isin(edges[0], u)
        pos_i = edges[1][mask][: u.shape[0]]

        if pos_i.shape[0] < u.shape[0]:
            return torch.tensor(0.0, device=self.device)

        pos_i = pos_i[: u.shape[0]]
        neg_i = torch.randint(0, self.iid_all, pos_i.shape, device=self.device)

        pos_score = (u_g[u] * i_g[pos_i]).sum(dim=1)
        neg_score = (u_g[u] * i_g[neg_i]).sum(dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach()

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        write("=========DataAug========")
        n_epoch = self.epoch

        for i in range(n_epoch):
            loss = self.train(data_aug, model, criterion, optimizer, i, stage="train_aug")
            mae, rmse = self.eval_mae(model, data_test, stage="test_aug")
            self.update_results(mae, rmse, "aug")
            write("MAE: {} RMSE: {} ".format(mae, rmse))

    def Diff_CDR(self, model, diff_model, data_diff, data_test, optimizer):
        write(f"{' Diff_CDR ':=^{30}}")
        for i in range(self.epoch):
            loss, task_loss = self.train(data_diff, [model, diff_model], None, optimizer, i, stage="train_diff", mapping=False, diff=True)

            mae, rmse = self.eval_mae([model, diff_model], data_test, stage="test_diff")
            self.update_results(mae, rmse, "diff")
            write(f"DIFF LOSS {loss.item():>10.6f} |  TASK LOSS {task_loss.item():>10.6f} | MAE: {mae:>10.6f} | RMSE: {rmse:>10.6f}")

    def Diff_Parallel(
        self, model, diff_model, data_src, data_diff, data_test, optimizer, graph_train, graph_test, style_src, style_tgt_item, style_tgt_user
    ):
        write(f"{' Diff_Parallel ':=^{30}}")

        diff_model.style_tgt_item = style_tgt_item

        diff_model.style_tgt_user = style_tgt_user.cuda()

        src_graph = graph_train.get("src")
        tgt_graph = graph_train.get("tgt")

        smooth_user_emb_src, degree_by_uid = self.compute_user_graph_embeddings(model, diff_model, src_graph, use_target=False)
        smooth_user_emb_tgt, _ = self.compute_user_graph_embeddings(model, diff_model, tgt_graph, use_target=True)

        diff_model.smooth_user_emb_src = smooth_user_emb_src
        diff_model.smooth_user_emb_tgt = smooth_user_emb_tgt
        diff_model.degree_by_uid = degree_by_uid

        # [PRETRAIN] RQ-VAE pretraining if requested
        if self.rqvae_setting.get("pretrain_rq", False):
            if os.path.exists(self.rqvae_ckpt_root):
                self.load_rqvae(diff_model, self.rqvae_ckpt_root)
            else:
                self.pretrain_rqvae(model, diff_model, data_src)
                self.save_rqvae(diff_model, self.rqvae_ckpt_root)

        # [FREEZE] Freeze RQ-VAE parameters if requested
        if self.rqvae_setting.get("freeze_rq", False):
            write("Freezing RQ-VAE parameters.")
            if hasattr(diff_model, "rq_mf"):
                for p in diff_model.rq_mf.parameters():
                    p.requires_grad = False
            if hasattr(diff_model, "rq_aggr"):
                for p in diff_model.rq_aggr.parameters():
                    p.requires_grad = False

        for i in range(self.epoch):

            loss, task_loss = self.train(
                data_diff, [model, diff_model], None, optimizer, i, stage="train_diff_parallel", mapping=False, diff=True, style_src=style_src
            )

            mae, rmse = self.eval_mae([model, diff_model], data_test, stage="test_diff_parallel", style_src=style_src)
            self.update_results(mae, rmse, "diff_parallel")
            write(f"Epoch {i+1} : MAE: {mae:>10.6f} | RMSE: {rmse:>10.6f}")

    def save_rqvae(self, diff_model, path):
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        state = {}

        if hasattr(diff_model, "rq_mf"):
            state["rq_mf"] = diff_model.rq_mf.state_dict()

        if self.parallel_setting.get("aggregation", True) and hasattr(diff_model, "rq_aggr"):
            state["rq_aggr"] = diff_model.rq_aggr.state_dict()

        torch.save(state, path)
        write(f"Saved RQ-VAE checkpoint to {path}")

    def load_rqvae(self, diff_model, path):
        write(f"Loading pretrained RQ-VAE.")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        if "rq_mf" in ckpt and hasattr(diff_model, "rq_mf"):
            diff_model.rq_mf.load_state_dict(ckpt["rq_mf"])

        if "rq_aggr" in ckpt and hasattr(diff_model, "rq_aggr"):
            diff_model.rq_aggr.load_state_dict(ckpt["rq_aggr"])

    def debug_rqvae_quantizer(self, name, quantizer, z, epoch=None):
        """
        z: [N, D] or [B, D]
        """
        quantizer.eval()

        with torch.no_grad():
            quantized, all_level_vectors, rq_loss = quantizer(z)

            # reconstruction
            recon_mse = F.mse_loss(quantized, z).item()
            recon_mae = torch.mean(torch.abs(quantized - z)).item()

            # cosine similarity
            cos = F.cosine_similarity(quantized, z, dim=1)
            cos_mean = cos.mean().item()
            cos_std = cos.std().item()

            # norm
            z_norm = z.norm(dim=1).mean().item()
            q_norm = quantized.norm(dim=1).mean().item()

            # level별 norm: [L]
            level_norms = all_level_vectors.norm(dim=-1).mean(dim=1)

            # level별 contribution ratio
            level_ratios = level_norms / (level_norms.sum() + 1e-8)

            title = f"[{name}] RQ-VAE Debug"
            if epoch is not None:
                title += f" | Epoch {epoch}"

            print("\n" + title)
            print(f"Recon MSE : {recon_mse:.6f}")
            print(f"Recon MAE : {recon_mae:.6f}")
            print(f"Cos Mean  : {cos_mean:.6f}")
            print(f"Cos Std   : {cos_std:.6f}")
            print(f"Z Norm    : {z_norm:.6f}")
            print(f"Q Norm    : {q_norm:.6f}")

            print("Level Norms:")
            for l, v in enumerate(level_norms):
                print(f"  L{l+1}: {v.item():.6f}")

            print("Level Ratios:")
            for l, v in enumerate(level_ratios):
                print(f"  L{l+1}: {v.item():.4f}")

        quantizer.train()

    def pretrain_rqvae(self, model, diff_model, data_diff):
        write(f"{' RQ-VAE Pretraining ':=^{30}}")

        # 1) MF quantizer 따로 pretrain
        if hasattr(diff_model, "rq_mf"):
            self._pretrain_single_quantizer(
                name="rq_mf",
                quantizer=diff_model.rq_mf,
                # embed_fetch_fn=lambda tgt_uid: model.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze(),
                embed_fetch_fn=lambda tgt_uid: model.src_model.user_mlp(model.src_model.uid_embedding(tgt_uid.unsqueeze(1)).squeeze()),
                data_diff=data_diff,
            )

        # 2) Aggr quantizer 따로 pretrain
        if hasattr(diff_model, "rq_aggr"):
            self._pretrain_single_quantizer(
                name="rq_aggr",
                quantizer=diff_model.rq_aggr,
                embed_fetch_fn=lambda tgt_uid: model._fetch_vbge_user_embedding(diff_model, tgt_uid, use_target=False),
                data_diff=data_diff,
            )

    def _pretrain_single_quantizer(self, name, quantizer, embed_fetch_fn, data_diff):
        """
        rq_mf 또는 rq_aggr 하나만 따로 pretrain
        """
        if quantizer is None:
            write(f"Skip {name}: module is None")
            return

        pretrain_epochs = self.rqvae_setting.get("pretrain_epochs", 50)
        optimizer = torch.optim.Adam(quantizer.parameters(), lr=self.rqvae_setting["rqvae_lr"])

        quantizer.train()

        write(f"{' Pretraining ' + name + ' ':=^{30}}")

        with torch.no_grad():
            all_uids = torch.arange(self.uid_all).to(self.device)
            all_z = embed_fetch_fn(all_uids)  # [N, D]

        z_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(all_z), batch_size=self.batchsize_diff, shuffle=True)

        for epoch in tqdm.tqdm(range(pretrain_epochs)):
            total_loss = 0.0

            for (z_batch,) in z_loader:
                _, _, loss = quantizer(z_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / len(z_loader)

                write(f"{name} Epoch {epoch+1}/{pretrain_epochs} | Loss: {avg_loss:.6f}")

                # 전체가 너무 크면 일부만 디버깅
                debug_z = all_z[: min(4096, all_z.size(0))]
                self.debug_rqvae_quantizer(name=name, quantizer=quantizer, z=debug_z, epoch=epoch + 1)

    def SS_CDR(self, model, ss_model, data_ss, data_test, optimizer_ss):
        write("==========SS_CDR==========")
        for i in range(self.epoch):
            loss = self.train(data_ss, [model, ss_model], None, optimizer_ss, i, stage="train_ss", mapping=False, diff=False, ss=True)
            mae, rmse = self.eval_mae([model, ss_model], data_test, stage="test_ss")
            self.update_results(mae, rmse, "sscdr")
            write("MAE: {} RMSE: {}".format(mae, rmse))

    def LA_CDR(self, model, la_model, data_la, data_test, optimizer_la):
        write("==========LA_CDR==========")
        for i in range(self.epoch):
            loss = self.train(data_la, [model, la_model], None, optimizer_la, i, stage="train_la", mapping=False, diff=False, ss=False, la=True)
            mae, rmse = self.eval_mae([model, la_model], data_test, stage="test_la")
            self.update_results(mae, rmse, "lacdr")
            write("LA LOSS", loss.item(), "MAE: {} RMSE: {}  ".format(mae, rmse))

    def CDR(self, model, data_map, data_meta, data_test, criterion, optimizer_map, optimizer_meta):

        write("==========EMCDR==========")
        for i in range(self.epoch):
            loss = self.train(data_map, model, criterion, optimizer_map, i, stage="train_map", mapping=True)
            mae, rmse = self.eval_mae(model, data_test, stage="test_map")
            self.update_results(mae, rmse, "emcdr")
            write("MAE: {} RMSE: {}  ".format(mae, rmse))
        write("==========PTUPCDR==========")
        for i in range(self.epoch):
            loss = self.train(data_meta, model, criterion, optimizer_meta, i, stage="train_meta")
            mae, rmse = self.eval_mae(model, data_test, stage="test_meta")
            self.update_results(mae, rmse, "ptupcdr")
            write("MAE: {} RMSE: {} ".format(mae, rmse))

    def model_save(self, model, path):
        torch.save(model.state_dict(), path)

    def model_load(self, model, path):
        if self.device == "cuda":
            # model.load_state_dict(torch.load(path))
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(torch.load(path, map_location="cpu"))

    def result_print(self, phase, exp_part):
        print_str = ""
        for p in phase:
            write(f"⬇️ Eval {p}: MAE & RMSE ")
            with open(self.csvfile, 'a', newline='') as f:
                writer = csv.writer(f)
                for m in ["_mae", "_rmse"]:
                    metric_name = p + m
                    print_str += metric_name + ": {:.6f} ".format(self.results[metric_name])
                    write(f"{self.results[metric_name]:.6f}")
                    writer.writerow([
                        self.seed,
                        p,
                        self.task,
                        self.ratio[0] * 10, 
                        self.ratio[1] * 10,
                        m.strip("_"),
                        self.results[metric_name]
                    ])

    def main(self, exp_part="None_CDR", save_path=None):
        # exp_part 에 따라 모델, 옵티마이져 초기화하고 학습.
        if exp_part == "diff_CDR":
            diff_model = Diff.DiffCDR(
                self.diff_steps, self.diff_dim, self.emb_dim, self.diff_scale, self.diff_sample_steps, self.diff_task_lambda, self.diff_mask_rate
            )
            diff_model = diff_model.cuda() if self.use_cuda else diff_model

            model = self.get_model()

            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_diff, optimizer_map = self.get_optimizer(model, diff_model)

        elif exp_part == "diff_parallel":
            diff_model = Diff.DiffParallel(
                self.diff_steps,
                self.diff_dim,
                self.emb_dim,
                self.diff_scale,
                self.diff_sample_steps,
                self.diff_task_lambda,
                self.diff_mask_rate,
                parallel=self.parallel_setting,
                rqvae=self.rqvae_setting,
            )
            diff_model = diff_model.cuda() if self.use_cuda else diff_model

            model = self.get_model()

            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_diff, optimizer_map = self.get_optimizer(model, diff_model)

        elif exp_part == "ss_CDR":
            ss_model = SSCDR.SSCDR(self.emb_dim)
            ss_model = ss_model.cuda() if self.use_cuda else ss_model

            model = self.get_model()
            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_ss, optimizer_map = self.get_optimizer(model, None, ss_model)

        elif exp_part == "la_CDR":
            la_model = LACDR.LACDR(self.emb_dim)
            la_model = la_model.cuda() if self.use_cuda else la_model

            model = self.get_model()
            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_la, optimizer_map = self.get_optimizer(model, None, None, la_model)

        else:
            model = self.get_model()
            optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)

        data_src, data_tgt, data_meta, data_map, data_diff, data_aug, data_ss, data_la, data_test, data_diff_test, graph_data = self.get_data()

        test_users_degree = get_test_users_degree(self.src_path, self.test_path)

        print("num test users:", len(test_users_degree))

        test_users_pop_group, user_pop_ratio, popular_items = get_test_users_pop_group(test_users_degree, self.src_path)

        for group, users in test_users_pop_group.items():
            print(group, len(users))

        print(f"\n소스 도메인 내 유저의 레이팅 스타일 정보 추출\n")
        cache_path = f"{self.stylecache_root}.pt"
        if os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
            style_src = ckpt["style"]
            info = ckpt["info"]
        else:
            style_src, info = build_src_user_rating_style_from_loader(
                data_src=data_src, num_users=self.uid_all, rating_min=1.0, rating_max=5.0, device="cpu", cache_path=cache_path
            )

        print(f"\n타겟 도메인 내 아이템의 레이팅 스타일 정보 추출\n")
        cache_path = f"{self.stylecache_root}_tgt_item.pt"
        if os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
            style_tgt_item = ckpt["style_tgt_item"]
            info_tgt = ckpt["info_tgt"]
        else:
            style_tgt_item, info_tgt = build_tgt_item_rating_style_from_loader(
                data_tgt=data_tgt, num_items_total=self.iid_all + 1, rating_min=1.0, rating_max=5.0, device="cpu", cache_path=cache_path
            )

        print(f"\n타겟 도메인 내 유저의 레이팅 스타일 정보 추출\n")
        cache_path = f"{self.stylecache_root}_tgt_user.pt"
        if os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
            style_tgt_user = ckpt["style"]
            info = ckpt["info"]
        else:
            style_tgt_user, info = build_src_user_rating_style_from_loader(
                data_src=data_tgt, num_users=self.uid_all, rating_min=1.0, rating_max=5.0, device="cpu", cache_path=cache_path
            )

        criterion = torch.nn.MSELoss()

        if exp_part == "None_CDR":
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.SrcOnly(model, data_src, criterion, optimizer_src)
            # CMF
            if self.base_model == "CMF":
                self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
                self.result_print(["CMF", "aug"], exp_part)
            else: 
                self.result_print(["tgt"], exp_part)
            self.model_save(model, path=save_path)

        elif exp_part == "CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.CDR(model, data_map, data_meta, data_test, criterion, optimizer_map, optimizer_meta)
            self.result_print(["emcdr", "ptupcdr"], exp_part)

        elif exp_part == "ss_CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.SS_CDR(model, ss_model, data_ss, data_test, optimizer_ss)
            self.result_print(["sscdr"], exp_part)

        elif exp_part == "la_CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.LA_CDR(model, la_model, data_la, data_test, optimizer_la)
            self.result_print(["lacdr"], exp_part)

        elif exp_part == "diff_CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.Diff_CDR(model, diff_model, data_diff, data_diff_test, optimizer_diff)
            self.result_print(["diff"], exp_part)

        elif exp_part == "diff_parallel":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            diff_model.test_users_degree = test_users_degree
            diff_model.test_users_pop_group = test_users_pop_group
            self.Diff_Parallel(
                model,
                diff_model,
                data_src,
                data_diff,
                data_diff_test,
                optimizer_diff,
                graph_data["train"],
                graph_data["test"],
                style_src,
                style_tgt_item,
                style_tgt_user,
            )
            self.result_print(["diff_parallel"], exp_part)
