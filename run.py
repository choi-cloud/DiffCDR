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

from utils import write
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

        self.item_cond = config["item_cond"]
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
            "set_loss": config["set_loss"],
            "set_init": config["set_init"],
            "set_proj": config["set_proj"],
            "set_aggr": config["set_aggr"],
        }
        self.rqvae_setting = {
            "codebook_num": config["codebook_num"],
            "codebook_size": config["codebook_size"],
            "alpha_rq": config["alpha_rq"],
        }
        self.w = config["w"]

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

    def build_shared_train_graph(self, src_path, tgt_path, exclude_users=None):
        """Build a single graph using both train_src and train_tgt interactions (same CSV schema)."""
        src_interactions = pd.read_csv(src_path, header=None, usecols=[0, 1])
        tgt_interactions = pd.read_csv(tgt_path, header=None, usecols=[0, 1])
        src_interactions.columns = ["uid", "iid"]
        tgt_interactions.columns = ["uid", "iid"]
        interactions = pd.concat([src_interactions, tgt_interactions], ignore_index=True)

        # keep item split info for later slicing
        self.num_src_items = src_interactions["iid"].nunique()
        self.num_tgt_items = tgt_interactions["iid"].nunique()

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

    def compute_item_popularity(self, paths):
        """Count item frequency from given CSVs (col 1 = iid), clip to >=1, normalize by max."""
        frames = []  # src train + tgt train
        for p in paths:
            df = pd.read_csv(p, header=None, usecols=[1])
            df.columns = ["iid"]
            frames.append(df)
        if not frames:
            return None
        counts = pd.concat(frames, ignore_index=True)["iid"].value_counts()  # 각 iid 마다 등장 횟수 카운팅
        full_counts = counts.reindex(range(self.iid_all + 1), fill_value=0).to_numpy()  # iid 번호 순서대로 정렬

        # TODO 정규화 - src/tgt 따로 or 같이? (현재는 같이 한번에 정규화)
        max_count = full_counts.max() if full_counts.size > 0 else 0
        if max_count == 0:
            return torch.zeros(self.iid_all + 1, dtype=torch.float32)
        full_counts = np.clip(full_counts, 1, max_count)
        pop_norm = full_counts / max_count
        return torch.tensor(pop_norm, dtype=torch.float32)

    def build_shared_test_graph(self, data_path, include_users=None, exclude_users=None):
        """Build a single graph from test.csv (has pos_seq, but only uid/iid are used for edges)."""
        interactions = pd.read_csv(data_path, header=None)
        interactions.columns = ["uid", "iid", "y", "pos_seq"]

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

    def build_test_graph_inputs(self, data_path, include_users=None, exclude_users=None):
        interactions = pd.read_csv(data_path, header=None)
        interactions.columns = ["uid", "iid", "y", "pos_seq"]

        if include_users is not None:
            include_users = set(include_users)
            interactions = interactions[interactions["uid"].isin(include_users)]
        if exclude_users is not None:
            exclude_users = set(exclude_users)
            interactions = interactions[~interactions["uid"].isin(exclude_users)]

        if interactions.empty:
            return None

        user_list = []
        item_list = []
        MAX_POS = 20  # history 길이 제한

        # 🔑 핵심: user 단위로 그룹핑 -> test_user 별로 pos_seq 한번씩만 추가
        for uid, group in interactions.groupby("uid"):
            pos_seq = ast.literal_eval(group.iloc[0]["pos_seq"])  # pos_seq는 user별로 모두 동일하므로 첫 row만 사용

            if not pos_seq:
                continue

            if len(pos_seq) > MAX_POS:
                pos_seq = pos_seq[-MAX_POS:]

            for iid in pos_seq:
                user_list.append(uid)
                item_list.append(iid)

        user_ids = torch.tensor(user_list, dtype=torch.long)
        item_ids = torch.tensor(item_list, dtype=torch.long)
        edge_values = torch.ones(len(user_ids), dtype=torch.float32)

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

        # item popularity on train src+tgt (normalized)
        self.item_popularity = self.compute_item_popularity([self.src_path, self.tgt_path]).cuda()

        graph_src_train = self.build_graph_inputs(self.src_path)  # 전체 그래프 생성
        graph_tgt_train = self.build_graph_inputs(self.tgt_path, exclude_users=test_users)

        graph_src_test = graph_src_train  # train, test graph 동일
        graph_tgt_test = graph_tgt_train  # tgt_test는 안 쓰임

        graph_shared_train = self.build_shared_train_graph(self.src_path, self.tgt_path, exclude_users=test_users)
        graph_shared_test = self.build_shared_test_graph(self.test_path)  # pos seq 와 그래프 생성 -> test user가 인터랙션한 source items.

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
        _print_graph_stats("graph shared train", graph_shared_train)
        _print_graph_stats("graph shared test", graph_shared_test)

        graph_data = {
            "train": {"src": graph_src_train, "tgt": graph_tgt_train, "shared": graph_shared_train},
            "test": {"src": graph_src_test, "tgt": graph_tgt_test, "shared": graph_shared_test},
        }

        return data_src, data_tgt, data_meta, data_map, data_diff, data_aug, data_ss, data_la, data_test, data_diff_test, graph_data

    def compute_user_graph_embeddings(self, base_model, diff_model, graph_data, use_target=False):
        if graph_data is None:
            return None, None

        # 단순 2홉 aggr
        # simple 2-hop aggregation (user -> items -> users) excluding 1-hop self contribution
        uv_adj = graph_data["uv_adj"].to(self.device)
        vu_adj = graph_data["vu_adj"].to(self.device)
        if use_target:
            user_feat = base_model.tgt_model.uid_embedding.weight.detach().to(self.device)
        else:
            user_feat = base_model.src_model.uid_embedding.weight.detach().to(self.device)
        with torch.no_grad():
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

        return user_emb, None

    def compute_item_aggregation_popularity(self, base_model, graph_data, src_item_num):
        uv_adj = graph_data["uv_adj"].to(self.device)  # [num_users, num_items]

        # MF item embedding
        src_item_feat = base_model.src_model.iid_embedding.weight.detach().to(self.device)[:src_item_num]  # [num_items, emb_dim]
        tgt_item_feat = base_model.tgt_model.iid_embedding.weight.detach().to(self.device)[src_item_num:]  # [num_items, emb_dim]
        item_feat = torch.cat([src_item_feat, tgt_item_feat], dim=0)  # [num_items, emb_dim]

        # item popularity
        conf_weight = self.item_popularity.to(self.device).unsqueeze(1)  # [num_items, 1]
        int_weight = torch.ones_like(conf_weight) - conf_weight

        with torch.no_grad():
            # popularity-weighted item embedding
            item_feat_conf = item_feat * conf_weight  # [num_items, d]
            item_feat_int = item_feat * int_weight  # [num_items, d]

            # 1-hop aggregation: user <- items
            user_agg_conf = torch.sparse.mm(uv_adj, item_feat_conf)  # [num_users, d]
            user_agg_int = torch.sparse.mm(uv_adj, item_feat_int)  # [num_users, d]

            # normalization term: sum of item popularities per user
            pop_sum_conf = torch.sparse.mm(uv_adj, conf_weight).clamp(min=1e-8)  # [num_users, 1] # 이웃 item들의 pop sum으로 정규화
            pop_sum_int = torch.sparse.mm(uv_adj, int_weight).clamp(min=1e-8)  # [num_users, 1] # 이웃 item들의 pop sum으로 정규화

            user_emb_conf = user_agg_conf / pop_sum_conf
            user_emb_int = user_agg_int / pop_sum_int

        return user_emb_conf, user_emb_int

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

        elif diff_model is not None:
            # optimizer_diff = torch.optim.Adam(params=diff_model.parameters(), lr=self.diff_lr)
            optimizer_diff = torch.optim.Adam(
                params=list(diff_model.parameters()) +
                    list(model.user_embedding.parameters()) +
                    list(model.item_embedding.parameters()),
                lr=self.diff_lr
            )
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

                for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval()
                    model[1].eval()
                    pred = model[0](X, stage, self.device, diff_model=model[1], item_cond=self.item_cond, style_src=style_src)
                    y_input = X[-1]
                    targets.extend(y_input.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

                    mae = (pred.view(-1) - y_input.squeeze(1)).abs()
                    y_all.append(y_input.squeeze(1).cpu())
                    mae_all.append(mae.cpu())

                y_all = torch.cat(y_all).numpy()
                mae_all = torch.cat(mae_all).numpy()

                df_score_summary = mae_summary_by_score(y_all, mae_all)
                print(df_score_summary)

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
        print(f"Target mean: {targets.mean().item()} +- {targets.std().item()}")
        print(f"Predic mean: {predicts.mean().item()} +- {predicts.std().item()}")

        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(
        self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False, diff=False, ss=False, la=False, graph_train=None, style_src=None
    ):
        print("Training Epoch {}:".format(epoch + 1))

        loss_ls = []
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

                    pred = model(X, stage, self.device)
                    loss = criterion(pred, y.squeeze().float())

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_ls.append(loss.item())
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
            task_loss_ls = []
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()
                # diff first, then task
                # loss = model[0](X, stage, self.device, diff_model=model[1], is_task=False, item_cond=self.item_cond)
                # model[1].zero_grad()
                # loss.backward()
                # _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(), 1.0)
                # optimizer.step()

                task_loss = model[0](X, stage, self.device, diff_model=model[1], is_task=True, item_cond=self.item_cond, style_src=style_src)
                model[1].zero_grad()
                task_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(), 1.0)
                optimizer.step()
                loss = torch.zeros(1, device=task_loss.device)
                loss_ls.append(loss.item())
                task_loss_ls.append(task_loss.item())
            # return torch.tensor(loss_ls).mean()
            return torch.tensor(loss_ls).mean(), torch.tensor(task_loss_ls).mean()

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
        write("=========Diff_CDR========")
        for i in range(self.epoch):
            loss, task_loss = self.train(data_diff, [model, diff_model], None, optimizer, i, stage="train_diff", mapping=False, diff=True)

            mae, rmse = self.eval_mae([model, diff_model], data_test, stage="test_diff")
            self.update_results(mae, rmse, "diff")
            write(f"DIFF LOSS {loss.item()}, TASK LOSS {task_loss.item()}, MAE: {mae} RMSE: {rmse}")

    def Diff_Parallel(self, model, diff_model, data_diff, data_test, optimizer, graph_train, graph_test, style_src, style_tgt_item):
        write("=========Diff_Parallel========")

        diff_model.style_tgt_item = style_tgt_item

        
        src_graph = graph_train.get("src")
        tgt_graph = graph_train.get("tgt")
        shared_graph = graph_train.get("shared")
        # smooth_user_emb_src, _ = self.compute_user_graph_embeddings(model, diff_model, src_graph, use_target=False)
        # smooth_user_emb_tgt, _ = self.compute_user_graph_embeddings(model, diff_model, tgt_graph, use_target=True)
        # diff_model.smooth_user_emb_src = smooth_user_emb_src
        # diff_model.smooth_user_emb_tgt = smooth_user_emb_tgt
        model.graph_shared_train = graph_train.get("shared")
        model.graph_shared_test = graph_test.get("shared")
        model.graph_src = src_graph
        #         graph_data = {
        #     "train": {"src": graph_src_train, "tgt": graph_tgt_train, "shared": graph_shared_train},
        #     "test": {"src": graph_src_test, "tgt": graph_tgt_test, "shared": graph_shared_test},
        # }

        for i in range(self.epoch):
            loss, task_loss = self.train(
                data_diff,
                [model, diff_model],
                None,
                optimizer,
                i,
                stage="train_diff_parallel",
                mapping=False,
                diff=True,
                style_src=style_src,
            )

            mae, rmse = self.eval_mae([model, diff_model], data_test, stage="test_diff_parallel", style_src=style_src)
            self.update_results(mae, rmse, "diff_parallel")
            write(f"DIFF LOSS {loss.item()}, TASK LOSS {task_loss.item()}, MAE: {mae} RMSE: {rmse}")

    def SS_CDR(self, model, ss_model, data_ss, data_test, optimizer_ss):
        write("==========SS_CDR==========")
        for i in range(self.epoch):
            loss = self.train(data_ss, [model, ss_model], None, optimizer_ss, i, stage="train_ss", mapping=False, diff=False, ss=True)
            mae, rmse = self.eval_mae([model, ss_model], data_test, stage="test_ss")
            self.update_results(mae, rmse, "sscdr")
            write("MAE: {} RMSE: {}".format(mae, rmse))

    def LA_CDR(self, model, la_model, data_la, data_test, test_uid, optimizer_la):
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

    def result_print(self, phase):
        print_str = ""
        for p in phase:
            for m in ["_mae", "_rmse"]:
                metric_name = p + m
                print_str += metric_name + ": {:.6f} ".format(self.results[metric_name])
        write(print_str)

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
                w=self.w,
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

        print(f"\n소스 도메인 내 유저의 레이팅 스타일 정보 추출\n")
        cache_path = f"{self.stylecache_root}.pt"
        if os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
            style_src = ckpt["style"]
            info = ckpt["info"]
        else: 
            style_src, info = build_src_user_rating_style_from_loader(
                data_src=data_src, num_users=self.uid_all, rating_min=1.0, rating_max=5.0, device="cpu",
                cache_path=cache_path
            )

        # print("\n소스 유저 percentile-style 추출\n")
        # style_src, info_u = build_src_user_percentile_style_from_loader(
        #     data_src=data_src, num_users=self.uid_all, rating_min=1.0, rating_max=5.0, alpha=0.1, device="cpu"
        # )

        print(f"\n타겟 도메인 내 아이템의 레이팅 스타일 정보 추출\n")
        cache_path = f"{self.stylecache_root}_tgt_item.pt"
        if os.path.exists(cache_path):
            ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)
            style_tgt_item = ckpt["style_tgt_item"]
            info_tgt = ckpt["info_tgt"]
        else:
            style_tgt_item, info_tgt = build_tgt_item_rating_style_from_loader(
                data_tgt=data_tgt, num_items_total=self.iid_all + 1, rating_min=1.0, rating_max=5.0, device="cpu",
                cache_path=cache_path
            )

        # print("\n타겟 아이템 percentile-style 추출\n")
        # style_tgt_item, info_i = build_tgt_item_percentile_style_from_loader(
        #     data_tgt=data_tgt, num_items_total=self.iid_all, rating_min=1.0, rating_max=5.0, alpha=0.1, device="cpu"  # 전역 아이템 개수
        # )

        criterion = torch.nn.MSELoss()

        if exp_part == "None_CDR":
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.SrcOnly(model, data_src, criterion, optimizer_src)
            # CMF
            if self.base_model == "CMF":
                self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
            self.result_print(["tgt", "aug"])
            self.model_save(model, path=save_path)

        #################### BPRMF #######################
        if exp_part == "BPRMF":
            write("========== BPRMF ==========")

            # SRC domain BPR training
            write("--- BPRMF on SRC domain ---")
            for epoch in range(self.epoch):
                loss = self.BPRMF(data_src, model, optimizer_src, stage="src")  # ✅ 위에서 만든 model  # ✅ 공통 optimizer
                write(f"[SRC][Epoch {epoch}] BPR Loss: {loss:.4f}")

            # TGT domain BPR training
            write("--- BPRMF on TGT domain ---")
            for epoch in range(self.epoch):
                loss = self.BPRMF(data_tgt, model, optimizer_tgt, stage="tgt")
                mae, rmse = self.eval_mae(model, data_test, stage="test_tgt")
                self.update_results(mae, rmse, "tgt")
                write(f"[TGT][Epoch {epoch}] " f"BPR Loss: {loss:.4f} | MAE {mae:.4f} RMSE {rmse:.4f}")

            self.result_print(["tgt"])
            self.model_save(model, path=save_path)

        #################### LIGHT GCN #######################
        if exp_part == "LightGCN":
            write("========== LightGCN ==========")

            # -------- SRC domain --------
            write("--- LightGCN on SRC domain ---")
            for epoch in range(self.epoch):
                loss = self.LightGCN_BPR(model, graph_data["train"]["src"], optimizer_src, stage="src", num_layers=2)
                write(f"[SRC][Epoch {epoch}] LightGCN BPR Loss: {loss:.4f}")

            # -------- TGT domain --------
            write("--- LightGCN on TGT domain ---")
            for epoch in range(self.epoch):
                loss = self.LightGCN_BPR(model, graph_data["train"]["tgt"], optimizer_tgt, stage="tgt", num_layers=2)
                mae, rmse = self.eval_mae(model, data_test, stage="test_tgt")
                self.update_results(mae, rmse, "tgt")
                write(f"[TGT][Epoch {epoch}] " f"LightGCN BPR Loss: {loss:.4f} | MAE {mae:.4f} RMSE {rmse:.4f}")

            self.result_print(["tgt"])
            self.model_save(model, path=save_path)

        elif exp_part == "CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.CDR(model, data_map, data_meta, data_test, criterion, optimizer_map, optimizer_meta)
            self.result_print(["emcdr", "ptupcdr"])

        elif exp_part == "ss_CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.SS_CDR(model, ss_model, data_ss, data_test, optimizer_ss)
            self.result_print(["sscdr"])

        elif exp_part == "la_CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.LA_CDR(model, la_model, data_la, data_test, optimizer_la)
            self.result_print(["lacdr"])

        elif exp_part == "diff_CDR":
            self.model_load(model, path=save_path)
            print("None_CDR model loaded")
            self.Diff_CDR(model, diff_model, data_diff, data_diff_test, optimizer_diff)
            self.result_print(["diff"])

        elif exp_part == "diff_parallel":
            self.model_load(model, path=save_path)
            # model.build_user_prototype_cache(self.device, 0.5, 0.5, user_batch=1024)
            print("None_CDR model loaded")
            # optimizer_diff: DiffParallel 의 파라미터만 포함, model에 있는 user/item embedding update X
            self.Diff_Parallel(
                model, diff_model, data_diff, data_diff_test, optimizer_diff, graph_data["train"], graph_data["test"], style_src, style_tgt_item
            )
            self.result_print(["diff_parallel"])


def mae_hist_by_score(y_true: np.ndarray, mae: np.ndarray, mae_bins=None):  # (N,) ground-truth score, e.g. 1~5  # (N,) |pred - y|
    """
    score별 MAE bin 분포를 table로 반환
    """
    y_true = np.asarray(y_true).reshape(-1)
    mae = np.asarray(mae).reshape(-1)

    assert len(y_true) == len(mae)

    if mae_bins is None:
        # 기본: 0~3까지 0.05 간격 (필요하면 조절)
        mae_bins = np.arange(0.0, 3.01, 0.05)

    rows = []

    for score in sorted(np.unique(y_true)):
        mask = y_true == score
        mae_s = mae[mask]

        counts, edges = np.histogram(mae_s, bins=mae_bins)

        for i in range(len(counts)):
            if counts[i] == 0:
                continue
            rows.append(
                {
                    "score": int(score),
                    "mae_min": edges[i],
                    "mae_max": edges[i + 1],
                    "count": int(counts[i]),
                }
            )

    df = pd.DataFrame(rows)
    return df


def mae_summary_by_score(y_true, mae):
    rows = []
    for s in sorted(np.unique(y_true)):
        m = mae[y_true == s]
        rows.append(
            {
                "score": int(s),
                "count": len(m),
                "mae_mean": m.mean(),
                "mae_median": np.median(m),
                "mae_q75": np.quantile(m, 0.75),
                "mae_q90": np.quantile(m, 0.90),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def build_src_user_rating_style_from_loader(
    data_src,
    num_users: int,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    device: str = "cpu",
    cache_path = ""
):
    """
    data_src yields: (X, y)
      - X: torch.Size([B, 2]) where X[:,0]=uid, X[:,1]=iid
      - y: torch.Size([B, 1]) rating

    Returns
    -------
    style: FloatTensor [num_users, 9]
      style[u] = [mean, var, std, min, max, cnt, frac_min, frac_max, frac_extreme]
    info: dict
    """
    # 누적 통계 (double로 안전하게)
    sums = torch.zeros(num_users, dtype=torch.float64)
    sumsqs = torch.zeros(num_users, dtype=torch.float64)
    cnts = torch.zeros(num_users, dtype=torch.float64)

    mins = torch.full((num_users,), float("inf"), dtype=torch.float64)
    maxs = torch.full((num_users,), float("-inf"), dtype=torch.float64)

    cnt_min = torch.zeros(num_users, dtype=torch.float64)
    cnt_max = torch.zeros(num_users, dtype=torch.float64)

    for X, y in data_src:
        # X: [B,2], y: [B,1]
        uid = X[:, 0].detach().to("cpu").long().view(-1)  # [B]
        r = y.detach().to("cpu").double().view(-1)  # [B]

        if uid.numel() != r.numel():
            raise ValueError(f"uid/rating mismatch: uid={uid.shape}, r={r.shape}")

        # sum, sumsq, count
        sums.index_add_(0, uid, r)
        sumsqs.index_add_(0, uid, r * r)
        cnts.index_add_(0, uid, torch.ones_like(r, dtype=torch.float64))

        # min/max (유저별 배치 내부 min/max로 갱신)
        uniq = torch.unique(uid)
        for u in uniq.tolist():
            mask = uid == u
            rv = r[mask]
            mins[u] = torch.minimum(mins[u], rv.min())
            maxs[u] = torch.maximum(maxs[u], rv.max())

        # extreme counts
        is_min = r <= rating_min + 1e-12
        is_max = r >= rating_max - 1e-12
        cnt_min.index_add_(0, uid, is_min.double())
        cnt_max.index_add_(0, uid, is_max.double())

    # 통계 계산
    eps = 1e-12
    cnt_safe = torch.clamp(cnts, min=1.0)

    mean = sums / cnt_safe
    ex2 = sumsqs / cnt_safe
    var = torch.clamp(ex2 - mean * mean, min=0.0)
    std = torch.sqrt(var + eps)

    rmin = torch.where(cnts > 0, mins, torch.zeros_like(mins))
    rmax = torch.where(cnts > 0, maxs, torch.zeros_like(maxs))

    frac_min = cnt_min / cnt_safe
    frac_max = cnt_max / cnt_safe
    frac_extreme = (cnt_min + cnt_max) / cnt_safe

    # [U, 9]
    style = torch.stack([mean, var, std, rmin, rmax, cnts, frac_min, frac_max, frac_extreme], dim=1).to(torch.float32).to(device)

    info = {
        "feature_names": ["mean", "var", "std", "min", "max", "cnt", "frac_min", "frac_max", "frac_extreme"],
        "rating_min": rating_min,
        "rating_max": rating_max,
        "num_users": num_users,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "style": style.cpu(),   # 저장은 CPU 권장
            "info": info,
        },
        cache_path,
    )

    return style, info


@torch.no_grad()
def build_tgt_item_rating_style_from_loader(
    data_tgt,
    num_items_total: int,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    device: str = "cpu",
    cache_path="",
):
    """
    data_tgt yields: (X, y)
      - X: torch.Size([B, 2]) where X[:,0]=uid, X[:,1]=iid (GLOBAL item index)
      - y: torch.Size([B, 1]) rating

    Returns
    -------
    style_item: FloatTensor [num_items_total, 9]
      style_item[i] = [mean, var, std, min, max, cnt, frac_min, frac_max, frac_extreme]
      - only target items that appear in data_tgt will have non-zero stats.
    info: dict
    """
    # 누적 통계 (double로 안전하게)
    sums = torch.zeros(num_items_total, dtype=torch.float64)
    sumsqs = torch.zeros(num_items_total, dtype=torch.float64)
    cnts = torch.zeros(num_items_total, dtype=torch.float64)

    mins = torch.full((num_items_total,), float("inf"), dtype=torch.float64)
    maxs = torch.full((num_items_total,), float("-inf"), dtype=torch.float64)

    cnt_min = torch.zeros(num_items_total, dtype=torch.float64)
    cnt_max = torch.zeros(num_items_total, dtype=torch.float64)

    for X, y in data_tgt:
        # X: [B,2], y: [B,1]
        iid = X[:, 1].detach().to("cpu").long().view(-1)  # [B]
        r = y.detach().to("cpu").double().view(-1)  # [B]

        if iid.numel() != r.numel():
            raise ValueError(f"iid/rating mismatch: iid={iid.shape}, r={r.shape}")

        # 안전 체크: 전역 iid가 num_items_total 범위 안인지
        if iid.numel() > 0:
            if iid.min().item() < 0 or iid.max().item() >= num_items_total:
                raise ValueError(f"iid out of range: min={iid.min().item()}, max={iid.max().item()}, " f"num_items_total={num_items_total}")

        # sum, sumsq, count
        sums.index_add_(0, iid, r)
        sumsqs.index_add_(0, iid, r * r)
        cnts.index_add_(0, iid, torch.ones_like(r, dtype=torch.float64))

        # min/max (아이템별 배치 내부 min/max로 갱신)
        uniq = torch.unique(iid)
        for it in uniq.tolist():
            mask = iid == it
            rv = r[mask]
            mins[it] = torch.minimum(mins[it], rv.min())
            maxs[it] = torch.maximum(maxs[it], rv.max())

        # extreme counts
        is_min = r <= rating_min + 1e-12
        is_max = r >= rating_max - 1e-12
        cnt_min.index_add_(0, iid, is_min.double())
        cnt_max.index_add_(0, iid, is_max.double())

    # 통계 계산
    eps = 1e-12
    cnt_safe = torch.clamp(cnts, min=1.0)

    mean = sums / cnt_safe
    ex2 = sumsqs / cnt_safe
    var = torch.clamp(ex2 - mean * mean, min=0.0)
    std = torch.sqrt(var + eps)

    rmin = torch.where(cnts > 0, mins, torch.zeros_like(mins))
    rmax = torch.where(cnts > 0, maxs, torch.zeros_like(maxs))

    frac_min = cnt_min / cnt_safe
    frac_max = cnt_max / cnt_safe
    frac_extreme = (cnt_min + cnt_max) / cnt_safe

    # [I_total, 9]
    style_item = torch.stack([mean, var, std, rmin, rmax, cnts, frac_min, frac_max, frac_extreme], dim=1).to(torch.float32).to(device)

    info = {
        "feature_names": ["mean", "var", "std", "min", "max", "cnt", "frac_min", "frac_max", "frac_extreme"],
        "rating_min": rating_min,
        "rating_max": rating_max,
        "num_items_total": num_items_total,
    }

    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "style_tgt_item": style_item.cpu(),   # 저장은 CPU 권장
            "info_tgt": info,
        },
        cache_path,
    )

    return style_item, info


@torch.no_grad()
def build_src_user_percentile_style_from_loader(
    data_src,
    num_users: int,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    device: str = "cpu",
    alpha: float = 0.1,  # low/high 구간(예: 하위 10%, 상위 10%)
):
    """
    Percentile-style features per user computed from rating histograms.

    Returns
    -------
    style: FloatTensor [num_users, 9]
      style[u] = [p_mean, p_var, p_std, p_min, p_max, cnt,
                  frac_low, frac_high, frac_extreme]
      - p_* are computed on percentile values in [0,1]
    info: dict
    """
    # ---- rating levels: assumes integer levels in [rating_min, rating_max]
    # (너 데이터가 1~5 정수 평점이라는 전제. half-step이면 levels를 바꿔야 함)
    levels = torch.arange(int(rating_min), int(rating_max) + 1, dtype=torch.long)  # [K]
    K = levels.numel()

    # per-user histogram counts: [U, K]
    hist = torch.zeros((num_users, K), dtype=torch.float64)

    for X, y in data_src:
        uid = X[:, 0].detach().to("cpu").long().view(-1)  # [B]
        r = y.detach().to("cpu").view(-1)  # [B]

        if uid.numel() != r.numel():
            raise ValueError(f"uid/rating mismatch: uid={uid.shape}, r={r.shape}")

        # rating을 레벨 인덱스로 변환 (정수 평점 전제)
        r_int = r.round().long()  # 혹시 float여도 1~5 근처면 반올림
        if r_int.numel() > 0:
            if r_int.min().item() < levels.min().item() or r_int.max().item() > levels.max().item():
                raise ValueError(f"rating out of level range: min={r_int.min().item()}, max={r_int.max().item()}")

        ridx = (r_int - levels.min()).clamp(0, K - 1)  # [B]

        # scatter-add로 히스토그램 누적
        # hist[uid, ridx] += 1
        hist.index_put_((uid, ridx), torch.ones_like(ridx, dtype=torch.float64), accumulate=True)

    # ---- counts per user
    cnts = hist.sum(dim=1)  # [U]
    cnt_safe = torch.clamp(cnts, min=1.0)

    # ---- build percentile per level per user (mid-rank)
    # cum_less: cumulative counts strictly less than level k
    cum = torch.cumsum(hist, dim=1)  # [U,K] cumulative including self
    cum_less = cum - hist  # [U,K]
    p_level = (cum_less + 0.5 * hist) / cnt_safe.unsqueeze(1)  # [U,K] in [0,1]

    # ---- stats on percentile values, weighted by hist
    # mean_p = sum_k count_k * p_k / N
    mean_p = (hist * p_level).sum(dim=1) / cnt_safe  # [U]
    ex2_p = (hist * (p_level**2)).sum(dim=1) / cnt_safe  # [U]
    var_p = torch.clamp(ex2_p - mean_p * mean_p, min=0.0)  # [U]
    std_p = torch.sqrt(var_p + 1e-12)  # [U]

    # p_min/p_max: among levels that exist
    has = hist > 0
    pmin = torch.where(has, p_level, torch.full_like(p_level, float("inf"))).min(dim=1).values
    pmax = torch.where(has, p_level, torch.full_like(p_level, float("-inf"))).max(dim=1).values
    pmin = torch.where(cnts > 0, pmin, torch.zeros_like(pmin))
    pmax = torch.where(cnts > 0, pmax, torch.zeros_like(pmax))

    # low/high/extreme fractions based on percentile thresholds
    low_mask = p_level <= alpha
    high_mask = p_level >= (1.0 - alpha)

    frac_low = (hist * low_mask.double()).sum(dim=1) / cnt_safe
    frac_high = (hist * high_mask.double()).sum(dim=1) / cnt_safe
    frac_extreme = frac_low + frac_high

    # [U, 9]
    style = torch.stack([mean_p, var_p, std_p, pmin, pmax, cnts, frac_low, frac_high, frac_extreme], dim=1).to(torch.float32).to(device)

    info = {
        "feature_names": ["p_mean", "p_var", "p_std", "p_min", "p_max", "cnt", "frac_low", "frac_high", "frac_extreme"],
        "alpha": alpha,
        "rating_min": rating_min,
        "rating_max": rating_max,
        "levels": levels.tolist(),
        "num_users": num_users,
        "note": "percentile is computed within each user's rating distribution (mid-rank).",
    }
    return style, info


@torch.no_grad()
def build_tgt_item_percentile_style_from_loader(
    data_tgt,
    num_items_total: int,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    device: str = "cpu",
    alpha: float = 0.1,
):
    """
    Percentile-style features per item computed from rating histograms.

    Returns
    -------
    style_item: FloatTensor [num_items_total, 9]
      style_item[i] = [p_mean, p_var, p_std, p_min, p_max, cnt,
                       frac_low, frac_high, frac_extreme]
      - percentile is computed within each item's rating distribution (across users)
    info: dict
    """
    levels = torch.arange(int(rating_min), int(rating_max) + 1, dtype=torch.long)  # [K]
    K = levels.numel()

    hist = torch.zeros((num_items_total, K), dtype=torch.float64)

    for X, y in data_tgt:
        iid = X[:, 1].detach().to("cpu").long().view(-1)  # [B] global iid
        r = y.detach().to("cpu").view(-1)  # [B]

        if iid.numel() != r.numel():
            raise ValueError(f"iid/rating mismatch: iid={iid.shape}, r={r.shape}")

        if iid.numel() > 0:
            if iid.min().item() < 0 or iid.max().item() >= num_items_total:
                raise ValueError(f"iid out of range: min={iid.min().item()}, max={iid.max().item()}, num_items_total={num_items_total}")

        r_int = r.round().long()
        if r_int.numel() > 0:
            if r_int.min().item() < levels.min().item() or r_int.max().item() > levels.max().item():
                raise ValueError(f"rating out of level range: min={r_int.min().item()}, max={r_int.max().item()}")

        ridx = (r_int - levels.min()).clamp(0, K - 1)

        hist.index_put_((iid, ridx), torch.ones_like(ridx, dtype=torch.float64), accumulate=True)

    cnts = hist.sum(dim=1)
    cnt_safe = torch.clamp(cnts, min=1.0)

    cum = torch.cumsum(hist, dim=1)
    cum_less = cum - hist
    p_level = (cum_less + 0.5 * hist) / cnt_safe.unsqueeze(1)

    mean_p = (hist * p_level).sum(dim=1) / cnt_safe
    ex2_p = (hist * (p_level**2)).sum(dim=1) / cnt_safe
    var_p = torch.clamp(ex2_p - mean_p * mean_p, min=0.0)
    std_p = torch.sqrt(var_p + 1e-12)

    has = hist > 0
    pmin = torch.where(has, p_level, torch.full_like(p_level, float("inf"))).min(dim=1).values
    pmax = torch.where(has, p_level, torch.full_like(p_level, float("-inf"))).max(dim=1).values
    pmin = torch.where(cnts > 0, pmin, torch.zeros_like(pmin))
    pmax = torch.where(cnts > 0, pmax, torch.zeros_like(pmax))

    low_mask = p_level <= alpha
    high_mask = p_level >= (1.0 - alpha)

    frac_low = (hist * low_mask.double()).sum(dim=1) / cnt_safe
    frac_high = (hist * high_mask.double()).sum(dim=1) / cnt_safe
    frac_extreme = frac_low + frac_high

    style_item = torch.stack([mean_p, var_p, std_p, pmin, pmax, cnts, frac_low, frac_high, frac_extreme], dim=1).to(torch.float32).to(device)

    info = {
        "feature_names": ["p_mean", "p_var", "p_std", "p_min", "p_max", "cnt", "frac_low", "frac_high", "frac_extreme"],
        "alpha": alpha,
        "rating_min": rating_min,
        "rating_max": rating_max,
        "levels": levels.tolist(),
        "num_items_total": num_items_total,
        "note": "percentile is computed within each item's rating distribution (mid-rank).",
    }
    return style_item, info
