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

        self.use_vbge = bool(config.get("use_vbge", 1))
        self.vbge_opt = {
            "GNN": config["vbge_GNN"],  # num of layers
            "dropout": config["vbge_drouout"],
            "feature_dim": self.emb_dim,  # MF 임베딩 차원 사용
            "hidden_dim": config["vbge_hidden_dim"],
            "leakey": config["vbge_leakey"],
        }
        self.parallel_setting = {
            "set_loss": config["set_loss"],
            "set_init": config["set_init"],
            "set_proj": config["set_proj"],
            "set_aggr": config["set_aggr"],
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
        frames = [] # src train + tgt train 
        for p in paths:
            df = pd.read_csv(p, header=None, usecols=[1])
            df.columns = ["iid"]
            frames.append(df)
        if not frames:
            return None
        counts = pd.concat(frames, ignore_index=True)["iid"].value_counts() # 각 iid 마다 등장 횟수 카운팅
        full_counts = counts.reindex(range(self.iid_all + 1), fill_value=0).to_numpy() # iid 번호 순서대로 정렬 

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
        MAX_POS = 20 # history 길이 제한 

        # 🔑 핵심: user 단위로 그룹핑 -> test_user 별로 pos_seq 한번씩만 추가 
        for uid, group in interactions.groupby("uid"):
            pos_seq = ast.literal_eval(group.iloc[0]["pos_seq"]) # pos_seq는 user별로 모두 동일하므로 첫 row만 사용

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
        uv_adj = torch.sparse_coo_tensor(
            uv_indices,
            edge_values,
            size=(self.uid_all, self.iid_all + 1)
        ).coalesce()

        vu_indices = torch.stack([item_ids, user_ids])
        vu_adj = torch.sparse_coo_tensor(
            vu_indices,
            edge_values,
            size=(self.iid_all + 1, self.uid_all)
        ).coalesce()

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

        graph_src_train = self.build_graph_inputs(self.src_path, exclude_users=test_users)
        graph_tgt_train = self.build_graph_inputs(self.tgt_path, exclude_users=test_users)

        graph_src_test = self.build_test_graph_inputs(self.test_path) # src_path 말고 test_path에서 로드, pos_seq와 그래프 생성
        graph_tgt_test = graph_src_test # tgt_test는 안 쓰임 

        graph_shared_train = self.build_shared_train_graph(self.src_path, self.tgt_path, exclude_users=test_users)
        graph_shared_test = self.build_shared_test_graph(self.test_path)

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

        if self.use_vbge:  # VBGE로 agg
            gnn_attr = "t_gnn" if use_target else "s_gnn"
            gnn_module = getattr(diff_model, gnn_attr, None)
            if gnn_module is None:
                return None, None

            uv_adj = graph_data["uv_adj"].to(self.device)
            vu_adj = graph_data["vu_adj"].to(self.device)

            if use_target:
                user_feat = base_model.tgt_model.uid_embedding.weight.detach().to(self.device)
                item_feat = base_model.tgt_model.iid_embedding.weight.detach().to(self.device)
            else:
                user_feat = base_model.src_model.uid_embedding.weight.detach().to(self.device)
                item_feat = base_model.src_model.iid_embedding.weight.detach().to(self.device)

            was_training = gnn_module.training
            # gnn_module.eval()
            # with torch.no_grad():
            user_emb, item_emb = gnn_module(user_feat, item_feat, uv_adj, vu_adj)
            # if was_training:
            # gnn_module.train()
            return user_emb, item_emb
        else:  # 단순 2홉 aggr
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

    def compute_item_aggregation_popularity(self, base_model, graph_data):
        uv_adj = graph_data["uv_adj"].to(self.device)  # [num_users, num_items]

        # item embedding 선택
        item_feat = base_model.src_model.iid_embedding.weight.detach().to(self.device)

        # item popularity (assumed shape: [num_items])
        conf_weight = self.item_popularity.to(self.device).unsqueeze(1)  # [num_items, 1]
        int_weight = torch.ones_like(conf_weight)-conf_weight

        
        
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

    def compute_global_graph_embeddings(self, base_model, graph_data_shared):
        """Aggregate on shared graph with two sequential 1-hop steps; split src/tgt item embeddings."""
        if graph_data_shared is None:
            return None, None, None

        uv_adj = graph_data_shared["uv_adj"].to(self.device)
        vu_adj = graph_data_shared["vu_adj"].to(self.device)

        # prefer pre-computed split; fall back to even split if missing
        num_items_total = uv_adj.shape[1]

        with torch.no_grad():
            # use averaged user embedding as seed
            user_feat_src = base_model.src_model.uid_embedding.weight.detach().to(self.device)
            user_feat_tgt = base_model.tgt_model.uid_embedding.weight.detach().to(self.device)
            user_emb = (user_feat_src + user_feat_tgt) / 2

            # iterative 1-hop aggregation twice
            for _ in range(2):
                item_emb = torch.sparse.mm(vu_adj, user_emb)  # users -> items
                user_emb = torch.sparse.mm(uv_adj, item_emb)  # items -> users

        return user_emb, item_emb

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
            optimizer_diff = torch.optim.Adam(params=diff_model.parameters(), lr=self.diff_lr)
            return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_diff, optimizer_map

    def eval_mae(self, model, data_loader, stage, graph_test=None):
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
                src_graph = graph_test.get("src")
                tgt_graph = graph_test.get("tgt")
                shared_graph = graph_test.get("shared")
                smooth_user_emb_src, _ = self.compute_user_graph_embeddings(model[0], model[1], src_graph, use_target=False)
                smooth_user_emb_tgt, _ = self.compute_user_graph_embeddings(model[0], model[1], tgt_graph, use_target=True)
                global_user_emb, global_item_emb = self.compute_global_graph_embeddings(model[0], shared_graph)
                conf_item_aggr, int_item_aggr = self.compute_item_aggregation_popularity(model[0], shared_graph)
                model[1].smooth_user_emb_src = smooth_user_emb_src
                model[1].smooth_user_emb_tgt = smooth_user_emb_tgt
                model[1].smooth_user_emb_global = global_user_emb
                model[1].smooth_item_emb = global_item_emb
                model[1].item_popularity = self.item_popularity
                model[1].conf_item_aggr = conf_item_aggr
                model[1].int_item_aggr = int_item_aggr

                for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    model[0].eval()
                    model[1].eval()
                    pred = model[0](X, stage, self.device, diff_model=model[1])
                    y_input = X[-1]
                    targets.extend(y_input.squeeze(1).tolist())
                    predicts.extend(pred.tolist())

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

        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False, diff=False, ss=False, la=False, graph_train=None):
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
            if graph_train is not None:  # DiffParallel else DiffCDR
                src_graph = graph_train.get("src")
                tgt_graph = graph_train.get("tgt")
                shared_graph = graph_train.get("shared")
                smooth_user_emb_src, _ = self.compute_user_graph_embeddings(model[0], model[1], src_graph, use_target=False)
                smooth_user_emb_tgt, _ = self.compute_user_graph_embeddings(model[0], model[1], tgt_graph, use_target=True)
                global_user_emb, global_item_emb = self.compute_global_graph_embeddings(model[0], shared_graph)
                conf_item_aggr, int_item_aggr = self.compute_item_aggregation_popularity(model[0], shared_graph)
                model[1].smooth_user_emb_src = smooth_user_emb_src
                model[1].smooth_user_emb_tgt = smooth_user_emb_tgt
                model[1].smooth_user_emb_global = global_user_emb
                model[1].smooth_item_emb = global_item_emb
                model[1].item_popularity = self.item_popularity
                model[1].conf_item_aggr = conf_item_aggr 
                model[1].int_item_aggr = int_item_aggr 
            task_loss_ls = []
            for X in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                model[1].train()
                # diff first, then task
                loss = model[0](X, stage, self.device, diff_model=model[1], is_task=False)
                model[1].zero_grad()
                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(), 1.0)
                optimizer.step()

                task_loss = model[0](X, stage, self.device, diff_model=model[1], is_task=True)
                model[1].zero_grad()
                task_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model[1].parameters(), 1.0)
                optimizer.step()

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

    def Diff_Parallel(self, model, diff_model, data_diff, data_test, optimizer, graph_train, graph_test):
        write("=========Diff_Parallel========")
        for i in range(self.epoch):
            loss, task_loss = self.train(
                data_diff, [model, diff_model], None, optimizer, i, stage="train_diff_parallel", mapping=False, diff=True, graph_train=graph_train
            )

            mae, rmse = self.eval_mae([model, diff_model], data_test, stage="test_diff_parallel", graph_test=graph_test)
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
            model.load_state_dict(torch.load(path))
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
                self.vbge_opt,
                use_vbge=self.use_vbge,
                parallel=self.parallel_setting,
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

        criterion = torch.nn.MSELoss()

        if exp_part == "None_CDR":
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.SrcOnly(model, data_src, criterion, optimizer_src)
            # CMF
            if self.base_model == "CMF":
                self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
            self.result_print(["tgt", "aug"])
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
            print("None_CDR model loaded")
            self.Diff_Parallel(model, diff_model, data_diff, data_diff_test, optimizer_diff, graph_data["train"], graph_data["test"])
            self.result_print(["diff_parallel"])
