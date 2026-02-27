import numpy as np
import os
import sys
import logging
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List


@torch.no_grad()
def build_user_knn_graph_tfidf_cosine_large(
    graph_data: Dict[str, torch.Tensor],
    k: int = 20,
    max_df_users: int = 300,  # df가 이 값 초과면 인기템으로 보고 제거
    topL_per_user: int = 80,  # 유저당 TF-IDF 상위 L개만 유지
    min_sim: float = 0.0,  # 너무 약한 유사도는 제거
    use_tf_values: bool = False,  # uv_adj 값이 count면 True, 아니면 False(TF=1)
    tf_log1p: bool = True,
    idf_smooth: bool = True,  # idf = log((U+1)/(df+1))+1
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """
    큰 U,I에서도 돌아가도록:
    - 인기 아이템 제거(df>max_df_users)
    - 유저별 top-L TF-IDF만 남김
    - '희소 아이템을 공유하는 유저들'만 후보로 만들어 dot 누적 -> kNN

    출력: graph_data에 uu_adj(sparse UxU, values=cosine approx) 추가
    """
    assert "uv_adj" in graph_data
    uv = graph_data["uv_adj"].coalesce()
    if device is None:
        device = uv.device
    else:
        device = torch.device(device)
    uv = uv.to(device)

    U, I = uv.shape
    idx = uv.indices()
    vals = uv.values()
    u_idx = idx[0]
    i_idx = idx[1]

    # -------- 1) item df 계산 (각 아이템을 소비한 유저 수) --------
    # uv가 binary든 count든 df는 "연결 존재"만 보면 됨
    ones = torch.ones_like(vals, dtype=torch.float32)
    uv_bin = torch.sparse_coo_tensor(idx, ones, size=(U, I), device=device).coalesce()
    item_df = torch.sparse.sum(uv_bin, dim=0).to_dense().long()  # (I,)

    # 인기 아이템 제거 마스크
    keep_item = item_df <= max_df_users
    keep_item_idx = torch.where(keep_item)[0]  # 남길 아이템들 (1D)

    # idf
    df_f = item_df.float()
    if idf_smooth:
        idf = torch.log((U + 1.0) / (df_f + 1.0)) + 1.0
    else:
        idf = torch.log(U / torch.clamp(df_f, min=1.0))

    # -------- 2) edge 필터: 인기 아이템 제거 --------
    mask_edge = keep_item[i_idx]
    u_idx = u_idx[mask_edge]
    i_idx = i_idx[mask_edge]
    vals = vals[mask_edge]

    # -------- 3) TF-IDF weight 계산 --------
    if use_tf_values:
        tf = vals.float()
        if tf_log1p:
            tf = torch.log1p(tf)
    else:
        tf = torch.ones_like(vals, dtype=torch.float32)
    w = tf * idf[i_idx]  # (nnz_filtered,)

    # -------- 4) 유저별 top-L TF-IDF만 남기기 --------
    # 유저별로 (item, w) 모아서 topL만 유지
    # (큰 규모에서도 단순/안전하게 파이썬 리스트로 모으는 방식)
    user_items: List[List[int]] = [[] for _ in range(U)]
    user_weights: List[List[float]] = [[] for _ in range(U)]

    # u_idx, i_idx, w 는 같은 길이
    for uu, ii, ww in zip(u_idx.tolist(), i_idx.tolist(), w.tolist()):
        user_items[uu].append(ii)
        user_weights[uu].append(float(ww))

    # topL 적용 + L2 normalize(코사인 위해)
    # normalize는 '유저 TF-IDF 벡터'의 L2 norm으로 나눔
    for uu in range(U):
        if len(user_items[uu]) == 0:
            continue
        # topL
        if len(user_items[uu]) > topL_per_user:
            # 큰 값 상위 L개만
            pairs = list(zip(user_items[uu], user_weights[uu]))
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:topL_per_user]
            user_items[uu] = [p[0] for p in pairs]
            user_weights[uu] = [p[1] for p in pairs]

        # L2 normalize
        norm = 0.0
        for ww in user_weights[uu]:
            norm += ww * ww
        norm = norm**0.5
        if norm <= 1e-12:
            continue
        user_weights[uu] = [ww / norm for ww in user_weights[uu]]

    # -------- 5) item -> (users, weights) inverted index 만들기 --------
    # 남은 (user, item)들로만 구성됨
    item_users: List[List[int]] = [[] for _ in range(I)]
    item_uw: List[List[float]] = [[] for _ in range(I)]
    for uu in range(U):
        items = user_items[uu]
        ws = user_weights[uu]
        for ii, ww in zip(items, ws):
            item_users[ii].append(uu)
            item_uw[ii].append(ww)

    # -------- 6) 유저별 이웃 점수 누적 (희소 아이템 공유하는 쌍만) --------
    # score(u,v) += w(u,i) * w(v,i)  (이미 L2 normalize 했으니 dot=cosine)
    # 이후 u별 top-k만 남김
    neigh_scores: List[Dict[int, float]] = [dict() for _ in range(U)]

    for ii in range(I):
        us = item_users[ii]
        ws = item_uw[ii]
        m = len(us)
        if m <= 1:
            continue

        # 같은 아이템 ii를 공유한 유저들끼리만 pair 생성
        # m이 크면 느릴 수 있는데, 우리는 df>max_df_users로 인기템을 제거했기 때문에 m이 작다고 기대함
        for a in range(m):
            u = us[a]
            wu = ws[a]
            d = neigh_scores[u]
            for b in range(m):
                if a == b:
                    continue
                v = us[b]
                d[v] = d.get(v, 0.0) + wu * ws[b]

    # -------- 7) 유저별 top-k 추출 -> sparse uu_adj 생성 --------
    src_list, dst_list, val_list = [], [], []
    for u in range(U):
        d = neigh_scores[u]
        if not d:
            continue
        # min_sim 필터
        if min_sim > 0.0:
            items = [(v, s) for v, s in d.items() if s >= min_sim]
        else:
            items = list(d.items())
        if not items:
            continue
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:k]

        for v, s in items:
            src_list.append(u)
            dst_list.append(v)
            # val_list.append(float(s))
            val_list.append(1.0)

    if len(src_list) == 0:
        uu_adj = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float32, device=device), size=(U, U), device=device
        ).coalesce()
    else:
        uu_idx = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        uu_val = torch.tensor(val_list, dtype=torch.float32, device=device)
        uu_adj = torch.sparse_coo_tensor(uu_idx, uu_val, size=(U, U), device=device).coalesce()

    out = dict(graph_data)
    out["uu_adj"] = uu_adj
    out["uu_adj_T"] = uu_adj.transpose(0, 1).coalesce()
    return out


@torch.no_grad()
def posthoc_user_residual_aggregate_from_uu(
    uu_adj: torch.Tensor,  # sparse (U,U)
    user_feat: torch.Tensor,  # (U,d) MF 유저 임베딩
    alpha: float = 0.3,  # residual mixing
    normalize: str = "row_sum",  # weighted mean
):
    uu_adj = uu_adj.coalesce().to(user_feat.device)

    neigh_msg = torch.sparse.mm(uu_adj, user_feat)  # (U,d)

    if normalize == "row_sum":
        deg = torch.sparse.sum(uu_adj, dim=1).to_dense().unsqueeze(1)  # (U,1)
        deg = torch.where(deg == 0, torch.ones_like(deg), deg)
        neigh_msg = neigh_msg / deg

    out = (1.0 - alpha) * user_feat + alpha * neigh_msg

    # 이웃이 전혀 없는 유저는 neigh_msg가 0쪽으로 가기 쉬우니, 원본 유지
    deg0 = torch.sparse.sum(uu_adj, dim=1).to_dense() == 0
    out[deg0] = user_feat[deg0]

    return out


def get_parent_curr_dir():
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    sys.path.append(parent_dir)
    current_dir = os.path.dirname(current_file_path)
    return parent_dir, current_dir


def make_dir(log_name):
    parent_dir, current_dir = get_parent_curr_dir()
    save_dir = os.path.join(current_dir, "logs")
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, f"{log_name}_log.txt")
    return logfile


def get_save_name(args, pretrain_dataset_names, save_dir, result_dir):
    pretrain_dataset_str = ""
    for strs in pretrain_dataset_names:
        pretrain_dataset_str += "_" + strs
    # set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
    # set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.ablation_pre}_{args.sample_size}_{args.nb_epochs}_{args.de_loss}_{args.de_weight}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'

    set_name = f"model_node_{args.pretrain_method}_{pretrain_dataset_str}_{args.ablation_pre}_{args.sample_size}_{args.nb_epochs}_{args.if_rand}_{args.w1loss}_{args.de_loss}_{args.de_weight}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}"

    save_name = os.path.join(save_dir, f"{set_name}.pkl")
    csv_name = os.path.join(result_dir, f"{set_name}.csv")

    return save_name, csv_name


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    elif v.lower() in ("false", "0", "no", "n"):
        return False


def write(txt="\n"):
    print(txt)
    logging.info(txt)


def log_args_table(args, max_per_line: int = 5, col_width: int = 30):
    """
    args: argparse.Namespace
    max_per_line: 한 줄에 몇 개 출력할지
    col_width: 각 열의 고정 폭
    """
    args_dict = vars(args)
    arg_items = [f"{k} = {v}" for k, v in sorted(args_dict.items())]

    # 패딩을 넣어 고정 길이 문자열로 변환
    padded_items = [item.ljust(col_width) for item in arg_items]

    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))
    logging.info("Arguments:")

    for i in range(0, len(padded_items), max_per_line):
        row = padded_items[i : i + max_per_line]
        logging.info(" | ".join(row))

    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.v = nn.Linear(in_dim, out_dim, bias=False)
        self.scale = in_dim**-0.5

    def forward(self, x, mask=None, query=None):
        """
        x: (B, T, D)
        mask: (B, T) or None
        """
        if query is not None:
            Q = self.q(query)
        else:
            Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, T, T)
        if mask is not None:
            score = score.masked_fill(mask[:, None, :] == 0, -1e9)

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, V)  # (B, T, D)
        return out


class SimilarityProjector(nn.Module):
    def __init__(self, out_dim=10):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, iid_emb, trans_emb_m):
        # cosine similarity (B, 1)
        sim = F.cosine_similarity(iid_emb, trans_emb_m, dim=1, eps=1e-8).unsqueeze(1)
        # (B, 10)
        return self.proj(sim)
