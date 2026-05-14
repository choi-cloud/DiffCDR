import numpy as np
import os
import sys
import logging
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from collections import Counter
from collections import defaultdict


def mae_rmse_summary_by_pop_group(y_true, y_pred, user_uid, test_users_pop_group):
    """
    Args:
        y_true: torch.Tensor or np.ndarray, shape [N]
        y_pred: torch.Tensor or np.ndarray, shape [N]
        user_uid: torch.Tensor or np.ndarray, shape [N]
        test_users_pop_group: dict
            {
                "0.0~0.2": [uid1, uid2, ...],
                "0.2~0.4": [...],
                ...
            }

    Returns:
        pd.DataFrame
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(user_uid, torch.Tensor):
        user_uid = user_uid.detach().cpu().numpy()

    rows = []

    for group_name, users in test_users_pop_group.items():
        user_set = set(users)

        mask = np.array([uid in user_set for uid in user_uid])

        n_samples = int(mask.sum())
        n_users = len(user_set)

        if n_samples == 0:
            rows.append(
                {
                    "group": group_name,
                    "num_users": n_users,
                    "num_samples": 0,
                    "mae": None,
                    "rmse": None,
                }
            )
            continue

        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]

        mae = np.mean(np.abs(group_y_pred - group_y_true))
        rmse = np.sqrt(np.mean((group_y_pred - group_y_true) ** 2))

        rows.append(
            {
                "group": group_name,
                "num_users": n_users,
                "num_samples": n_samples,
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )

    df = pd.DataFrame(rows)

    # group 이름이 0.0~0.2, 0.2~0.4 형식일 때 순서 정렬
    def group_order_key(x):
        try:
            return float(str(x).split("~")[0])
        except:
            return 999

    df = df.sort_values(by="group", key=lambda col: col.map(group_order_key)).reset_index(drop=True)
    return df


def get_test_users_pop_group(test_users_degree, src_path, pop_percentile=80, bin_edges=None):
    """
    테스트 유저를 '인기 아이템 소비 비율' 기준으로 binning.

    Args:
        test_users_degree:
            보통 {uid: degree} 형태를 가정.
            만약 list/set이면 그 자체를 테스트 유저 집합으로 사용.
        src_path:
            uid, iid, y 형식의 csv 경로
        pop_percentile:
            상위 몇 퍼센트 아이템을 인기 아이템으로 볼지.
            예: 80 -> item interaction count 기준 상위 20%를 인기 아이템으로 정의
        bin_edges:
            예: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    Returns:
        test_users_pop_group: dict
            {
                "0.0~0.2": [uid1, uid2, ...],
                "0.2~0.4": [...],
                ...
            }

        user_pop_ratio: dict
            {uid: popular_item_ratio}

        popular_items: set
            인기 아이템 집합
    """
    if bin_edges is None:
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # -----------------------------
    # 1. test user set 정리
    # -----------------------------
    if isinstance(test_users_degree, dict):
        test_user_set = set(test_users_degree.keys())
    else:
        test_user_set = set(test_users_degree)

    # -----------------------------
    # 2. 데이터 로드
    # -----------------------------
    cols = ["uid", "iid", "y"]
    data = pd.read_csv(src_path, header=None)
    data.columns = cols

    # -----------------------------
    # 3. 아이템 popularity 계산
    # -----------------------------
    item_count = data["iid"].value_counts()
    pop_threshold = np.percentile(item_count.values, pop_percentile)
    popular_items = set(item_count[item_count >= pop_threshold].index.tolist())

    # -----------------------------
    # 4. 유저별 소비 아이템 목록 구성
    # -----------------------------
    user_items = defaultdict(list)
    for row in data.itertuples(index=False):
        uid = int(row.uid)
        iid = int(row.iid)
        if uid in test_user_set:
            user_items[uid].append(iid)

    # -----------------------------
    # 5. 유저별 인기 아이템 소비 비율 계산
    # -----------------------------
    user_pop_ratio = {}
    for uid in test_user_set:
        items = user_items.get(uid, [])
        if len(items) == 0:
            user_pop_ratio[uid] = 0.0
            continue

        pop_cnt = sum(1 for iid in items if iid in popular_items)
        ratio = pop_cnt / len(items)
        user_pop_ratio[uid] = ratio

    # -----------------------------
    # 6. binning
    #    [0.0,0.2), [0.2,0.4), ..., [0.8,1.0]
    # -----------------------------
    test_users_pop_group = {}
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            group_name = f"{left:.1f}~{right:.1f}"
            group_users = [uid for uid, ratio in user_pop_ratio.items() if left <= ratio <= right]
        else:
            group_name = f"{left:.1f}~{right:.1f}"
            group_users = [uid for uid, ratio in user_pop_ratio.items() if left <= ratio < right]

        test_users_pop_group[group_name] = group_users

    return test_users_pop_group, user_pop_ratio, popular_items


def mae_rmse_summary_by_sparsity(y_true, y_pred, user_degree, user_uid, n_bins=5):
    y_true = y_true.view(-1).cpu().float()
    y_pred = y_pred.view(-1).cpu().float()
    user_degree = user_degree.view(-1).cpu().float()
    user_uid = user_uid.view(-1).cpu().long()

    abs_err = (y_pred - y_true).abs()
    sq_err = (y_pred - y_true).pow(2)

    unique_degree = torch.unique(user_degree)
    if len(unique_degree) < n_bins:
        n_bins = len(unique_degree)

    if n_bins <= 1:
        return pd.DataFrame(
            [
                {
                    "group": "all",
                    "num_samples": len(y_true),
                    "num_users": int(len(torch.unique(user_uid))),
                    "degree_min": float(user_degree.min().item()),
                    "degree_max": float(user_degree.max().item()),
                    "MAE": float(abs_err.mean().item()),
                    "RMSE": float(torch.sqrt(sq_err.mean()).item()),
                }
            ]
        )

    qs = torch.linspace(0, 1, steps=n_bins + 1)
    boundaries = torch.quantile(user_degree, qs)

    rows = []
    for i in range(n_bins):
        left = boundaries[i]
        right = boundaries[i + 1]

        if i == 0:
            mask = (user_degree >= left) & (user_degree <= right)
        else:
            mask = (user_degree > left) & (user_degree <= right)

        if mask.sum() == 0:
            continue

        deg_group = user_degree[mask]
        uid_group = user_uid[mask]
        abs_group = abs_err[mask]
        sq_group = sq_err[mask]

        rows.append(
            {
                "group": f"Q{i+1}",
                "num_samples": int(mask.sum().item()),
                "num_users": int(len(torch.unique(uid_group))),
                "degree_min": float(deg_group.min().item()),
                "degree_max": float(deg_group.max().item()),
                "MAE": float(abs_group.mean().item()),
                "RMSE": float(torch.sqrt(sq_group.mean()).item()),
            }
        )

    return pd.DataFrame(rows)


def get_test_users_degree(src_path, test_path):
    """
    src_path:
        source train csv path
        format: [uid, iid, y]

    test_path:
        diff test csv path
        format: [meta_uid, iid, y, pos_seq]

    Returns:
        test_users_degree: dict
            {test_uid: degree_in_src_train}
    """

    # -----------------------------
    # 1) src train에서 uid degree 계산
    # -----------------------------
    src_df = pd.read_csv(src_path, header=None)
    src_df.columns = ["uid", "iid", "y"]

    user_degree = Counter(src_df["uid"].tolist())

    # -----------------------------
    # 2) test에 등장하는 unique user 추출
    # -----------------------------
    test_df = pd.read_csv(test_path, header=None)
    test_df.columns = ["meta_uid", "iid", "y", "pos_seq"]

    test_users = test_df["meta_uid"].unique().tolist()

    # -----------------------------
    # 3) test user별 src degree 매핑
    # -----------------------------
    test_users_degree = {uid: user_degree.get(uid, 0) for uid in test_users}

    return test_users_degree


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
    data_src, num_users: int, rating_min: float = 1.0, rating_max: float = 5.0, device: str = "cpu", cache_path=""
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
            "style": style.cpu(),  # 저장은 CPU 권장
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
            "style_tgt_item": style_item.cpu(),  # 저장은 CPU 권장
            "info_tgt": info,
        },
        cache_path,
    )

    return style_item, info


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

def make_csv_dir(log_name, headers):
    parent_dir, current_dir = get_parent_curr_dir()
    save_dir = os.path.join(current_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, f"{log_name}_result.csv")

    if not os.path.exists(logfile) and headers is not None:
        with open(logfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

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

    save_path_item = None
    items = []
    for k, v in args_dict.items():  # save_path 분리
        if k == "save_path":
            save_path_item = f"{k} = {v}"
        else:
            items.append(f"{k} = {v}")

    items = sorted(items)  # 나머지는 정렬
    if save_path_item is not None:  # save_path는 맨 마지막
        items.append(save_path_item)

    # 패딩을 넣어 고정 길이 문자열로 변환
    padded_items = [item.ljust(col_width) for item in items]

    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))
    now_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")  # 현재 한국 시각
    logging.info(f"Run Time (KST): {now_kst}")
    logging.info("Arguments:")

    for i in range(0, len(padded_items), max_per_line):
        row = padded_items[i : i + max_per_line]
        logging.info(" | ".join(row))

    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))


def log_batch_similarity_stats(user_emb, global_step, log_every=200, prefix="train_src"):
    if global_step % log_every != 0:
        return

    with torch.no_grad():
        # -------------------------
        # user-user cosine (off-diagonal)
        # -------------------------
        user_norm = F.normalize(user_emb, p=2, dim=1)
        user_sim = torch.matmul(user_norm, user_norm.t())  # [B, B]

        user_mask = ~torch.eye(user_sim.size(0), dtype=torch.bool, device=user_sim.device)
        user_offdiag = user_sim[user_mask]

        print(
            f"[{prefix}] [step {global_step}] "
            f"user offdiag cosine mean={user_offdiag.mean().item():.4f}, "
            f"std={user_offdiag.std().item():.4f}, "
            f"min={user_offdiag.min().item():.4f}, "
            f"max={user_offdiag.max().item():.4f}"
        )


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.v = nn.Linear(in_dim, out_dim, bias=False)

        self.scale = in_dim**-0.5

    def forward(self, x, mask=None, query=None, return_score=False):
        """
        x:     (B, T, D)
        query: (B, Q, D) or None
        mask:  (B, T) or None
        """

        if query is not None:
            Q = self.q(query)
        else:
            Q = self.q(x)

        K = self.k(x)
        V = self.v(x)

        # raw attention score
        # (B, Q, T)
        score = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            score = score.masked_fill(mask[:, None, :] == 0, -1e9)

        # normalized attention weight
        attn = F.softmax(score, dim=-1)

        # attention output
        out = torch.matmul(attn, V)

        if return_score:
            return out, score, attn

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
