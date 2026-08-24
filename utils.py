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


@torch.no_grad()
def build_src_user_rating_style_from_loader(
    data_src, num_users: int, rating_min: float = 1.0, rating_max: float = 5.0, device: str = "cpu", cache_path=""
):

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

        sums.index_add_(0, uid, r)
        sumsqs.index_add_(0, uid, r * r)
        cnts.index_add_(0, uid, torch.ones_like(r, dtype=torch.float64))

        uniq = torch.unique(uid)
        for u in uniq.tolist():
            mask = uid == u
            rv = r[mask]
            mins[u] = torch.minimum(mins[u], rv.min())
            maxs[u] = torch.maximum(maxs[u], rv.max())

        is_min = r <= rating_min + 1e-12
        is_max = r >= rating_max - 1e-12
        cnt_min.index_add_(0, uid, is_min.double())
        cnt_max.index_add_(0, uid, is_max.double())

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

    style = torch.stack([mean, var, std, rmin, rmax, cnts, frac_min, frac_max, frac_extreme], dim=1).to(torch.float32).to(device)

    info = {
        "feature_names": ["mean", "var", "std", "min", "max", "cnt", "frac_min", "frac_max", "frac_extreme"],
        "rating_min": rating_min,
        "rating_max": rating_max,
        "num_users": num_users,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {"style": style.cpu(), "info": info},
        cache_path,
    )

    return style, info


@torch.no_grad()
def build_tgt_item_rating_style_from_loader(
    data_tgt, num_items_total: int, rating_min: float = 1.0, rating_max: float = 5.0, device: str = "cpu", cache_path=""
):

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

        if iid.numel() > 0:
            if iid.min().item() < 0 or iid.max().item() >= num_items_total:
                raise ValueError(f"iid out of range: min={iid.min().item()}, max={iid.max().item()}, " f"num_items_total={num_items_total}")

        sums.index_add_(0, iid, r)
        sumsqs.index_add_(0, iid, r * r)
        cnts.index_add_(0, iid, torch.ones_like(r, dtype=torch.float64))

        uniq = torch.unique(iid)
        for it in uniq.tolist():
            mask = iid == it
            rv = r[mask]
            mins[it] = torch.minimum(mins[it], rv.min())
            maxs[it] = torch.maximum(maxs[it], rv.max())

        is_min = r <= rating_min + 1e-12
        is_max = r >= rating_max - 1e-12
        cnt_min.index_add_(0, iid, is_min.double())
        cnt_max.index_add_(0, iid, is_max.double())

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
        with open(logfile, "w", newline="") as f:
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
    args_dict = vars(args)

    save_path_item = None
    items = []
    for k, v in args_dict.items():
        if k == "save_path":
            save_path_item = f"{k} = {v}"
        else:
            items.append(f"{k} = {v}")

    items = sorted(items)
    if save_path_item is not None:
        items.append(save_path_item)

    padded_items = [item.ljust(col_width) for item in items]

    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))
    now_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    logging.info(f"Run Time (KST): {now_kst}")
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
