import os
import torch
import numpy as np
import random
import argparse
import json
import logging
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run
import utils
from utils import write


def prepare_1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_data_mid", default=0)
    parser.add_argument("--process_data_ready", default=0)
    parser.add_argument("--task", default="1")
    parser.add_argument("--base_model", default="MF")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratio", default="[0.8, 0.2]")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--la_lr", type=float, default=0.001)
    parser.add_argument("--diff_lr", type=float, default=0.001)

    parser.add_argument("--root", default="./")
    parser.add_argument("--exp_part", default="None_CDR")
    parser.add_argument("--save_path", default="./model_save_default/model")
    parser.add_argument("--use_cuda", default=1)
    parser.add_argument("--experiment", default="DiffCDR")

    # VBGE
    parser.add_argument("--vbge_GNN", type=int, default=2, help="GNN layer.")
    parser.add_argument("--vbge_drouout", type=float, default=0.3, help="GNN layer dropout rate.")
    parser.add_argument("--vbge_feature_dim", type=int, default=128, help="Initialize network embedding dimension.")
    parser.add_argument("--vbge_hidden_dim", type=int, default=128, help="GNN network hidden embedding dimension.")
    parser.add_argument("--vbge_leakey", type=float, default=0.1)
    parser.add_argument("--use_vbge", type=int, default=0, help="Use VBGE aggregation (1) or simple 2-hop (0).")

    # parallel setting
    parser.add_argument("--set_loss", type=int, default=0, help="loss 계산, 0: MF, 1: aggr, 2: avg, 3: 따로따로")
    parser.add_argument("--set_init", type=int, default=1, help="디퓨전2의 초기 x_T 설정, 0: MF, 1: aggr, 2: avg")
    parser.add_argument("--set_proj", type=int, default=1, help="diff 결과 proj 위치 - 0: 따로, 1: aggr 이후 같이")
    parser.add_argument("--set_aggr", type=str, default="attn", help="두 디퓨전 모델 아웃풋 aggregation 방법, [avg, add, concat]")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args


def prepare_2(args, config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        config["base_model"] = args.base_model
        config["task"] = args.task
        config["ratio"] = args.ratio
        config["epoch"] = args.epoch
        config["lr"] = args.lr
        config["la_lr"] = args.la_lr
        config["diff_lr"] = args.diff_lr
        config["vbge_GNN"] = args.vbge_GNN
        config["vbge_drouout"] = args.vbge_drouout
        config["vbge_feature_dim"] = args.vbge_feature_dim
        config["vbge_hidden_dim"] = args.vbge_hidden_dim
        config["vbge_leakey"] = args.vbge_leakey
        config["use_vbge"] = int(args.use_vbge)
        config["set_loss"] = int(args.set_loss)
        config["set_init"] = int(args.set_init)
        config["set_proj"] = int(args.set_proj)
        config["set_aggr"] = args.set_aggr

    return config


if __name__ == "__main__":
    args = prepare_1()

    config_path = args.root + "config.json"

    config = prepare_2(args, config_path)
    config["root"] = args.root + "data/"
    config["use_cuda"] = 0 if args.use_cuda == "0" else 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 데이터 전처리 필요할 경우 실행.
    if args.process_data_mid:
        for dealing in ["Books", "CDs_and_Vinyl", "Movies_and_TV"]:
            DataPreprocessingMid(config["root"], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            for task in ["1", "2", "3"]:
                DataPreprocessingReady(config["root"], config["src_tgt_pairs"], task, ratio).main()
    print(
        "task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};".format(
            args.task, args.base_model, args.ratio, args.epoch, args.lr, args.gpu, args.seed
        )
    )
    print(
        "diff_steps:{};diff_sample_steps:{};diff_scale:{};diff_dim:{};diff_task_lambda:{};".format(
            config["diff_steps"], config["diff_sample_steps"], config["diff_scale"], config["diff_dim"], config["diff_task_lambda"]
        )
    )

    logfile = utils.make_dir(f"{args.experiment}")
    logging.basicConfig(
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        level=logging.INFO,
        filename=logfile,
        filemode="a",
        encoding="utf-8",
    )

    utils.log_args_table(args, max_per_line=5, col_width=30)
    write(f"======= {args.experiment} ======")
    write(f"✅ Task  {args.task}")
    write(f"✅ Ratio {args.ratio}")
    write(f"✅ Model {args.exp_part}")
    write(f"✅ VBGE  {args.use_vbge}")

    if args.exp_part == "diff_parallel":
        set_init = ["MF로 초기화", "Aggr로 초기화", "MF+Aggr 평균으로 초기화", "DiffCDR에 cond만 agg로"]
        set_loss = ["둘 다 MF", "둘 다 Aggr", "둘 다 MF+Aggr 평균", "따로따로"]
        set_proj = ["따로 Proj", "합치고 proj"]

        write(f"⭐ Diff2 초기화  : ({args.set_init}) x_T = {set_init[args.set_init]}")
        write(f"⭐ Diff loss 계산: ({args.set_loss}) {set_loss[args.set_loss]}")
        write(f"⭐ Project 위치  : ({args.set_proj}) {set_proj[args.set_proj]}")
    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main(args.exp_part, f"{args.save_path}_{args.task}_{args.ratio}.pth")
        write(f"==============================")
