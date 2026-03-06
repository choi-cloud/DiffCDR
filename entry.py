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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument("--save_path", default="/home/shared/cjp/model_save_default_9/model")
    parser.add_argument("--use_cuda", default=1)
    parser.add_argument("--experiment", default="DiffCDR")

    # parallel setting
    parser.add_argument("--set_loss", type=int, default=0, help="loss 계산, 0: MF, 1: aggr, 2: avg, 3: 따로따로")
    parser.add_argument("--set_init", type=int, default=1, help="디퓨전2의 초기 x_T 설정, 0: MF, 1: aggr")
    parser.add_argument("--set_proj", type=int, default=1, help="diff 결과 proj 위치 - 0: 따로, 1: aggr 이후 같이")
    parser.add_argument("--set_aggr", type=str, default="item_diu", help="[item_diu, item_d, item_i, item_u, item_di, item_du, item_iu]")
    parser.add_argument("--aggregation", type=str2bool, default=True, help="MF+Aggr or MF only")

    # RQVAE(code_dim=input_dim, num_levels=4, codebook_size=256)
    parser.add_argument("--codebook_num", type=int, default=4, help="RQVAE 코드북 개수(level)")
    parser.add_argument("--codebook_size", type=int, default=256, help="RQVAE 코드북 크기")
    parser.add_argument("--alpha_rq", type=float, default=1e-2, help="RQVAE loss 가중치")
    parser.add_argument("--RQVAE", type=str2bool, default=True, help="rq 사용 여부")
    parser.add_argument("--start_point", default="src_u", help="[src_u, quant_u, noise]")
    

    # item cond
    parser.add_argument("--item_cond", type=bool, default=False, help="아이템 조건 사용 여부")
    parser.add_argument("--w", type=float, default=0.0, help="Diffusion inference - uncond 가중치 w")
    parser.add_argument("--emb_dim", type=int, default=10, help="MF emb dim")

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
        config["set_loss"] = int(args.set_loss)
        config["set_init"] = int(args.set_init)
        config["set_proj"] = int(args.set_proj)
        config["set_aggr"] = args.set_aggr
        config["aggregation"] = args.aggregation
        config["item_cond"] = args.item_cond
        config["codebook_num"] = args.codebook_num
        config["codebook_size"] = args.codebook_size
        config["alpha_rq"] = args.alpha_rq
        config["w"] = args.w
        config["emb_dim"] = args.emb_dim
        config["RQVAE"] = args.RQVAE
        config["start_point"] = args.start_point

    return config


if __name__ == "__main__":
    args = prepare_1()

    config_path = args.root + "config.json"

    config = prepare_2(args, config_path)
    # config["root"] = args.root + "data/"
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

    print()
    print("tgt_global_bias 학습")
    print()

    logfile = utils.make_dir(f"{args.experiment}")
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=logfile,
        filemode="a",
        encoding="utf-8",
    )

    utils.log_args_table(args, max_per_line=5, col_width=30)
    write(f"{' '+args.experiment+' ':=^{30}}")
    write(f"✅ Task  {args.task}")
    write(f"✅ Ratio {args.ratio}")
    write(f"✅ Model {args.exp_part}")

    write(f"🍎 emb dim     : {args.emb_dim}")
    write(f"🍎 bias  : {args.set_aggr}")
    write(f"🍎 aggregation  : {args.aggregation}")
    write(f"🍎 RQVAE       : {args.RQVAE}")
    write(f"🍎 start_point : {args.start_point}")

    write(f"🍏 Bias        : {args.set_aggr}")

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main(args.exp_part, f"{args.save_path}_{args.task}_{args.ratio}.pth")
        write(f"{'':=^{30}}")

 