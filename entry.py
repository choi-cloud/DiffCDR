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
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    parser.add_argument("--save_path", default="/home/shared/cjp/model_save_default_10/model_mlp")
    # parser.add_argument("--save_path", default="/home/schoi/DiffCDR/model_save_default/model_mlp")
    parser.add_argument("--use_cuda", default=1)
    parser.add_argument("--experiment", default="0414_uni23")

    # parallel setting
    parser.add_argument("--set_aggr", type=str, default="item_iu", help="[item_diu, item_d, item_i, item_u, item_di, item_du, item_iu]")
    parser.add_argument("--aggregation", type=str, default="aggregation", help="[aggregation, aggregation_ab1, aggregation_ab2]")

    # RQVAE(code_dim=input_dim, num_levels=4, codebook_size=256)
    parser.add_argument("--codebook_num", type=int, default=4, help="RQVAE 코드북 개수(level)")
    parser.add_argument("--codebook_size", type=int, default=12, help="RQVAE 코드북 크기")
    parser.add_argument("--RQVAE", type=str2bool, default=True, help="rq 사용 여부")
    parser.add_argument("--pretrain_rq", type=str2bool, default=True, help="rqvae pretrain 여부")
    parser.add_argument("--pretrain_epochs", type=int, default=50, help="rqvae pretrain epoch")
    parser.add_argument("--freeze_rq", type=str2bool, default=True, help="rqvae 파라미터 고정 여부")
    parser.add_argument("--cross_cond", type=str2bool, default=False, help="MF vs Aggr 컨디션 교차 여부")
    parser.add_argument("--start_point", default="noise", help="[src_u, quant_u, noise]")
    parser.add_argument("--rqvae_lr", type=float, default=0.001, help="rqvae pretrain lr")
    parser.add_argument("--rq_num", type=int, default=0, help="bias reconstruction loss")

    # item cond
    parser.add_argument("--diff_task_lambda", type=float, default=1.0, help="Task loss weight")
    parser.add_argument("--diff_scale", type=float, default=0.5, help="Classifier-free guidance scale")
    parser.add_argument("--diff_mask_rate", type=float, default=0.1, help="Diffusion condition mask rate")
    parser.add_argument("--emb_dim", type=int, default=10, help="MF emb dim")
    parser.add_argument("--bias_mapping", type=str, default="user", help="[None, user, user_domain] mapper input")
    parser.add_argument("--mapping_lambda", type=float, default=10, help="mapping loss 가중치")
    parser.add_argument("--uniformity_loss", type=float, default=0.1, help="uniformity loss 가중치")
    parser.add_argument("--zero_cond", type=str2bool, default=False, help="cond zero 실험")
    parser.add_argument("--batch_norm", type=str2bool, default=False, help="batch norm 사용 여부")
    parser.add_argument("--recon_loss", type=float, default=0.1, help="bias reconstruction loss")

    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
        config["rqvae_lr"] = args.rqvae_lr
        config["set_aggr"] = args.set_aggr
        config["aggregation"] = args.aggregation
        config["bias_mapping"] = args.bias_mapping
        config["codebook_num"] = args.codebook_num
        config["codebook_size"] = args.codebook_size
        config["emb_dim"] = args.emb_dim
        config["RQVAE"] = args.RQVAE
        config["pretrain_rq"] = args.pretrain_rq
        config["pretrain_epochs"] = args.pretrain_epochs
        config["freeze_rq"] = args.freeze_rq
        config["cross_cond"] = args.cross_cond
        config["start_point"] = args.start_point
        config["diff_task_lambda"] = args.diff_task_lambda
        config["diff_scale"] = args.diff_scale
        config["diff_mask_rate"] = args.diff_mask_rate
        config["mapping_lambda"] = args.mapping_lambda
        config["uniformity_loss"] = args.uniformity_loss
        config["zero_cond"] = args.zero_cond
        config["batch_norm"] = args.batch_norm
        config["recon_loss"] = args.recon_loss
        config["rq_num"] = args.rq_num

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

    logfile = utils.make_dir(f"{args.experiment}")
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        filename=logfile,
        filemode="a",
        encoding="utf-8",
    )

    headers = ["seed", "baseline", "task", "ratio0", "ratio1", "metric", "result"]
    csvfile = utils.make_csv_dir(f"{args.experiment}", headers)
    config["csvfile"] = csvfile
    config["seed"] = args.seed

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

    write(f"🍏 cross cond   : {args.cross_cond}")
    write(f"🍏 bias mapping : {args.bias_mapping}")

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main(
            args.exp_part,
            f"{args.save_path}_{args.task}_{args.ratio}.pth" if args.seed == 1 else f"{args.save_path}_{args.seed}_{args.task}_{args.ratio}.pth",
        )
        write(f"{'':=^{30}}")
