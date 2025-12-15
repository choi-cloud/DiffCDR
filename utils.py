import numpy as np
import os
import sys
import logging
import csv

def get_parent_curr_dir():
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    sys.path.append(parent_dir)
    current_dir = os.path.dirname(current_file_path)
    return parent_dir, current_dir

def make_dir(log_name): 
    parent_dir, current_dir = get_parent_curr_dir()
    save_dir = os.path.join(current_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)
    logfile = os.path.join(save_dir, f'{log_name}_log.txt')
    return logfile 

def get_save_name(args, pretrain_dataset_names, save_dir, result_dir): 
    pretrain_dataset_str = ''
    for strs in pretrain_dataset_names: 
        pretrain_dataset_str += '_'+strs
    # set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
    # set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.ablation_pre}_{args.sample_size}_{args.nb_epochs}_{args.de_loss}_{args.de_weight}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'

    set_name = f'model_node_{args.pretrain_method}_{pretrain_dataset_str}_{args.ablation_pre}_{args.sample_size}_{args.nb_epochs}_{args.if_rand}_{args.w1loss}_{args.de_loss}_{args.de_weight}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'

    save_name = os.path.join(save_dir, f'{set_name}.pkl')
    csv_name = os.path.join(result_dir, f'{set_name}.csv')

    return save_name, csv_name

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False

def write(txt='\n'): 
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
        row = padded_items[i:i+max_per_line]
        logging.info(" | ".join(row))
    
    logging.info("=" * (col_width * max_per_line + (max_per_line - 1)))