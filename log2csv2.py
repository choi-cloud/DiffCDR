import csv
import itertools
import math
import os
import re


def parse_eval_results(log_path: str):
    """
    log 파일에서
    '⬇️ Eval diff_parallel: MAE & RMSE'
    다음의 두 줄 숫자(MAE, RMSE)를 순서대로 추출한다.

    반환:
        [(mae1, rmse1), (mae2, rmse2), ...]
    """
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    results = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if "Eval diff_parallel: MAE & RMSE" in line:
            # 다음 두 줄에서 float 숫자 찾기
            if i + 2 < len(lines):
                mae_line = lines[i + 1].strip()
                rmse_line = lines[i + 2].strip()

                try:
                    mae = float(mae_line)
                    rmse = float(rmse_line)
                    results.append((mae, rmse))
                    i += 3
                    continue
                except ValueError:
                    pass

        i += 1

    return results


def generate_param_combinations():
    """
    bash 스크립트의 반복 순서와 동일하게 하이퍼파라미터 조합 생성
    """
    codebook_sizes = [64, 256]
    diff_scales = [0.5, 1.5]
    diff_lrs = [0.001, 0.005]
    task_lambdas = [0.1, 0.5, 1, 1.5]
    mapping_lambdas = [0.1, 1, 5, 10]
    ratios = ["[0.2, 0.8]", "[0.5, 0.5]", "[0.8, 0.2]"]
    tasks = [2]

    combos = list(itertools.product(
        codebook_sizes,
        diff_scales,
        diff_lrs,
        task_lambdas,
        mapping_lambdas,
        ratios,
        tasks
    ))

    return combos


def build_rows_from_results(results):
    """
    parsed result 순서와 bash 실행 순서를 맞춰
    csv row 형태로 변환

    각 하이퍼파라미터 조합마다 2행:
    - metric=MAE
    - metric=RMSE

    컬럼:
    codebook_size, diff_scale, diff_lr, diff_task_lambda, mapping_lambda, ratio, task, metric, ALL, RQ X, ab1, ab2
    """
    combos = generate_param_combinations()
    ablations = ["ALL", "RQ X", "ab1", "ab2"]

    used_combo_count = math.ceil(len(results) / 4)
    used_combos = combos[:used_combo_count]

    grouped = []
    for combo in used_combos:
        grouped.append({
            "codebook_size": combo[0],
            "diff_scale": combo[1],
            "diff_lr": combo[2],
            "diff_task_lambda": combo[3],
            "mapping_lambda": combo[4],
            "ratio": combo[5],
            "task": combo[6],
            "MAE": {"ALL": "", "RQ X": "", "ab1": "", "ab2": ""},
            "RMSE": {"ALL": "", "RQ X": "", "ab1": "", "ab2": ""},
        })

    for idx, (mae, rmse) in enumerate(results):
        combo_idx = idx // 4
        ablation_idx = idx % 4

        if combo_idx >= len(grouped):
            break

        ablation_name = ablations[ablation_idx]
        grouped[combo_idx]["MAE"][ablation_name] = mae
        grouped[combo_idx]["RMSE"][ablation_name] = rmse

    rows = []
    for item in grouped:
        mae_row = {
            "codebook_size": item["codebook_size"],
            "diff_scale": item["diff_scale"],
            "diff_lr": item["diff_lr"],
            "diff_task_lambda": item["diff_task_lambda"],
            "mapping_lambda": item["mapping_lambda"],
            "ratio": item["ratio"],
            "task": item["task"],
            "metric": "MAE",
            "ALL": item["MAE"]["ALL"],
            "RQ X": item["MAE"]["RQ X"],
            "ab1": item["MAE"]["ab1"],
            "ab2": item["MAE"]["ab2"],
        }

        rmse_row = {
            "codebook_size": item["codebook_size"],
            "diff_scale": item["diff_scale"],
            "diff_lr": item["diff_lr"],
            "diff_task_lambda": item["diff_task_lambda"],
            "mapping_lambda": item["mapping_lambda"],
            "ratio": item["ratio"],
            "task": item["task"],
            "metric": "RMSE",
            "ALL": item["RMSE"]["ALL"],
            "RQ X": item["RMSE"]["RQ X"],
            "ab1": item["RMSE"]["ab1"],
            "ab2": item["RMSE"]["ab2"],
        }

        rows.append(mae_row)
        rows.append(rmse_row)

    return rows


def save_csv(rows, save_path: str):
    fieldnames = [
        "codebook_size",
        "diff_scale",
        "diff_lr",
        "diff_task_lambda",
        "mapping_lambda",
        "ratio",
        "task",
        "metric",
        "ALL",
        "RQ X",
        "ab1",
        "ab2",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    log_path = "./logs/0404Ablation_log.txt"
    csv_path = "./logs/0404Ablation_summary.csv"

    results = parse_eval_results(log_path)
    rows = build_rows_from_results(results)
    save_csv(rows, csv_path)

    print(f"Parsed runs: {len(results)}")
    print(f"Saved csv: {csv_path}")


if __name__ == "__main__":
    main()