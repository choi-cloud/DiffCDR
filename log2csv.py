import argparse
import os
import re
import pandas as pd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert log to CSV")

    parser.add_argument("--experiment", type=str, required=True,
                        help="experiment name (e.g., AAABLATION)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="output csv path")
    parser.add_argument("--print_df", type=str2bool, default=True)

    return parser.parse_args()


# ✅ utils.make_dir와 동일한 로직
def get_log_path(experiment):
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, "logs")
    return os.path.join(log_dir, f"{experiment}_log.txt")


def parse_log_to_df(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    runs = re.split(r"=+\nRun Time \(KST\):", text)
    runs = [r for r in runs if r.strip()]

    rows = []

    for run in runs:
        task_match = re.search(r"✅ Task\s+(\d+)", run)
        ratio_match = re.search(r"✅ Ratio\s+(\[.*?\])", run)

        rqvae_match = re.search(r"RQVAE\s*=\s*(\w+)", run)
        aggregation_match = re.search(r"aggregation\s*=\s*(\w+)", run)
        start_point_match = re.search(r"start_point\s*=\s*(\w+)", run)
        cross_cond_match = re.search(r"cross_cond\s*=\s*(\w+)", run)
        set_aggr_match = re.search(r"set_aggr\s*=\s*(\w+)", run)

        eval_match = re.search(
            r"⬇️ Eval diff_parallel: MAE & RMSE\s*\n([0-9.]+)\s*\n([0-9.]+)",
            run
        )

        if eval_match:
            rows.append({
                "task": int(task_match.group(1)) if task_match else None,
                "ratio": ratio_match.group(1) if ratio_match else None,
                "RQVAE": rqvae_match.group(1) if rqvae_match else None,
                "aggregation": aggregation_match.group(1) if aggregation_match else None,
                "start_point": start_point_match.group(1) if start_point_match else None,
                "cross_cond": cross_cond_match.group(1) if cross_cond_match else None,
                "set_aggr": set_aggr_match.group(1) if set_aggr_match else None,
                "MAE": float(eval_match.group(1)),
                "RMSE": float(eval_match.group(2)),
            })

    df = pd.DataFrame(rows)

    # bool 변환
    if not df.empty:
        if "RQVAE" in df.columns:
            df["RQVAE"] = df["RQVAE"].map(lambda x: x == "True")
        if "cross_cond" in df.columns:
            df["cross_cond"] = df["cross_cond"].map(lambda x: x == "True")

    return df


def main():
    args = parse_args()

    log_path = get_log_path(args.experiment)

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = parse_log_to_df(log_path)

    # csv 저장 경로
    if args.csv_path is None:
        base, _ = os.path.splitext(log_path)
        csv_path = base + ".csv"
    else:
        csv_path = args.csv_path

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[Log] {log_path}")
    print(f"[CSV] {csv_path}")

    if args.print_df:
        print(df)


if __name__ == "__main__":
    main()