'''
@File    :   dataset_creation.py
@Time    :   04/2025
@Author  :   nikifori
@Version :   -
'''
import yaml
import pandas as pd
from pathlib import Path
from itertools import combinations
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice individual CSVs and build combined normality datasets"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="/home/nikifori/Desktop/anomaly_detection/dataset_creation_conf.yaml",
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    single_data = cfg.get("single_data", [])
    offsets     = cfg.get("offsets", [])
    lengths     = cfg.get("lengths", [])
    output_dir  = cfg.get("output_dir", ".")

    if not (len(single_data) == len(offsets) == len(lengths)):
        raise ValueError(
            "`single_data`, `offsets` and `lengths` must all be the same length"
        )
    return single_data, offsets, lengths, output_dir


def slice_file(fp: str, offset: int, length: int) -> pd.DataFrame:
    # load, drop rows with any NaN, then slice rows [offset:offset+length]
    df = pd.read_csv(fp, header=None).dropna()
    total_rows = len(df)

    # Error if offset beyond available data
    if offset >= total_rows:
        raise IndexError(
            f"Offset {offset} is beyond total rows ({total_rows}) for file {fp}"
        )

    # Warn if requested length exceeds data bounds
    end = offset + length
    if end > total_rows:
        print(
            f"[WARNING] Requested end {end} exceeds total rows ({total_rows}); slicing to end.",
        )
        end = total_rows

    return df.iloc[offset:end].reset_index(drop=True)


def make_name(paths):
    # for each filepath, grab the name of parent-of-parent folder
    names = [Path(fp).parents[0].name for fp in paths]
    return "_".join(names)


def main():
    args = parse_args()
    single_data, offsets, lengths, output_dir = load_config(args.config)

    # ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load & slice singles
    sliced = {}
    for fp, ofs, ln in zip(single_data, offsets, lengths):
        s = slice_file(fp, ofs, ln)
        sliced[fp] = s
        filename = make_name([fp]) + ".csv"
        out_path = out_dir / filename
        s.to_csv(out_path, index=False, header=False)
        print(f"→ wrote single: {out_path}")

    # build combinations of size 2…N
    n = len(single_data)
    for r in range(2, n + 1):
        for combo in combinations(single_data, r):
            concatenated = pd.concat([sliced[fp] for fp in combo], ignore_index=True)
            filename = make_name(combo) + ".csv"
            out_path = out_dir / filename
            concatenated.to_csv(out_path, index=False, header=False)
            print(f"→ wrote {len(concatenated)} rows to {out_path}")
            print(f"→ wrote {r}-normality: {out_path}")


if __name__ == "__main__":
    main()