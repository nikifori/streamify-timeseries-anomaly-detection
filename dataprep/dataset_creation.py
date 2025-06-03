"""
@File    :   dataset_creation.py
@Time    :   04/2025
@Author  :   nikifori
@Version :   -
"""

import yaml
import pandas as pd
from pathlib import Path
from itertools import combinations, permutations
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import sys

from TSB_UAD.vus.metrics import metricor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice individual CSVs and build combined normality datasets"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="/home/nikifori/Desktop/anomaly_detection/streamify-timeseries-anomaly-detection/dataprep/dataset_creation_conf.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def save_plot(
        ts_id,
        s,
        out_dir,
        anomalies_percentage,
):
    """
    Creates and saves a time-series plot of the first column in `s`,
    highlighting anomaly ranges from `s.iloc[:,1]` using metricor's range_convers_new,
    and titles the figure with the time series identifier (ts_id).

    Lower DPI improves font readability by reducing plot resolution.

    Parameters
    ----------
    fp : str or Path
        Filepath whose parent stem is used as the time series identifier.
    s : pandas.DataFrame
        DataFrame with two columns: values and binary labels (0 normal, 1 anomaly).
    out_dir : str or Path
        Directory where the plot image will be saved.
    """
    out_dir = Path(out_dir)

    # Extract data and labelss
    data = s.iloc[:, 0].values
    labels = s.iloc[:, 1].values

    # Compute anomaly ranges
    grader = metricor()
    range_anomaly = grader.range_convers_new(labels)
    max_length = len(data)
    plot_range = [0, max_length]

    # Lower resolution for better font readability
    plt.rcParams.update({'figure.dpi': 80, 'font.size': 14})

    # Create full-frame figure
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)

    # Plot the time series in black
    ax.plot(data[:max_length], 'k', label='Value')

    # Highlight anomalies in red
    for start, end in range_anomaly:
        if start == end:
            ax.plot(start, data[start], 'r.')
        else:
            idx = range(start, end + 1)
            ax.plot(idx, data[start:end + 1], 'r')

    # Format plot
    ax.set_xlim(plot_range)
    ax.set_title(ts_id, fontsize=25)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # Save figure
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{ts_id}_ts_anomalies={anomalies_percentage}%.png"
    fig.savefig(str(save_path), dpi=80)
    plt.close(fig)

    print(f"Figure saved to {save_path}")


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    single_data = cfg.get("single_data", [])
    offsets = cfg.get("offsets", [])
    lengths = cfg.get("lengths", [])
    output_dir = cfg.get("output_dir", ".")

    if not (len(single_data) == len(offsets) == len(lengths)):
        raise ValueError(
            "`single_data`, `offsets` and `lengths` must all be the same length"
        )
    
    return single_data, offsets, lengths, output_dir


def slice_file(fp: str, offset: int, length: int) -> pd.DataFrame:
    # load, drop rows with any NaN, then slice rows [offset:offset+length]
    df = pd.read_csv(fp, header=None).dropna()
    total_rows = len(df)

    # Ensure exactly two columns
    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {fp}, but found {df.shape[1]}")

    # Convert the anomaly label to float
    try:
        df[1] = df[1].astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert the second column to float: {e}")

    # Determine numeric offset
    if isinstance(offset, int):
        start = offset
    elif isinstance(offset, str) and offset.lower() == "random":
        if total_rows == 0:
            raise ValueError(f"No rows available in {fp} to pick a random offset.")
        # Ensure at least one row in slice; if length >= total_rows, any offset is fine (we'll slice to end)
        max_start = max(total_rows - length, 0)
        start = random.randint(0, max_start)
    else:
        raise TypeError(f"offset must be an int or 'random', got {offset!r}")

    # Error if offset beyond available data
    if start >= total_rows:
        raise IndexError(
            f"Offset {start} is beyond total rows ({total_rows}) for file {fp}"
        )

    # Warn if requested length exceeds data bounds
    end = start + length
    if end > total_rows:
        print(
            f"[WARNING] Requested end {end} exceeds total rows ({total_rows}); slicing to end.",
        )
        end = total_rows

    return df.iloc[start:end].reset_index(drop=True)


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
        anomalies_percentage = 0.0
        while not 9.0 < anomalies_percentage < 11.0:
            s = slice_file(fp, ofs, ln)

            anomalies_percentage = int(s.iloc[:, 1].sum()) / len(s) * 100
            print(
                f"Trying {fp} with offset {ofs} and length {ln}: {anomalies_percentage:.2f}% anomalies"
            )

            sliced[fp] = s
            filename = make_name([fp]) + ".csv"
            out_path = out_dir / filename
            s.to_csv(out_path, index=False, header=False)
            print(f"→ wrote single: {out_path}")
        
        ts_id = Path(fp).parent.stem
        save_plot(
            ts_id,
            s,
            out_dir,
            anomalies_percentage,
        )

    # build combinations of size 2…N
    n = len(single_data)
    for r in range(2, n + 1):
        for combo in permutations(single_data, r):
            concatenated = pd.concat([sliced[fp] for fp in combo], ignore_index=True)

            ts_id = "_".join([Path(fp).parent.stem for fp in combo])
            anomalies_percentage = int(concatenated.iloc[:, 1].sum()) / len(concatenated) * 100
            save_plot(
                ts_id,
                concatenated,
                out_dir,
                anomalies_percentage,
            )

            filename = make_name(combo) + ".csv"
            out_path = out_dir / filename
            concatenated.to_csv(out_path, index=False, header=False)
            print(f"→ wrote {len(concatenated)} rows to {out_path}")
            print(f"→ wrote {r}-normality: {out_path}")


if __name__ == "__main__":
    main()
