'''
@File    :   eval.py
@Time    :   04/2025
@Author  :   nikifori
@Version :   -
'''
import argparse
import yaml
import math
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import multiprocessing

from sklearn.preprocessing import MinMaxScaler

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.sand import SAND
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length

from figures_creation import plotFig

def prepare_data_unsupervised(
        filepath: Path = None,
        max_length: int = -1,
):
    df = pd.read_csv(filepath, header=None).dropna().to_numpy()
    name = filepath.stem

    print(f"Data shape is: {df.shape}")
    print(f"Data header is: {df[:5]}")

    data = df[:max_length,0].astype(float)
    label = df[:max_length,1].astype(int)

    slidingWindow = find_length(data)
    X_data = Window(window = slidingWindow).convert(data).to_numpy()
    return data, X_data, label, name, slidingWindow


def method_eval(
        method: str = None,
        type: str = None,
        online_variant: str = None,
        data_file: Path = None,
        cfg: Dict = None,
):  
    print("#"*50)
    print(f"[INFO] Evaluating {method} on {data_file.stem} with type {type} and online variant {online_variant}")
    print("#"*50)
    exp_folder = Path(cfg["output_dir"]) / cfg["exp_name"] / f"{method}_{type}_{online_variant}_{data_file.stem}"
    exp_folder.mkdir(parents=True, exist_ok=True)
    data, X_data, label, name, slidingWindow = prepare_data_unsupervised(
        data_file,
        max_length=cfg["max_length"],
    )
    match method:
        case "SAND":
            if type == "offline":
                modelName='SAND (offline)'
                clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
                clf.fit(data,overlaping_rate=int(1.5*slidingWindow))
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName, exp_folder=str(exp_folder)) 
            elif type == "online":
                modelName='SAND (online)'
                clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
                clf.fit(data,online=True,alpha=0.5,init_length=5000,batch_size=2000,verbose=True,overlaping_rate=int(4*slidingWindow))
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName, exp_folder=str(exp_folder)) 
            else:
                raise ValueError(f"Unknown type: {type}")
        case "IForest":
            if type == "offline":
                modelName='IForest (offline)'
                clf = IForest(n_jobs=1)
                clf.fit(X_data)
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
                plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName, exp_folder=str(exp_folder)) 

            elif type == "online":
                raise NotImplementedError("Online IForest not implemented yet.")
            else:
                raise ValueError(f"Unknown type: {type}")
        case _:
            raise ValueError(f"Unknown method: {method}")


def safe_load_yaml(yaml_file) -> Dict:
    """
    Load a YAML file and return the contents as a dictionary.
    """
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    assert (
        len(cfg.get("methods", []))
        == len(cfg.get("type", []))
        == len(cfg.get("online_variant", []))
    ), "`methods`, `type` and `online_variant` must all be the same length"

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation on multi-normality data."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="/home/nikifori/Desktop/anomaly_detection/streamify-timeseries-anomaly-detection/eval/config.yaml",
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()


def load_valid_filepaths(
        num_of_normalities_under_test: List,
        normalities_under_test: List,
        generated_data_path: str,
):
    # using pathlib Path find all .csv files in generated_data_path
    data_files = list(Path(generated_data_path).glob("*.csv"))
    valid_files = [
        filepath for filepath in data_files if len(filepath.stem.split("_")) in num_of_normalities_under_test
    ]

    valid_files = [
        filepath for filepath in valid_files if any(
            normality in filepath.stem for normality in normalities_under_test
        )
    ]
    return valid_files


def main():
    args = parse_args()

    params = safe_load_yaml(args.config)

    valid_files = load_valid_filepaths(
        num_of_normalities_under_test=params.get("num_of_normalities_under_test", [1,2,3,4,5]),
        normalities_under_test=params.get("normalities_under_test", ["ECG","GHL","IOPS","MITDB","SVDB"]),
        generated_data_path=params.get("generated_data_path", None),
    )

    starmap_args = [
        (x, y, z, valid_file, params)for x, y, z in zip(
            params.get("methods", []),
            params.get("type", []),
            params.get("online_variant", [])
        )
        for valid_file in valid_files
    ]

    total_cpus = multiprocessing.cpu_count()
    requested = params.get("cpu_num", 1)
    cpu_num = min(requested, total_cpus - 1) if total_cpus > 1 else 1

    with multiprocessing.Pool(processes=cpu_num) as pool:
        pool.starmap(method_eval, starmap_args)



if __name__ == '__main__':
    main()


# modelName='SAND (offline)'
# clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
# x = data
# clf.fit(x,overlaping_rate=int(1.5*slidingWindow))
# score = clf.decision_scores_
# score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
# plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName) #, plotRange=[1775,2200]


# modelName='SAND (online)'
# clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
# x = data
# clf.fit(x,online=True,alpha=0.5,init_length=5000,batch_size=2000,verbose=True,overlaping_rate=int(4*slidingWindow))
# score = clf.decision_scores_
# score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
# plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName) #, plotRange=[1775,2200]


# modelName='IForest'
# clf = IForest(n_jobs=1)
# x = X_data
# clf.fit(x)
# score = clf.decision_scores_
# score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
# score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

# plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName)


# filepath = '/home/nikifori/Desktop/anomaly_detection/TSB-UAD-Public/TSB-UAD-Public/ECG/MBA_ECG805_data.out'
# df = pd.read_csv(filepath, header=None).dropna().to_numpy()

# name = filepath.split('/')[-1]
# max_length = 10000

# print(f"Data shape is: {df.shape}")
# print(f"Data header is: {df[:5]}")
# data = df[:max_length,0].astype(float)
# label = df[:max_length,1].astype(int)

# slidingWindow = find_length(data)
# X_data = Window(window = slidingWindow).convert(data).to_numpy()


# # Prepare data for semisupervised method.
# # Here, the training ratio = 0.1

# data_train = data[:int(0.1*len(data))]
# data_test = data

# X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
# X_test = Window(window = slidingWindow).convert(data_test).to_numpy()

# print("Estimated Subsequence length: ",slidingWindow)
# print("Time series length: ",len(data))
# print("Number of abnormal points: ", list(label).count(1))
