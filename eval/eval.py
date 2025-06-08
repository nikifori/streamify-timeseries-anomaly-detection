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
import time
import json

from sklearn.preprocessing import MinMaxScaler

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.hbos import HBOS
from TSB_UAD.models.sand import SAND
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from custom_models import IForestOnline, HBOSOnline, MPOnline
from TSB_UAD.models.matrix_profile import MatrixProfile

from figures_creation import plotFig

def prepare_data_unsupervised(
        filepath: Path = None,
):
    df = pd.read_csv(filepath, header=None).dropna().to_numpy()
    name = filepath.stem

    print(f"Data shape is: {df.shape}")
    # print(f"Data header is: {df[:5]}")

    data = df[:,0].astype(float)
    label = df[:,1].astype(int)

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

    # Check if exp_folder already exists
    if exp_folder.exists():
        print(f"Experiment folder {exp_folder} already exists. Skipping...")
        return

    exp_folder.mkdir(parents=True, exist_ok=True)

    data, X_data, label, name, slidingWindow = prepare_data_unsupervised(
        data_file,
    )

    # --- timing starts here ---
    start_time = time.perf_counter()
    match method:
        case "SAND":
            if type == "offline":
                modelName='SAND (offline)'
                clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
                clf.fit(data,overlaping_rate=int(1.5*slidingWindow))
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                
            elif type == "online":
                modelName='SAND (online)'
                clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
                clf.fit(data,online=True,alpha=0.5,init_length=5000,batch_size=2000,verbose=True,overlaping_rate=int(4*slidingWindow))
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                
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
                

            elif type == "online":
                if online_variant=="naive":
                    modelName='IForest (online) (naive)'
                    base_classifier = IForest(n_jobs=1)
                    clf = IForestOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                    

                elif online_variant=="movingAverage":
                    modelName='IForest (online) (movingAverage)'
                    base_classifier = IForest(n_jobs=1)
                    clf = IForestOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        if_models_buffer_size=5,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                
                elif online_variant=="precedingHistory":
                    modelName='IForest (online) (precedingHistory)'
                    base_classifier = IForest(n_jobs=1)
                    clf = IForestOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        num_windows=5,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                
                elif online_variant=="adaptiveDecay":
                     modelName = 'IForest (online) (adaptiveDecay)'
                     base_classifier = IForest(n_jobs=1)
                     clf = IForestOnline(
                        base_classifier=base_classifier,
                        method=online_variant,   # already in kwargs list
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        if_models_buffer_size=5,
                        decay_lambda=math.log(2)/3,
                        replace_frac=0.2,
                        adwin_delta=0.002,
                     )
                     clf.fit(data)
                     score = clf.decision_scores_

                else:
                    raise NotImplementedError(f"Online variant {online_variant} not implemented yet for {method}")
            else:
                raise ValueError(f"Unknown type: {type}")

        case "HBOS":
            if type == "offline":
                modelName='HBOS (offline)'
                clf = HBOS(
                    n_bins=10,
                    alpha=np.float64(0.1),
                    tol=np.float64(0.5),
                    contamination=np.float64(0.1),
                )
                clf.fit(X_data)
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
                

            elif type == "online":
                if online_variant=="naive":
                    modelName='HBOS (online) (naive)'
                    base_classifier = HBOS(
                        n_bins=10,
                        alpha=np.float64(0.1),
                        tol=np.float64(0.5),
                        contamination=np.float64(0.1),
                    )
                    clf = HBOSOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                    

                elif online_variant=="movingAverage":
                    modelName='HBOS (online) (movingAverage)'
                    base_classifier = HBOS(
                        n_bins=10,
                        alpha=np.float64(0.1),
                        tol=np.float64(0.5),
                        contamination=np.float64(0.1),
                    )
                    clf = HBOSOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        hbos_models_buffer_size=5,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                
                elif online_variant=="precedingHistory":
                    modelName='HBOS (online) (precedingHistory)'
                    base_classifier = HBOS(
                        n_bins=10,
                        alpha=np.float64(0.1),
                        tol=np.float64(0.5),
                        contamination=np.float64(0.1),
                    )
                    clf = HBOSOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        num_windows=5,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                
                elif online_variant == "adaptiveDecay":
                    modelName = 'HBOS (online) (adaptiveDecay)'
                    base_classifier = HBOS(
                        n_bins=10,
                        alpha=np.float64(0.1),
                        tol=np.float64(0.5),
                        contamination=np.float64(0.1),
                    )
                    clf = HBOSOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        hbos_models_buffer_size=5,
                        decay_lambda=math.log(2)/3,
                        replace_frac=0.2,
                        adwin_delta=0.002,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                    
                else:
                    raise NotImplementedError(f"Online variant {online_variant} not implemented yet for {method}")
            else:
                raise ValueError(f"Unknown type: {type}")

        case "MatrixProfile":
            if type == "offline":
                modelName='MatrixProfile'
                clf = MatrixProfile(window = slidingWindow)
                clf.fit(data)
                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
                                
            elif type == "online":
                if online_variant=="naive":
                    modelName='MatrixProfile (online) (naive)'
                    base_classifier = MatrixProfile(window = slidingWindow)
                    clf = MPOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                    
                elif online_variant=="movingAverage":
                    raise NotImplementedError(f"Online variant {online_variant} not implemented for {method}")
                    
                
                elif online_variant=="precedingHistory":
                    modelName='MatrixProfile (online) (precedingHistory)'
                    base_classifier = MatrixProfile(window = slidingWindow)
                    clf = MPOnline(
                        base_classifier=base_classifier,
                        method=online_variant,
                        batch_size=2000,
                        slidingWindow=slidingWindow,
                        num_windows = 5,
                    )
                    clf.fit(data)
                    score = clf.decision_scores_
                           
                else:
                    raise NotImplementedError(f"Online variant {online_variant} not implemented yet for {method}")
            else:
                raise ValueError(f"Unknown type: {type}")

        case _:
            raise ValueError(f"Unknown method: {method}")

    # --- timing ends here ---
    end_time = time.perf_counter()
    duration = end_time - start_time

    plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName, exp_folder=str(exp_folder), save_plots=cfg.get("save_plots", False)) 
    
    # save timing info
    timing_info = {
        "method": method,
        "type": type,
        "online_variant": online_variant,
        "data_file": data_file.stem,
        "execution_time_seconds": duration
    }
    timings_path = exp_folder / "execution_time.json"
    with open(timings_path, "w") as f:
        json.dump(timing_info, f, indent=2)

    print(f"[INFO] Execution took {duration:.4f} seconds; saved to {timings_path}")


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
        num_of_normalities_under_test=params.get("num_of_normalities_under_test", [1,2,3]),
        normalities_under_test=params.get("normalities_under_test", ["SVDB","ECG","Dodgers"]),
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