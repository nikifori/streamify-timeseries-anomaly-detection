from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
import os
import json
from sklearn.metrics import roc_curve, auc

import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

from TSB_UAD.vus.metrics import metricor
from TSB_UAD.vus.metrics import get_metrics


def plotFig(data, label, score, slidingWindow,
            fileName, modelName,
            exp_folder=None, plotRange=None):

    grader = metricor()
    range_anomaly = grader.range_convers_new(label)
    max_length = len(score)

    if plotRange is None:
        plotRange = [0, max_length]

    # --- calculate all metrics and save to pickle if needed ---
    metrics = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
    if exp_folder:
        metrics_json = {
            key: float(val) for key, val in metrics.items() 
        }
        os.makedirs(exp_folder, exist_ok=True)
        metrics_path = os.path.join(exp_folder, f"{fileName}_metrics.json")
        with open(metrics_path, "w") as mf:
            json.dump(metrics_json, mf, indent=2)

    # create figure
    fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)

    # 1) Raw data with anomalies highlighted
    ax1 = fig3.add_subplot(gs[0, :-1])
    ax1.tick_params(labelbottom=False)
    ax1.plot(data[:max_length], 'k')
    for r in range_anomaly:
        if r[0] == r[1]:
            ax1.plot(r[0], data[r[0]], 'r.')
        else:
            ax1.plot(range(r[0], r[1] + 1),
                     data[r[0]:r[1] + 1], 'r')
    ax1.set_xlim(plotRange)

    # 2) Score over time with threshold
    ax2 = fig3.add_subplot(gs[1, :-1])
    ax2.plot(score[:max_length])
    threshold = np.mean(score) + 3 * np.std(score)
    ax2.hlines(threshold, 0, max_length,
               linestyles='--', color='red')
    ax2.set_ylabel('score')
    ax2.set_xlim(plotRange)

    # 3) Scatter with TP/FP/TN/FN coloring
    ax3 = fig3.add_subplot(gs[2, :-1])
    index = label + 2 * (score > threshold)
    cf = lambda x: 'k' if x == 0 else ('r' if x == 1 else ('g' if x == 2 else 'b'))
    color = np.vectorize(cf)(index[:max_length])
    patches = [
        mpatches.Patch(color='black', label='TN'),
        mpatches.Patch(color='red',   label='FN'),
        mpatches.Patch(color='green', label='FP'),
        mpatches.Patch(color='blue',  label='TP')
    ]
    ax3.scatter(np.arange(max_length),
                data[:max_length], c=color, marker='.')
    ax3.legend(handles=patches, loc='best')
    ax3.set_xlim(plotRange)

    # save or show
    if exp_folder:
        os.makedirs(exp_folder, exist_ok=True)
        save_path = os.path.join(exp_folder, fileName)
        fig3.savefig(save_path)
        plt.close(fig3)
    else:
        plt.show()