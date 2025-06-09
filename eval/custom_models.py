"""
@File    :   custom_models.py
@Time    :   05/2025
@Author  :   nikifori
@Version :   -
"""

import copy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from tqdm import tqdm

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.feature import Window
from TSB_UAD.models.hbos import HBOS
from TSB_UAD.models.matrix_profile import MatrixProfile


class IForestOnline:
    def __init__(
        self,
        base_classifier: IForest,
        method: str = "naive",  # naive, movingAverage, precedingHistory, adaptiveDecay
        batch_size: int = 2000,
        slidingWindow: int = 100,
        if_models_buffer_size: int = 5,
        num_windows: int = 5,
        decay_lambda: float = math.log(2) / 3,  # half-life = 3 batches
        replace_frac: float = 0.2,  # 20 % of trees refreshed / batch
        adwin_delta: float = 0.002,  # ADWIN sensitivity
    ):
        self.base_classifier = base_classifier
        self.method = method
        self.batch_size = batch_size
        self.slidingWindow = slidingWindow
        self.if_models_buffer_size = if_models_buffer_size
        self.if_models_buffer = []
        self.decision_scores_ = None
        self.num_windows = num_windows

        self.decay_lambda = decay_lambda
        self.replace_frac = replace_frac
        self.adwin_delta = adwin_delta
        self.adwin = None  # created lazily

    def fit(self, X):
        if self.method == "naive":
            result = []
            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )
                self.base_classifier.fit(X_batch)
                score = self.base_classifier.decision_scores_
                score = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(score.reshape(-1, 1))
                    .ravel()
                )
                score = np.array(
                    [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                    + list(score)
                    + [score[-1]] * ((self.slidingWindow - 1) // 2)
                )
                result.append(score)
            result = np.concatenate(result)
            self.decision_scores_ = result
            return self

        elif self.method == "movingAverage":
            result = []
            # reset buffer
            self.if_models_buffer = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(X_batch)

                self.if_models_buffer.append(clf)
                if len(self.if_models_buffer) > self.if_models_buffer_size:
                    self.if_models_buffer.pop(0)

                scores_list = []
                for model in self.if_models_buffer:
                    score = -model.detector_.score_samples(X_batch)
                    score = (
                        MinMaxScaler(feature_range=(0, 1))
                        .fit_transform(score.reshape(-1, 1))
                        .ravel()
                    )
                    score = np.array(
                        [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(score)
                        + [score[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    scores_list.append(score)

                n = len(scores_list)
                weights = np.arange(1, n + 1, dtype=float)
                # weights = np.square(weights)
                weights /= weights.sum()

                stacked = np.vstack(scores_list)  # shape (n, batch_size)
                avg_scores = np.dot(weights, stacked)  # shape (batch_size,)

                result.append(avg_scores)

            self.decision_scores_ = np.concatenate(result)
            return self

        elif self.method == "precedingHistory":
            result = []
            # reset buffer of raw windows (no models)
            self.windows_buffer = []
            train_windows = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )

                if self.num_windows > 1 and len(self.windows_buffer) >= 1:
                    # take the last (num_windows - 1) windows
                    train_windows = self.windows_buffer[-(self.num_windows - 1) :]
                    new_X_train = np.vstack(train_windows + [X_batch])
                else:
                    # if not enough history yet, fall back to current window only
                    new_X_train = X_batch

                clf = copy.deepcopy(self.base_classifier)
                clf.fit(new_X_train)

                score = -clf.detector_.score_samples(new_X_train)
                score = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(score.reshape(-1, 1))
                    .ravel()
                )
                score = score[-int(len(score) / (len(train_windows) + 1)) :]
                score = np.array(
                    [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                    + list(score)
                    + [score[-1]] * ((self.slidingWindow - 1) // 2)
                )
                result.append(score)

                self.windows_buffer.append(X_batch)
                if len(self.windows_buffer) > self.num_windows:
                    self.windows_buffer.pop(0)

            self.decision_scores_ = np.concatenate(result)
            return self

        elif self.method == "adaptiveDecay":  # ← Variant name (feel free to shorten)
            from river.drift import ADWIN

            result, self.if_models_buffer = [], []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )

                # --- lazy init ---
                if self.adwin is None:
                    self.adwin = ADWIN(
                        delta=self.adwin_delta, clock=int(0.25 * self.batch_size)
                    )

                # 1) age & weight existing forests
                aged_scores, weights = [], []
                for mdl_idx, (mdl, t_stamp, w) in enumerate(self.if_models_buffer):
                    w *= math.exp(-self.decay_lambda)  # exponential decay
                    self.if_models_buffer[mdl_idx] = (mdl, t_stamp, w)
                    sc = -mdl.detector_.score_samples(X_batch)
                    sc = MinMaxScaler((0, 1)).fit_transform(sc.reshape(-1, 1)).ravel()
                    sc = np.array(
                        [sc[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(sc)
                        + [sc[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    aged_scores.append(sc)
                    weights.append(w)

                # 2) detect drift on median score
                if len(aged_scores):
                    for point in X[i : i + self.batch_size]:
                        self.adwin.update(point)
                        if self.adwin.drift_detected:
                            # fast-forget: halve all weights
                            self.if_models_buffer = [
                                (m, t, w * 0.5) for m, t, w in self.if_models_buffer
                            ]
                            weights = [w * 0.5 for w in weights]
                            print(f"[INFO] Found drift at {i} batch.")
                            break

                # 3) partial refresh: replace lowest-weight trees

                if len(self.if_models_buffer) >= self.if_models_buffer_size:
                    n_replace = max(
                        1, int(self.replace_frac * max(1, len(self.if_models_buffer)))
                    )
                    victims = np.argsort([w for _, _, w in self.if_models_buffer])[
                        :n_replace
                    ]
                    for idx in sorted(victims, reverse=True):
                        self.if_models_buffer.pop(idx)
                new_clf = copy.deepcopy(self.base_classifier)
                new_clf.fit(X_batch)
                self.if_models_buffer.append((new_clf, i, 1.0))

                # cap buffer
                self.if_models_buffer = sorted(
                    self.if_models_buffer, key=lambda v: v[2]
                )[-self.if_models_buffer_size :]

                # 4) weighted score aggregation
                if not aged_scores:  # first batch
                    sc = -new_clf.detector_.score_samples(X_batch)
                    sc = MinMaxScaler((0, 1)).fit_transform(sc.reshape(-1, 1)).ravel()
                    sc = np.array(
                        [sc[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(sc)
                        + [sc[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    result.append(sc)
                else:
                    # include fresh model in arrays
                    aged_scores.append(result_sc := sc)  # sc from new_clf above
                    weights.append(1.0)
                    w_arr = np.array(weights) / np.sum(weights)
                    stacked = np.vstack(aged_scores)
                    result.append(np.dot(w_arr, stacked))

            self.decision_scores_ = np.concatenate(result)
            return self

        else:
            raise NotImplementedError(f"Method {self.method} not implemented yet")


class HBOSOnline:
    def __init__(
        self,
        base_classifier: HBOS,
        method: str = "naive",  # naive, movingAverage
        batch_size: int = 2000,
        slidingWindow: int = 100,
        hbos_models_buffer_size: int = 5,
        num_windows: int = 5,
        decay_lambda: float = math.log(2) / 3,  # half-life = 3 batches
        replace_frac: float = 0.2,  # 20 % of trees refreshed / batch
        adwin_delta: float = 0.002,  # ADWIN sensitivity
    ):
        self.base_classifier = base_classifier
        self.method = method
        self.batch_size = batch_size
        self.slidingWindow = slidingWindow
        self.hbos_models_buffer_size = hbos_models_buffer_size
        self.hbos_models_buffer = []
        self.decision_scores_ = None
        self.num_windows = num_windows
        self.decay_lambda = decay_lambda
        self.replace_frac = replace_frac
        self.adwin_delta = adwin_delta
        self.adwin = None

    def fit(self, X):
        if self.method == "naive":
            result = []
            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )
                self.base_classifier.fit(X_batch)
                score = self.base_classifier.decision_scores_
                score = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(score.reshape(-1, 1))
                    .ravel()
                )
                score = np.array(
                    [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                    + list(score)
                    + [score[-1]] * ((self.slidingWindow - 1) // 2)
                )
                result.append(score)
            result = np.concatenate(result)
            self.decision_scores_ = result
            return self

        elif self.method == "movingAverage":
            result = []
            # reset buffer
            self.hbos_models_buffer = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(X_batch)

                self.hbos_models_buffer.append(clf)
                if len(self.hbos_models_buffer) > self.hbos_models_buffer_size:
                    self.hbos_models_buffer.pop(0)

                scores_list = []
                for model in self.hbos_models_buffer:
                    score = model.decision_function(X_batch)
                    score = (
                        MinMaxScaler(feature_range=(0, 1))
                        .fit_transform(score.reshape(-1, 1))
                        .ravel()
                    )
                    score = np.array(
                        [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(score)
                        + [score[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    scores_list.append(score)

                n = len(scores_list)
                weights = np.arange(1, n + 1, dtype=float)
                # weights = np.square(weights)
                weights /= weights.sum()

                stacked = np.vstack(scores_list)  # shape (n, batch_size)
                avg_scores = np.dot(weights, stacked)  # shape (batch_size,)

                result.append(avg_scores)

            self.decision_scores_ = np.concatenate(result)
            return self

        elif self.method == "precedingHistory":
            result = []
            # reset buffer of raw windows (no models)
            self.windows_buffer = []
            train_windows = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )

                if self.num_windows > 1 and len(self.windows_buffer) >= 1:
                    # take the last (num_windows - 1) windows
                    train_windows = self.windows_buffer[-(self.num_windows - 1) :]
                    new_X_train = np.vstack(train_windows + [X_batch])
                else:
                    # if not enough history yet, fall back to current window only
                    new_X_train = X_batch

                clf = copy.deepcopy(self.base_classifier)
                clf.fit(new_X_train)

                score = clf.decision_function(new_X_train)
                score = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(score.reshape(-1, 1))
                    .ravel()
                )
                score = score[-int(len(score) / (len(train_windows) + 1)) :]
                score = np.array(
                    [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                    + list(score)
                    + [score[-1]] * ((self.slidingWindow - 1) // 2)
                )
                result.append(score)

                self.windows_buffer.append(X_batch)
                if len(self.windows_buffer) > self.num_windows:
                    self.windows_buffer.pop(0)

            self.decision_scores_ = np.concatenate(result)
            return self

        elif self.method == "adaptiveDecay":
            from river.drift import ADWIN

            # prepare output and reset buffer
            result = []
            self.hbos_models_buffer = []

            # process each batch
            for i in tqdm(range(0, len(X), self.batch_size)):
                # 1) extract and window the batch
                X_batch = (
                    Window(window=self.slidingWindow)
                    .convert(X[i : i + self.batch_size])
                    .to_numpy()
                )

                # 2) lazy-initialize ADWIN
                if self.adwin is None:
                    self.adwin = ADWIN(
                        delta=self.adwin_delta, clock=int(0.25 * self.batch_size)
                    )

                # 3) age existing models and collect their scores
                aged_scores = []
                weights = []
                for idx, (mdl, ts, w) in enumerate(self.hbos_models_buffer):
                    # exponential decay
                    w *= math.exp(-self.decay_lambda)
                    self.hbos_models_buffer[idx] = (mdl, ts, w)

                    # score and normalize
                    sc = mdl.decision_function(X_batch)
                    sc = (
                        MinMaxScaler(feature_range=(0, 1))
                        .fit_transform(sc.reshape(-1, 1))
                        .ravel()
                    )
                    sc = np.array(
                        [sc[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(sc)
                        + [sc[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    aged_scores.append(sc)
                    weights.append(w)

                # 4) detect drift on median score
                if len(aged_scores):
                    for point in X[i : i + self.batch_size]:
                        self.adwin.update(point)
                        if self.adwin.drift_detected:
                            # fast-forget: halve all weights
                            self.hbos_models_buffer = [
                                (m, t, w * 0.5) for m, t, w in self.hbos_models_buffer
                            ]
                            weights = [w * 0.5 for w in weights]
                            print(f"[INFO] Found drift at {i} batch.")
                            break

                # 5) partial refresh: replace lowest-weight trees

                if len(self.hbos_models_buffer) >= self.hbos_models_buffer_size:
                    n_replace = max(
                        1, int(self.replace_frac * max(1, len(self.hbos_models_buffer)))
                    )
                    victims = np.argsort([w for _, _, w in self.hbos_models_buffer])[
                        :n_replace
                    ]
                    for idx in sorted(victims, reverse=True):
                        self.hbos_models_buffer.pop(idx)
                new_clf = copy.deepcopy(self.base_classifier)
                new_clf.fit(X_batch)
                self.hbos_models_buffer.append((new_clf, i, 1.0))

                # train a fresh model on current batch
                new_mdl = copy.deepcopy(self.base_classifier)
                new_mdl.fit(X_batch)
                self.hbos_models_buffer.append((new_mdl, i, 1.0))

                # enforce buffer size cap
                self.hbos_models_buffer = sorted(
                    self.hbos_models_buffer, key=lambda t: t[2]
                )[-self.hbos_models_buffer_size :]

                # 6) aggregate scores
                if not aged_scores:
                    # first batch: only the new model
                    sc = new_mdl.decision_function(X_batch)
                    sc = (
                        MinMaxScaler(feature_range=(0, 1))
                        .fit_transform(sc.reshape(-1, 1))
                        .ravel()
                    )
                    sc = np.array(
                        [sc[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(sc)
                        + [sc[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    result.append(sc)
                else:
                    # include latest model’s scores
                    sc = new_mdl.decision_function(X_batch)
                    sc = (
                        MinMaxScaler(feature_range=(0, 1))
                        .fit_transform(sc.reshape(-1, 1))
                        .ravel()
                    )
                    sc = np.array(
                        [sc[0]] * math.ceil((self.slidingWindow - 1) / 2)
                        + list(sc)
                        + [sc[-1]] * ((self.slidingWindow - 1) // 2)
                    )
                    aged_scores.append(sc)
                    weights.append(1.0)

                    # weighted average across all models
                    w_arr = np.array(weights)
                    w_arr /= w_arr.sum()
                    stacked = np.vstack(aged_scores)
                    result.append(w_arr.dot(stacked))

            # finalize
            self.decision_scores_ = np.concatenate(result)
            return self

        else:
            raise NotImplementedError(f"Method {self.method} not implemented yet")


class MPOnline:
    def __init__(
        self,
        base_classifier: MatrixProfile,
        method: str = "naive",  # naive, movingAverage
        batch_size: int = 2000,
        slidingWindow: int = 100,
        num_windows: int = 5,
    ):
        self.base_classifier = base_classifier
        self.method = method
        self.batch_size = batch_size
        self.slidingWindow = slidingWindow
        self.mp_models_buffer = []
        self.decision_scores_ = None
        self.num_windows = num_windows

    def fit(self, X):
        if self.method == "naive":
            result = []
            for i in tqdm(range(0, len(X), self.batch_size)):
                self.base_classifier.fit(X[i : i + self.batch_size])
                score = self.base_classifier.decision_scores_
                score = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(score.reshape(-1, 1))
                    .ravel()
                )
                score = np.array(
                    [score[0]] * math.ceil((self.slidingWindow - 1) / 2)
                    + list(score)
                    + [score[-1]] * ((self.slidingWindow - 1) // 2)
                )
                result.append(score)
            result = np.concatenate(result)
            self.decision_scores_ = result
            return self

        elif self.method == "movingAverage":
            raise NotImplementedError("Moving average not implemented for MPOnline")

        elif self.method == "precedingHistory":
            result = []
            # reset buffer of raw windows (no models)
            self.windows_buffer = []
            train_windows = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = X[i : i + self.batch_size]

                if self.num_windows > 1 and len(self.windows_buffer) >= 1:
                    # take the last (num_windows - 1) windows
                    train_windows = self.windows_buffer[-(self.num_windows - 1) :]
                    new_X_train = np.concat(train_windows + [X_batch])
                else:
                    # if not enough history yet, fall back to current window only
                    new_X_train = X_batch

                clf = copy.deepcopy(self.base_classifier)
                clf.fit(new_X_train)

                score = clf.decision_scores_
                score = (
                    MinMaxScaler(feature_range=(0, 1))
                    .fit_transform(score.reshape(-1, 1))
                    .ravel()
                )
                temp_window = (
                    self.batch_size - (int(len(score) / (len(train_windows) + 1))) + 1
                )
                score = score[-int(len(score) / (len(train_windows) + 1)) :]
                score = np.array(
                    [score[0]] * math.ceil((temp_window - 1) / 2)
                    + list(score)
                    + [score[-1]] * ((temp_window - 1) // 2)
                )
                result.append(score)

                self.windows_buffer.append(X_batch)
                if len(self.windows_buffer) > self.num_windows:
                    self.windows_buffer.pop(0)

            self.decision_scores_ = np.concatenate(result)
            return self

        else:
            raise NotImplementedError(f"Method {self.method} not implemented yet")


def main():
    pass


if __name__ == "__main__":
    main()
