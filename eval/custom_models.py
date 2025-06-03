'''
@File    :   custom_models.py
@Time    :   05/2025
@Author  :   nikifori
@Version :   -
'''
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
        method: str = "naive",  # naive, movingAverage
        batch_size: int = 2000,
        slidingWindow: int = 100,
        if_models_buffer_size: int = 5,
        num_windows: int = 5,
    ):  
        self.base_classifier = base_classifier
        self.method = method
        self.batch_size = batch_size
        self.slidingWindow = slidingWindow
        self.if_models_buffer_size = if_models_buffer_size
        self.if_models_buffer = []
        self.decision_scores_ = None
        self.num_windows = num_windows

    def fit(self, X):
        if self.method == "naive":
            result = []
            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = Window(window = self.slidingWindow).convert(X[i: i + self.batch_size]).to_numpy()
                self.base_classifier.fit(X_batch)
                score = self.base_classifier.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
                result.append(score)
            result = np.concatenate(result)
            self.decision_scores_ = result
            return self

        elif self.method == "movingAverage":
            result = []
            # reset buffer
            self.if_models_buffer = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = Window(window = self.slidingWindow).convert(X[i: i + self.batch_size]).to_numpy()
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(X_batch)

                self.if_models_buffer.append(clf)
                if len(self.if_models_buffer) > self.if_models_buffer_size:
                    self.if_models_buffer.pop(0)

                scores_list = []
                for model in self.if_models_buffer:
                    score = -model.detector_.score_samples(X_batch)
                    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                    score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
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
                X_batch = Window(window = self.slidingWindow).convert(X[i: i + self.batch_size]).to_numpy()

                if self.num_windows > 1 and len(self.windows_buffer) >= 1:
                    # take the last (num_windows - 1) windows
                    train_windows = self.windows_buffer[-(self.num_windows - 1):]
                    new_X_train = np.vstack(train_windows + [X_batch])
                else:
                    # if not enough history yet, fall back to current window only
                    new_X_train = X_batch
                
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(new_X_train)

                score = -clf.detector_.score_samples(new_X_train)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                score = score[-int(len(score) / (len(train_windows) + 1)) :]
                score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
                result.append(score)

                self.windows_buffer.append(X_batch)
                if len(self.windows_buffer) > self.num_windows:
                    self.windows_buffer.pop(0)
            
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
    ):  
        self.base_classifier = base_classifier
        self.method = method
        self.batch_size = batch_size
        self.slidingWindow = slidingWindow
        self.hbos_models_buffer_size = hbos_models_buffer_size
        self.hbos_models_buffer = []
        self.decision_scores_ = None
        self.num_windows = num_windows

    def fit(self, X):
        if self.method == "naive":
            result = []
            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = Window(window = self.slidingWindow).convert(X[i: i + self.batch_size]).to_numpy()
                self.base_classifier.fit(X_batch)
                score = self.base_classifier.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
                result.append(score)
            result = np.concatenate(result)
            self.decision_scores_ = result
            return self

        elif self.method == "movingAverage":
            result = []
            # reset buffer
            self.hbos_models_buffer = []

            for i in tqdm(range(0, len(X), self.batch_size)):
                X_batch = Window(window = self.slidingWindow).convert(X[i: i + self.batch_size]).to_numpy()
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(X_batch)

                self.hbos_models_buffer.append(clf)
                if len(self.hbos_models_buffer) > self.hbos_models_buffer_size:
                    self.hbos_models_buffer.pop(0)

                scores_list = []
                for model in self.hbos_models_buffer:
                    score = model.decision_function(X_batch)
                    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                    score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
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
                X_batch = Window(window = self.slidingWindow).convert(X[i: i + self.batch_size]).to_numpy()

                if self.num_windows > 1 and len(self.windows_buffer) >= 1:
                    # take the last (num_windows - 1) windows
                    train_windows = self.windows_buffer[-(self.num_windows - 1):]
                    new_X_train = np.vstack(train_windows + [X_batch])
                else:
                    # if not enough history yet, fall back to current window only
                    new_X_train = X_batch
                
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(new_X_train)

                score = -clf.decision_function(new_X_train)
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                score = score[-int(len(score) / (len(train_windows) + 1)) :]
                score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
                result.append(score)

                self.windows_buffer.append(X_batch)
                if len(self.windows_buffer) > self.num_windows:
                    self.windows_buffer.pop(0)
            
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
                self.base_classifier.fit(X[i: i + self.batch_size])
                score = self.base_classifier.decision_scores_
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))
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
                X_batch = X[i: i + self.batch_size]

                if self.num_windows > 1 and len(self.windows_buffer) >= 1:
                    # take the last (num_windows - 1) windows
                    train_windows = self.windows_buffer[-(self.num_windows - 1):]
                    new_X_train = np.concat(train_windows + [X_batch])
                else:
                    # if not enough history yet, fall back to current window only
                    new_X_train = X_batch
                
                clf = copy.deepcopy(self.base_classifier)
                clf.fit(new_X_train)

                score = clf.decision_scores_
                score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                temp_window = self.batch_size - (int(len(score) / (len(train_windows) + 1))) + 1
                score = score[-int(len(score) / (len(train_windows) + 1)) :]
                score = np.array([score[0]]*math.ceil((temp_window-1)/2) + list(score) + [score[-1]]*((temp_window-1)//2))
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


if __name__ == '__main__':
    main()