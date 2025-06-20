import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

class ConvNet(nn.Module):  # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        return self.model(x)

class DecisionTree:
    def __init__(self, max_depth: int = 10, criterion: str = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = 5
        self.max_features = None
        self.criterion = criterion

    def fit(self, X, y):
        X = np.array(X) if not hasattr(X, 'shape') else X
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'value': int(np.bincount(y).argmax())}

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return {'value': int(np.bincount(y).argmax())}

        left_X, left_y, right_X, right_y = self._split_data(X, y, feature_idx, threshold)
        return {
            'feature_index': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def predict(self, X) -> np.ndarray:
        X = np.array(X) if not hasattr(X, 'shape') else X
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x: np.ndarray, node: dict) -> int:
        if 'value' in node:
            return node['value']
        if x[node['feature_index']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])

    def _split_data(self, X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = X[:, feature_index] <= threshold
        return X[mask], y[mask], X[~mask], y[~mask]

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        best_gain = 0.0
        best_idx, best_thr = None, None
        base_impurity = self._entropy(y)
        n_samples, n_features = X.shape

        feat_indices = np.arange(n_features) if self.max_features is None else \
                       np.random.choice(n_features, self.max_features, replace=False)
        for idx in feat_indices:
            vals = X[:, idx]
            thresholds = np.unique(np.percentile(vals, np.arange(5, 100, 5)))
            for thr in thresholds:
                mask = vals <= thr
                if mask.all() or not mask.any():
                    continue
                left_y, right_y = y[mask], y[~mask]
                if left_y.size < self.min_samples_split or right_y.size < self.min_samples_split:
                    continue
                p_l, p_r = left_y.size / n_samples, right_y.size / n_samples
                gain = base_impurity - (p_l * self._entropy(left_y) + p_r * self._entropy(right_y))
                if gain > best_gain:
                    best_gain, best_idx, best_thr = gain, idx, thr
        return best_idx, best_thr

    def _entropy(self, y: np.ndarray) -> float:
        counts = np.bincount(y)
        ps = counts[counts > 0] / counts.sum()
        return -np.sum(ps * np.log2(ps))

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device: torch.device) -> Tuple[List[np.ndarray], List[int]]:
    model.eval()
    feats: List[np.ndarray] = []
    labs: List[int] = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            out = model(imgs).cpu().numpy()
            for vec, lbl in zip(out, targets.numpy()):
                feats.append(vec)
                labs.append(int(lbl))
    return feats, labs


def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device: torch.device) -> Tuple[List[np.ndarray], List[str]]:
    model.eval()
    feats: List[np.ndarray] = []
    paths: List[str] = []
    with torch.no_grad():
        for imgs, img_paths in dataloader:
            imgs = imgs.to(device)
            out = model(imgs).cpu().numpy()
            for vec, path in zip(out, img_paths):
                feats.append(vec)
                paths.append(path)
    return feats, paths
