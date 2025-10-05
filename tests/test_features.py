# tests/test_features.py
import pandas as pd
from src.evaluation.ab_test import _to_labels

def test_thresholding_basic():
    import numpy as np
    scores = np.array([0.1, 0.5, 0.51, 0.9])
    y = _to_labels(scores, 0.5)
    assert y.tolist() == [0, 1, 1, 1]
