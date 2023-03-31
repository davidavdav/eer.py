#!/usr/bin/env python3

## eer.py.  Thin wrapper around PYLLR.py to make it easy to just compute the equal error rate
## using the convex hull ROC interpretation.

## (c) 2023 David A. van Leeuwen

from typing import List, Union

import numpy as np

from pyllr.pav_rocch import PAV, ROCCH

def _eer(scores: np.ndarray, labels: np.ndarray):
    """Compute the EER, in the convex hull ROC interpretation, for numeric scores and labels"""
    pav = PAV(scores, labels)
    rocch = ROCCH(pav)
    eer = rocch.EER()
    return float(eer)

def eer(scores: Union[np.ndarray, List[float]], labels: Union[np.ndarray, List[int]]) -> float:
    """
    Calculate the equal error rate for a list of labelled scores.

    This computes the ROC convex hull interpretation of the EER.

    Positive labels should be 1, and negative scores should be 0.

    :param scores: a list of prediction float values, or numpy array
    :param labels: a list of class integer values (either 0 or 1)
    :return: a float value containing the equal error rate
    """
    if isinstance(scores, list):
        scores = np.asarray(scores, dtype=np.float64)
    elif not (isinstance(scores, np.ndarray) and scores.dtype in [np.float32, np.float64] and scores.ndim == 1):
        raise ValueError("`scores` should be an 1 dimensional numpy.ndarray(dtype=float) or list of floats")
    if isinstance(labels, list):
        labels = np.asarray(labels, dtype=np.float64)
    elif not (isinstance(labels, np.ndarray) and labels.ndim == 1):
        raise ValueError("`labels` should be an 1-dimensional numpy.ndarray or list")
    elif labels.dtype == np.dtype('bool') or np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.float64)
    elif labels.dtype not in [np.float32, np.float64]:
        raise ValueError("`labels` should be a numeric or boolean numpy array")
    if scores.shape != labels.shape:
        raise ValueError("`scores` and `labels` should have the same shape")
    return _eer(scores, labels)

def eer_tnt(target_scores: Union[np.ndarray, List[float]], nontarget_scores: Union[np.ndarray, List[float]]) -> float:
    """Calculate the equal error rate for a list of target and a list of nontarget scores.

    This computes the ROC convex hull interpretation of the EER.

    Args:
        target_scores (Union[np.ndarray, List[float]]): a list or ndarray of the target scores, tending towards higher values
        nontarget_scores (Union[np.ndarray, List[float]]): a list or ndarray of the non-target scores, tending towards lower values

    Returns:
        float: a float value containing the equal error rate
    """
    if isinstance(target_scores, list):
        target_scores = np.asarray(target_scores, dtype=np.float64)
    if isinstance(nontarget_scores, list):
        nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)

    scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate([np.ones(target_scores.shape[0]), np.zeros(nontarget_scores.shape[0])])

    return _eer(scores, labels)