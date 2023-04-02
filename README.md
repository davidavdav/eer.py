# Equal Error Rate

The Equal Error Rate is an overall performance metric for a binary classifier.  It is insensitive to

 - evaluation priors, so the metric doesn't change if the relative amounts of data points in the two classes varies
 - calibration, so the classifier only needs to produce consistent scores, and not set a threshold

This package is a thin wrapper around [llreval](https://github.com/davidavdav/llreval), which computes the equal error rate
in the ROC convex hull interpretation, which is consistent and meaningful, see [Niko Brümmer's PhD thesis](http://hdl.handle.net/10019.1/5139).

## Installation

```sh
pip install eer
```

## Usage

### `eer(scores, labels)`

 - Collect your classifier's (floating point) scores in a `numpy` array.
 - Prepare a parallel array with values `0` for the class related to lower scores, and `1` for the class related to higher scores.

### `eer_tnt(target_scores, nontarget_scores)`

 - Collect your classifiers scores in separate `numpy` arrays, one for the positive class (`target_scores`) and one for the negative class (`nontarget_scores`)

### Example

Simulate [calibrated scores](https://www.isca-speech.org/archive/interspeech_2013/leeuwen13_interspeech.html) for a binary classifier, and compute the equal error rate.
```python
import numpy as np
from eer import eer, eer_tnt

ntar, nnon = 100_000, 1_000_000
targets = 2 + 2 * np.random.randn(ntar)
non_targets = -2 + 2 * np.random.randn(nnon)
print("EER using target and non-target scores:", eer_tnt(targets, non_targets))

scores = np.concatenate([targets, non_targets])
labels = np.concatenate([np.ones(targets.shape[0]), np.zeros(non_targets.shape[0])])
print("EER using scores and labels:", eer(scores, labels))

```
The values printed should not be too far from [0.5 + 0.5erf(-1/√2) ≈ 0.1586552](https://github.com/davidavdav/ROCAnalysis.jl).

