#!/usr/bin/env python3

import unittest

import numpy as np

from eer import eer, eer_tnt

class TestEER(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        ntar, nnon = 100_000, 1_000_000
        self.targets = 2 + 2 * np.random.randn(ntar)
        self.non_targets = -2 + 2 * np.random.randn(nnon)

    def test_largish(self):
        scores = np.concatenate([self.targets, self.non_targets])
        labels = np.concatenate([np.ones(self.targets.shape[0]), np.zeros(self.non_targets.shape[0])])
        e = eer(scores, labels)
        self.assertTrue(0.15 < e < 0.165)

    def test_largish_tnt(self):
        self.assertTrue(0.15 < eer_tnt(self.targets, self.non_targets) < 0.165)

    def test_largish_list(self):
        scores = np.concatenate([self.targets, self.non_targets]).tolist()
        labels = np.concatenate([np.ones(self.targets.shape[0]), np.zeros(self.non_targets.shape[0])]).tolist()
        e = eer(scores, labels)
        self.assertTrue(0.15 < e < 0.165)

    def test_label_types(self):
        N = 1000
        scores = np.concatenate([self.targets[:N], self.non_targets[:N]])
        labels = np.concatenate([np.ones(N), np.zeros(N)])
        for dtype in bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32:
            e = eer(scores, labels.astype(dtype))
            self.assertTrue(0.10 < e < 0.2)

    def test_largish_tnt(self):
        self.assertTrue(0.15 < eer_tnt(self.targets.tolist(), self.non_targets.tolist()) < 0.165)

    def test_small(self):
        for scores, labels, result in [
            ([0, 1], [0, 1], 0),
            ([0, 1], [1, 0], 0.5),
            ([0, 0, 1], [0, 1, 1], 1/3),
            ([0, 1, 1], [0, 0, 1], 1/3)
        ]:
            self.assertAlmostEqual(eer(scores, labels), result)

    def test_ndim(self):
        scores = np.expand_dims(np.concatenate([self.targets, self.non_targets]), 0)
        labels = np.concatenate([np.ones(self.targets.shape[0]), np.zeros(self.non_targets.shape[0])])
        self.assertRaises(ValueError, eer, scores, labels)

    def test_shape(self):
        scores = np.concatenate([self.targets, self.non_targets])[1:]
        labels = np.concatenate([np.ones(self.targets.shape[0]), np.zeros(self.non_targets.shape[0])])
        self.assertRaises(ValueError, eer, scores, labels)


