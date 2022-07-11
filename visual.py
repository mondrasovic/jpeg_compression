#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Milan Ondrašovič <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

import numpy as np
from matplotlib import pyplot as plt

from utils import select_zigzag


def plot_compression_effect(images_dct_blocks, labels, reduction='mean'):
    iter_types = (list, tuple)

    if not isinstance(images_dct_blocks, iter_types):
        images_dct_blocks = [images_dct_blocks]

    if not isinstance(labels, iter_types):
        labels = [labels]

    assert len(images_dct_blocks) == len(labels)

    reduction_fn = getattr(np, reduction)
    n_items = len(images_dct_blocks)

    fig, axes = plt.subplots(nrows=n_items, sharex='all', figsize=(12, 5))

    if n_items == 1:
        axes = [axes]

    for i, (dct_blocks, label) in enumerate(zip(images_dct_blocks, labels)):
        ax = axes[i]
        dct_stats = reduction_fn(dct_blocks, axis=0)
        dct_stats_zigzag = select_zigzag(dct_stats)

        xs = np.arange(len(dct_stats_zigzag))
        ax.bar(xs, dct_stats_zigzag, label=label)
        ax.legend()

    fig.suptitle("Compression Effect Visualization Using DCT Coefficients")
    fig.supxlabel("DCT cell position (zig-zag pattern)")
    fig.supylabel(f"Value after '{reduction}' reduction")

    fig.tight_layout()

    return fig


def plot_linear_feature_importance(model):
    importance = model.coef_[0]
    fig, ax = plt.subplots(figsize=(16, 4))
    xs = np.arange(len(importance))
    ax.bar(xs, importance)

    fig.suptitle("Feature Importance in Linear Model")
    ax.set_xlabel("Feature position")
    ax.set_ylabel("Coefficient magnitude")

    return fig
