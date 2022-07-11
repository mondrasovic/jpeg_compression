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


def assure_size_divisibility(image_arr, height_div, width_div, mode='edge'):
    assert image_arr.ndim == 2, "image has to be a two-dimensional array"

    height, width = image_arr.shape

    def _calc_rem(dim, div):
        return (int(np.ceil(dim / div)) * div) - dim

    n_rem_rows = _calc_rem(height, height_div)
    n_rem_cols = _calc_rem(width, width_div)

    if max(n_rem_rows, n_rem_cols) > 0:
        image_arr = np.pad(
            image_arr, pad_width=((0, n_rem_rows), (0, n_rem_cols)), mode=mode
        )

    return image_arr


def block_split(arr, block_n_rows, block_n_cols):
    n_rows, n_cols = arr.shape

    assert (n_rows % block_n_rows) == 0, "number of rows not divisible"
    assert (n_cols % block_n_cols) == 0, "number of columns not divisible"

    return (
        arr.reshape(n_rows // block_n_rows, block_n_rows, -1,
                    block_n_cols).swapaxes(1, 2).reshape(
                        -1, block_n_rows, block_n_cols
                    )
    )


def select_zigzag(arr):
    n_rows = arr.shape[0]

    return np.concatenate(
        [
            np.diagonal(arr[::-1, :], i)[::(2 * (i % 2) - 1)]
            for i in range(1 - n_rows, n_rows)
        ]
    )
