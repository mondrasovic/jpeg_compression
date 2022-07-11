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

from utils import assure_size_divisibility, block_split

from scipy import fftpack


def dct_2d(arr, norm='ortho'):
    return fftpack.dct(fftpack.dct(arr, axis=0, norm=norm), axis=1, norm=norm)


def create_dct_image(image_arr, block_size=8):
    assert image_arr.ndim == 2, "DCT image supports only two-dimensional arrays"

    height, width = image_arr.shape
    dct_image = np.zeros_like(image_arr)

    for i in range(0, height, block_size):
        ii = i + block_size

        for j in range(0, width, block_size):
            jj = j + block_size

            dct_sub_image = dct_2d(image_arr[i:ii, j:jj])
            dct_image[i:ii, j:jj] = dct_sub_image

    return dct_image


def extract_dct_blocks(image, block_size=8):
    image_arr = np.asarray(image)
    image_padded = assure_size_divisibility(image_arr, block_size, block_size)
    dct_image = create_dct_image(image_padded)
    dct_blocks = block_split(dct_image, block_size, block_size)

    return dct_blocks
