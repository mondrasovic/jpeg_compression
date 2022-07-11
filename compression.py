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

import io
import itertools

import numpy as np
from PIL import Image


class QualityLevelsGen:
    def __init__(self, n_levels=None, default_levels=(1, 25, 50, 75, 100)):
        self._n_levels = min(
            len(default_levels) if n_levels is None else n_levels, 100
        )
        self._default_levels = default_levels

    def __call__(self):
        rnd_levels = np.random.permutation(np.arange(101))
        values = itertools.chain(iter(self._default_levels), iter(rnd_levels))

        generated = set()

        while len(generated) < self._n_levels:
            try:
                curr_level = next(values)
                if curr_level not in generated:
                    generated.add(curr_level)

                    yield curr_level
            except StopIteration:
                break


def _gen_quality_factors(self):
    jitter = np.random.randint(
        -self._jitter_magnitude, self._jitter_magnitude + 1,
        len(self._quality_factors)
    )
    rnd_quality_factors = self._quality_factors + jitter
    rnd_quality_factors = np.clip(rnd_quality_factors, 1, 100)

    return map(int, rnd_quality_factors)


def compress_image(image, quality=75):
    buffer = io.BytesIO()

    image.save(buffer, format='jpeg', quality=quality)
    buffer.seek(0)
    image_compressed = Image.open(buffer)

    return image_compressed
