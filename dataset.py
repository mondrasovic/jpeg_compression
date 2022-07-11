import bisect
import itertools
import multiprocessing
import pathlib
import shutil

import numpy as np
import tqdm
from PIL import Image

from compression import compress_image
from dct import extract_dct_blocks
from utils import select_zigzag


def read_data_sample(image_file, label_getter, block_size, color_mode):
    features = _load_image_and_extract_features(
        image_file, block_size, color_mode
    )

    is_uncompressed = 'uncompressed' in image_file.stem
    quality_factor = None if is_uncompressed else int(image_file.stem[-3:])
    label = label_getter(is_uncompressed, quality_factor)

    return features, label


def load_dataset_and_extract_features(
    dataset_dir_path, label_getter, block_size=8, color_mode='L'
):
    features = []
    labels = []

    with multiprocessing.Pool() as pool:
        params_iter = (
            (image_file, label_getter, block_size, color_mode)
            for image_file in pathlib.Path(dataset_dir_path).iterdir()
            if image_file.is_file()
        )
        results = [
            pool.apply_async(read_data_sample, params) for params in params_iter
        ]

        for result in tqdm.tqdm(results):
            curr_features, curr_label = result.get()

            features.append(curr_features)
            labels.append(curr_label)

    features = np.asarray(features)
    labels = np.asarray(labels)

    return features, labels


class MultiClassLabelGetter:
    def __init__(self, label_max_bounds):
        self.label_max_bounds = label_max_bounds

    def __call__(self, is_uncompressed, quality_factor):
        if is_uncompressed:
            return 0

        return bisect.bisect_left(self.label_max_bounds, quality_factor) + 1


def regression_label_getter(is_uncompressed, quality_factor):
    return quality_factor


def bin_class_labeL_getter(is_uncompressed, quality_factor):
    return 0 if is_uncompressed else 1


def create_image_compression_dataset(
    images_dir_path,
    train_dir_path,
    test_dir_path,
    quality_factors_gen,
    train_frac=0.8,
    save_uncompressed=True
):
    image_files = list(pathlib.Path(images_dir_path).iterdir())
    n_images = len(image_files)
    n_train_images = np.clip(int(round(n_images * train_frac)), 0, n_images)

    image_idx_counter = itertools.count()

    train_image_files = image_files[:n_train_images]
    test_image_files = image_files[n_train_images:]

    _create_dataset_subset(
        image_idx_counter, train_image_files, train_dir_path,
        quality_factors_gen, 'train', save_uncompressed
    )
    _create_dataset_subset(
        image_idx_counter, test_image_files, test_dir_path, quality_factors_gen,
        'test', save_uncompressed
    )


def build_quality_factors_gen(
    quality_factors, min_deviation=3, max_deviation=3
):
    quality_factors = np.asarray(quality_factors)

    def _gen():
        deviation = np.random.randint(
            -min_deviation, max_deviation + 1, len(quality_factors)
        )
        rnd_quality_factors = quality_factors + deviation
        rnd_quality_factors = np.clip(rnd_quality_factors, 1, 100)

        return map(int, rnd_quality_factors)

    return _gen


def _load_image_and_extract_features(image_file_path, block_size, color_mode):
    image_orig = Image.open(image_file_path)
    image_orig = image_orig.convert(color_mode)
    image_arr = np.atleast_3d(np.asarray(image_orig)).transpose(2, 0, 1)

    reduction_fns = (np.mean, np.std, np.median)

    def _process_channel(channel):
        dct_blocks = extract_dct_blocks(channel, block_size)

        for reduction_fn in reduction_fns:
            yield select_zigzag(reduction_fn(dct_blocks, axis=0))

    features = np.concatenate(
        list(itertools.chain.from_iterable(map(_process_channel, image_arr)))
    )

    return features


def _create_dataset_subset(
    image_idx_counter, image_files, output_dir_path, quality_factors_gen, desc,
    save_uncompressed
):
    output_dir = pathlib.Path(output_dir_path)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_image_file in tqdm.tqdm(image_files, desc=desc):
        image_idx = next(image_idx_counter)

        def _save(image, suffix):
            output_file_name = f'image_{image_idx:04d}_{suffix}'
            output_file = output_dir / output_file_name
            image.save(str(output_file))

        image_orig = Image.open(input_image_file).convert('RGB')
        if save_uncompressed:
            _save(image_orig, f'uncompressed{input_image_file.suffix}')

        for quality in quality_factors_gen():
            image_compressed = compress_image(image_orig, quality)
            _save(image_compressed, f'quality_{quality:03d}.jpg')
