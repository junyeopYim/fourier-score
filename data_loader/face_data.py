"""Lazy, CPU-only TFRecord readers adapted from mnist_compare/face_data.py.

Keeps source CelebA float32 crop/resize and original FFHQ record decoding.
TensorFlow is an optional dependency, not imported for MNIST/CIFAR training.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np
import torch

PREPROCESS_VERSION = 'upstream_tf_float32_celeba_crop140_bilinear_antialias_v1'


def tensorflow_cpu():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError('Install requirements-faces.txt in a Python 3.11 environment') from e
    tf.config.set_visible_devices([], 'GPU')
    return tf


def record_index(files):
    """Validate TFRecord framing/length, not CRC; uncompressed files only."""
    rows = []
    for file_id, path in enumerate(files):
        size = path.stat().st_size
        with path.open('rb') as stream:
            while stream.tell() < size:
                header = stream.read(12)
                if len(header) != 12:
                    raise ValueError(f'{path}: incomplete TFRecord header')
                length = struct.unpack('<Q', header[:8])[0]
                offset = stream.tell()
                if length == 0 or offset + length + 4 > size:
                    raise ValueError(f'{path}: incomplete TFRecord payload')
                rows.append((file_id, offset, length))
                stream.seek(length + 4, 1)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


class FaceDataset:
    def __init__(self, cfg, split='train'):
        self.cfg, self.split = cfg, split
        self.tf = tensorflow_cpu()
        self.kind, self.size = cfg.data.dataset.lower(), cfg.data.image_size
        self.handles = {}
        if (self.kind, self.size) not in {('celeba', 64), ('ffhq', 256)}:
            raise ValueError('Faces require CELEBA 64 or FFHQ 256')
        if self.kind == 'celeba':
            try:
                import tensorflow_datasets as tfds
            except ImportError as e:
                raise ImportError('Install requirements-faces.txt') from e
            builder = tfds.builder('celeb_a', data_dir=cfg.data.tfds_dir)
            if not builder.info.splits:
                raise FileNotFoundError('Prepare CelebA with TFDS download_and_prepare first')
            instructions = builder.info.splits[split].file_instructions
            if any(item.skip != 0 for item in instructions):
                raise ValueError('Expected whole TFDS split shards')
            self.files = [Path(builder.data_dir) / item.filename for item in instructions]
            expected = builder.info.splits[split].num_examples
        else:
            if split != 'train':
                raise ValueError('The FFHQ preset uses all 70000 records; no held-out split')
            self.files = [Path(cfg.data.tfrecords_path)]
            expected = 70000
        metadata = dict(files=[dict(path=str(p.resolve()), size=p.stat().st_size,
                                    mtime_ns=p.stat().st_mtime_ns) for p in self.files],
                        dataset=self.kind, image_size=self.size, centered=cfg.data.centered,
                        split=split, preprocessing=PREPROCESS_VERSION)
        self.fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        cache = Path(cfg.data_loader.cache_dir) / f'{self.kind}{self.size}_{split}_{self.fingerprint[:16]}.index.npy'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            self.index = np.load(cache, allow_pickle=False)
        else:
            self.index = record_index(self.files)
            temp = cache.with_suffix('.tmp.npy')
            np.save(temp, self.index, allow_pickle=False)
            temp.replace(cache)
        if len(self.index) != expected:
            raise ValueError(f'Expected {expected} records, found {len(self.index)}; check the prepared dataset')
        self.metadata = dict(metadata, n_images=len(self.index), fingerprint=self.fingerprint)

    def __len__(self):
        return len(self.index)

    def close(self):
        for stream in self.handles.values():
            stream.close()
        self.handles.clear()

    def batch(self, ids):
        tf = self.tf
        records = []
        for row in self.index[np.asarray(ids, dtype=np.int64)]:
            file_id, offset, length = map(int, row)
            if file_id not in self.handles:
                self.handles[file_id] = self.files[file_id].open('rb')
            stream = self.handles[file_id]
            stream.seek(offset)
            record = stream.read(length)
            if len(record) != length:
                raise ValueError('Source record changed after indexing')
            records.append(record)
        if not records:
            raise ValueError('Cannot read an empty batch')
        with tf.device('/CPU:0'):
            if self.kind == 'ffhq':
                parsed = tf.io.parse_example(records, {
                    'shape': tf.io.FixedLenFeature([3], tf.int64),
                    'data': tf.io.FixedLenFeature([], tf.string)})
                tf.debugging.assert_equal(parsed['shape'], tf.constant([3, self.size, self.size], tf.int64))
                image = tf.io.decode_raw(parsed['data'], tf.uint8)
                image = tf.reshape(image, [-1, 3, self.size, self.size])
                image = tf.transpose(image, [0, 2, 3, 1])
                image = tf.image.convert_image_dtype(image, tf.float32)
            else:
                parsed = tf.io.parse_example(records, {'image': tf.io.FixedLenFeature([], tf.string)})
                def decode(encoded):
                    image = tf.io.decode_jpeg(encoded, channels=3)
                    image = tf.ensure_shape(image, [218, 178, 3])
                    image = tf.image.convert_image_dtype(image, tf.float32)
                    image = tf.image.crop_to_bounding_box(image, 39, 19, 140, 140)
                    return tf.image.resize(image, [64, 64], antialias=True)
                image = tf.map_fn(decode, parsed['image'],
                                 fn_output_signature=tf.TensorSpec([64, 64, 3], tf.float32))
            if self.cfg.data.centered:
                image = image * 2. - 1.
            return torch.from_numpy(tf.transpose(image, [0, 3, 1, 2]).numpy()).contiguous()
