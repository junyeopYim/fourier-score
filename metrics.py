"""Optional offline FID/IS from NPZ sample shards, using source TF-Hub backends.

Not imported during training. Requires requirements-metrics.txt, Python 3.11,
network access for the first model download, and matching real feature statistics.
This optional integration has NOT been exercised in the bundled CPU-only tests.
"""
import argparse
import json
from pathlib import Path
import numpy as np

TFGAN_HUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
V3_HUB = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'


def engine(inception_v3=False):
    from data_loader.face_data import tensorflow_cpu
    tf = tensorflow_cpu()
    import tensorflow_gan as tfgan
    import tensorflow_hub as hub
    url = V3_HUB if inception_v3 else TFGAN_HUB
    with tf.device('/CPU:0'):
        network = hub.load(url)

    @tf.function
    def extract(images):
        images = tf.cast(images, tf.float32)
        images = images / 255. if inception_v3 else (images - 127.5) / 127.5
        def classifier(x):
            result = network(x)
            return tf.nest.map_structure(tf.compat.v1.layers.flatten, result)
        if inception_v3:
            pool = tfgan.eval.run_classifier_fn(images, num_batches=1,
                                                classifier_fn=classifier, dtypes=tf.float32)
            return {'pool_3': pool}
        return tfgan.eval.run_classifier_fn(images, num_batches=1, classifier_fn=classifier,
                                            dtypes={'pool_3': tf.float32, 'logits': tf.float32})
    return tf, tfgan, extract, url


def validate_images(images):
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Expected RGB uint8 NHWC images. MNIST uses a different metric; do not call this FID/IS.')


def main():
    from utils.util import json_write, file_hash
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    real = sub.add_parser('extract-real', help='Extract features from the configured real training images')
    real.add_argument('-c', '--config', required=True)
    real.add_argument('--set', action='append', default=[])
    real.add_argument('-o', '--output', required=True)
    real.add_argument('--max-images', type=int, default=None)
    score = sub.add_parser('score', help='Compare generated shards with real pool_3 features')
    score.add_argument('--samples', required=True)
    score.add_argument('--reference', required=True, help='NPZ containing real pool_3[N,2048]')
    score.add_argument('-o', '--output', required=True)
    for child in (real, score):
        child.add_argument('--inception-v3', action='store_true', help='Source FFHQ256 feature backend; no IS')
        child.add_argument('--batch-size', type=int, default=64)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('--batch-size must be positive')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    tf, tfgan, extract, url = engine(a.inception_v3)
    pools, logits = [], []
    if a.command == 'extract-real':
        import torch
        from parse_config import ConfigParser
        from data_loader.data_loaders import build_dataset, split_indices
        cfg = ConfigParser.from_file(a.config, a.set).config
        ds = build_dataset(cfg)
        try:
            ids, _ = split_indices(cfg, len(ds))
            if a.max_images is not None:
                if a.max_images < 2:
                    p.error('--max-images must be at least 2')
                ids = ids[:a.max_images]
            for part in ids.split(a.batch_size):
                x = ds.batch(part)
                if cfg.data.centered:
                    x = (x + 1.) / 2.
                images = np.clip(x.permute(0, 2, 3, 1).numpy() * 255., 0, 255).astype(np.uint8)
                validate_images(images)
                with tf.device('/CPU:0'):
                    result = extract(images)
                pools.append(result['pool_3'].numpy())
            output = Path(a.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            features = np.concatenate(pools)
            if len(features) < 2 or not np.isfinite(features).all():
                raise ValueError('Need at least two finite real features')
            with output.open('wb') as f:
                np.savez_compressed(f, pool_3=features)
            json_write(dict(backend=url, count=len(features), dataset=cfg.data.dataset,
                            protocol=cfg.data_loader.protocol, fingerprint=ds.fingerprint,
                            quantization='clip(x*255,0,255).astype(uint8)'), str(output) + '.json')
        finally:
            ds.close()
        return
    sidecar = Path(a.reference + '.json')
    if sidecar.exists() and json.loads(sidecar.read_text())['backend'] != url:
        raise ValueError('Reference feature backend differs from generated feature backend')
    with np.load(a.reference, allow_pickle=False) as data:
        reference = data['pool_3']
    if reference.ndim != 2 or reference.shape[1] != 2048 or len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError('Expected at least two finite pool_3 vectors with 2048 features')
    directory = Path(a.samples)
    settings = json.loads((directory / 'settings.json').read_text())
    if not settings.get('complete'):
        raise ValueError('Sampling manifest is incomplete')
    paths = sorted(directory.glob('samples_*.npz'))
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            images = data['samples']
        validate_images(images)
        for begin in range(0, len(images), a.batch_size):
            with tf.device('/CPU:0'):
                result = extract(images[begin:begin + a.batch_size])
            pools.append(result['pool_3'].numpy())
            if 'logits' in result:
                logits.append(result['logits'].numpy())
    generated = np.concatenate(pools) if pools else np.empty((0, 2048))
    if len(generated) != settings['num_samples'] or len(generated) < 2 or not np.isfinite(generated).all():
        raise ValueError('Generated feature count/finiteness does not match the completed sampling manifest')
    with tf.device('/CPU:0'):
        fid = float(tfgan.eval.frechet_classifier_distance_from_activations(reference, generated).numpy())
        inception_score = float(tfgan.eval.classifier_score_from_logits(np.concatenate(logits)).numpy()) if logits else None
    report = dict(fid=fid, inception_score=inception_score, backend=url, n_real=len(reference),
                  n_generated=len(generated), reference_sha256=file_hash(a.reference),
                  sample_settings=settings, tensorflow=str(tf.__version__),
                  tfgan=str(tfgan.__version__), reference_backend_verified=sidecar.exists(),
                  note='No claim of paper reproduction. Compare identical data, quantization, backend and sampling protocol.')
    json_write(report, a.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
