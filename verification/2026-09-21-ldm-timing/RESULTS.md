# Native LDM 100-update timing

Completed measurements: 16/16. RTX 5060 Ti 16GB; FP32, TF32 off, microbatch 4. Full paper effective batches.

Steady timing excludes the first 10 of the 100 actual optimizer updates. ETA covers optimizer work on cached latents; preparation, evaluation and checkpoint writes are excluded. Real repeated image subsets measure throughput, not convergence.

| Dataset | Arm | Batch | 100 updates (s) | Steady s/update | Paper updates | ETA (days/seed) | Peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| ffhq | epsilon | 42 | 384.45 | 3.8214 | 635,000 | 28.09 | 7.11 |
| ffhq | scalar_gaussian | 42 | 397.72 | 3.9869 | 635,000 | 29.30 | 7.10 |
| ffhq | fourier_gaussian_unscaled | 42 | 400.30 | 3.9843 | 635,000 | 29.28 | 7.10 |
| ffhq | fourier_gaussian | 42 | 388.58 | 3.8973 | 635,000 | 28.64 | 7.10 |
| celebahq | epsilon | 48 | 442.77 | 4.4266 | 410,000 | 21.01 | 7.11 |
| celebahq | scalar_gaussian | 48 | 428.69 | 4.2897 | 410,000 | 20.36 | 7.11 |
| celebahq | fourier_gaussian_unscaled | 48 | 441.68 | 4.4081 | 410,000 | 20.92 | 7.11 |
| celebahq | fourier_gaussian | 48 | 447.35 | 4.4978 | 410,000 | 21.34 | 7.11 |
| lsun_churches | epsilon | 96 | 357.82 | 3.5721 | 500,000 | 20.67 | 6.60 |
| lsun_churches | scalar_gaussian | 96 | 358.17 | 3.5839 | 500,000 | 20.74 | 6.60 |
| lsun_churches | fourier_gaussian_unscaled | 96 | 359.26 | 3.5921 | 500,000 | 20.79 | 6.60 |
| lsun_churches | fourier_gaussian | 96 | 383.07 | 3.8526 | 500,000 | 22.29 | 6.60 |
| lsun_bedrooms | epsilon | 48 | 449.07 | 4.4872 | 1,900,000 | 98.68 | 7.11 |
| lsun_bedrooms | scalar_gaussian | 48 | 453.92 | 4.5437 | 1,900,000 | 99.92 | 7.11 |
| lsun_bedrooms | fourier_gaussian_unscaled | 48 | 443.56 | 4.4130 | 1,900,000 | 97.04 | 7.11 |
| lsun_bedrooms | fourier_gaussian | 48 | 442.96 | 4.4423 | 1,900,000 | 97.69 | 7.11 |

Data: FFHQ original PNGs with publisher MD5 verification; Churches original LMDB with official split membership; Bedrooms fast.ai/HF repack with official image-key/split membership; CelebA-HQ 1024 PNG mirror with a custom disjoint 96/4 timing split (original index mapping unverified). Subset source manifests and cache identities are retained under data/ldm_benchmark100/.
