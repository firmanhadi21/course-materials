"""
BasicSR-Compatible Dataset for Sentinel-2 / PlanetScope Super-Resolution
=========================================================================

Drop-in dataset for the BasicSR training framework. Registers automatically
via ``DATASET_REGISTRY`` so you can reference it from YAML configs as::

    datasets:
      train:
        type: S2PSBasicSRDataset
        lr_dir: data/patches/train/sentinel2
        hr_dir: data/patches/train/planetscope
        ...

Supports both .npy and .tif patch files produced by preprocess.py.
"""

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.utils.registry import DATASET_REGISTRY

# ---------------------------------------------------------------------------
# Try to import the custom normalizer / augmentation from our package.
# If not on sys.path, fall back to inline implementations.
# ---------------------------------------------------------------------------
try:
    from dataset import RadiometricNormalizer, SRDataAugmentation
    _HAS_LOCAL = True
except ImportError:
    _HAS_LOCAL = False


# ====================================================================== #
#  Inline fallbacks (used only when dataset.py is not importable)         #
# ====================================================================== #

def _percentile_normalize(img: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Per-band percentile clipping to [0, 1]."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        band = img[c]
        vmin = np.percentile(band[band > 0], lo) if (band > 0).any() else 0
        vmax = np.percentile(band[band > 0], hi) if (band > 0).any() else 1
        if vmax - vmin < 1e-6:
            vmax = vmin + 1
        out[c] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return out


def _augment_pair(lr: np.ndarray, hr: np.ndarray):
    """Random flip + rot90, applied identically to LR/HR."""
    if random.random() > 0.5:
        lr = lr[:, :, ::-1].copy()
        hr = hr[:, :, ::-1].copy()
    if random.random() > 0.5:
        lr = lr[:, ::-1, :].copy()
        hr = hr[:, ::-1, :].copy()
    k = random.randint(0, 3)
    if k:
        lr = np.rot90(lr, k, axes=(1, 2)).copy()
        hr = np.rot90(hr, k, axes=(1, 2)).copy()
    return lr, hr


# ====================================================================== #
#  Dataset                                                                #
# ====================================================================== #

@DATASET_REGISTRY.register()
class S2PSBasicSRDataset(Dataset):
    """
    BasicSR dataset for Sentinel-2 (LR) / PlanetScope (HR) patch pairs.

    YAML options::

        type: S2PSBasicSRDataset
        lr_dir: path/to/sentinel2_patches      # required
        hr_dir: path/to/planetscope_patches     # required
        file_format: npy                        # npy | tif
        scale: 3                                # SR scale factor
        normalize: true                         # percentile norm
        norm_method: percentile                 # percentile | minmax | zscore
        augment: true                           # flip + rot90
        spectral_jitter: 0.0                    # random gain per band (0 = off)
        gt_size: 192                            # crop HR to this size (0 = no crop)
        mean: ~                                 # per-band mean for zscore (list)
        std: ~                                  # per-band std  for zscore (list)
    """

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt

        self.lr_dir = Path(opt['lr_dir'])
        self.hr_dir = Path(opt['hr_dir'])
        self.scale = int(opt.get('scale', 3))
        self.file_format = opt.get('file_format', 'npy')
        self.do_normalize = opt.get('normalize', True)
        self.norm_method = opt.get('norm_method', 'percentile')
        self.do_augment = opt.get('augment', True)
        self.spectral_jitter = float(opt.get('spectral_jitter', 0.0))
        self.gt_size = int(opt.get('gt_size', 0))  # 0 = use full patch
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        # --- Build normalizer / augmentor from our package if available ---
        if _HAS_LOCAL and self.do_normalize:
            self.normalizer = RadiometricNormalizer(method=self.norm_method)
        else:
            self.normalizer = None

        if _HAS_LOCAL and self.do_augment:
            self.augmentor = SRDataAugmentation(
                spectral_jitter=self.spectral_jitter
            )
        else:
            self.augmentor = None

        # --- Pair files by stem name ---
        ext = f'.{self.file_format}'
        lr_map = {f.stem: f for f in sorted(self.lr_dir.glob(f'*{ext}'))}
        hr_map = {f.stem: f for f in sorted(self.hr_dir.glob(f'*{ext}'))}
        common = sorted(set(lr_map) & set(hr_map))

        if not common:
            raise FileNotFoundError(
                f'No matching LR/HR pairs between {self.lr_dir} and '
                f'{self.hr_dir} with extension {ext}.'
            )

        self.pairs = [(str(lr_map[n]), str(hr_map[n])) for n in common]
        print(f'[S2PSBasicSRDataset] {len(self.pairs)} pairs  |  '
              f'scale={self.scale}  format={self.file_format}  '
              f'norm={self.norm_method if self.do_normalize else "off"}  '
              f'aug={self.do_augment}')

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def _load(self, path: str) -> np.ndarray:
        """Load patch → (C, H, W) float32."""
        if self.file_format == 'npy':
            arr = np.load(path).astype(np.float32)
        elif self.file_format == 'tif':
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)
        else:
            raise ValueError(f'Unknown format: {self.file_format}')

        # Ensure (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return arr

    # ------------------------------------------------------------------ #
    #  Normalization                                                       #
    # ------------------------------------------------------------------ #

    def _normalize(self, img: np.ndarray, sensor: str) -> np.ndarray:
        if not self.do_normalize:
            return img
        if self.normalizer is not None:
            return self.normalizer.normalize(img, sensor=sensor)
        # Inline fallback
        if self.norm_method == 'zscore' and self.mean and self.std:
            mean = np.array(self.mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.std, dtype=np.float32).reshape(-1, 1, 1)
            return (img - mean) / (std + 1e-8)
        return _percentile_normalize(img)

    # ------------------------------------------------------------------ #
    #  Random crop (optional)                                              #
    # ------------------------------------------------------------------ #

    def _random_crop(
        self, lr: np.ndarray, hr: np.ndarray
    ) -> tuple:
        if self.gt_size <= 0:
            return lr, hr
        gt_h, gt_w = self.gt_size, self.gt_size
        lq_h, lq_w = gt_h // self.scale, gt_w // self.scale

        _, h_lr, w_lr = lr.shape
        _, h_hr, w_hr = hr.shape

        if h_lr < lq_h or w_lr < lq_w:
            raise ValueError(
                f'LR patch ({h_lr}×{w_lr}) smaller than '
                f'crop ({lq_h}×{lq_w})'
            )

        top_lr = random.randint(0, h_lr - lq_h)
        left_lr = random.randint(0, w_lr - lq_w)
        top_hr = top_lr * self.scale
        left_hr = left_lr * self.scale

        lr = lr[:, top_lr:top_lr + lq_h, left_lr:left_lr + lq_w]
        hr = hr[:, top_hr:top_hr + gt_h, left_hr:left_hr + gt_w]
        return lr, hr

    # ------------------------------------------------------------------ #
    #  __getitem__  — returns BasicSR-style dict                           #
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> Dict[str, object]:
        lr_path, hr_path = self.pairs[idx]

        lr = self._load(lr_path)
        hr = self._load(hr_path)

        # Normalize
        lr = self._normalize(lr, sensor='sentinel2')
        hr = self._normalize(hr, sensor='planetscope')

        # Random crop
        lr, hr = self._random_crop(lr, hr)

        # Augmentation
        if self.do_augment:
            if self.augmentor is not None:
                lr, hr = self.augmentor(lr, hr)
            else:
                lr, hr = _augment_pair(lr, hr)

        # To tensor
        lq = torch.from_numpy(np.ascontiguousarray(lr)).float()
        gt = torch.from_numpy(np.ascontiguousarray(hr)).float()

        return {
            'lq': lq,
            'gt': gt,
            'lq_path': lr_path,
            'gt_path': hr_path,
        }

    def __len__(self) -> int:
        return len(self.pairs)


# ====================================================================== #
#  Paired-image dataset variant (for BasicSR PairedImageDataset parity)   #
# ====================================================================== #

@DATASET_REGISTRY.register()
class S2PSLMDBDataset(Dataset):
    """
    LMDB-backed variant for very large patch collections (>100k patches).

    Patches are stored in an LMDB database for fast random access.
    Use ``scripts/create_lmdb.py`` (from BasicSR) or the helper below
    to convert .npy patches into LMDB.

    YAML options::

        type: S2PSLMDBDataset
        dataroot_lq: data/patches/train_sentinel2.lmdb
        dataroot_gt: data/patches/train_planetscope.lmdb
        scale: 3
        gt_size: 192
        augment: true
        normalize: true
    """

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.scale = int(opt.get('scale', 3))
        self.gt_size = int(opt.get('gt_size', 0))
        self.do_augment = opt.get('augment', True)
        self.do_normalize = opt.get('normalize', True)

        import lmdb
        self.lq_env = lmdb.open(
            opt['dataroot_lq'], readonly=True, lock=False,
            readahead=False, meminit=False,
        )
        self.gt_env = lmdb.open(
            opt['dataroot_gt'], readonly=True, lock=False,
            readahead=False, meminit=False,
        )

        with self.lq_env.begin(write=False) as txn:
            self.keys = sorted(
                k.decode() for k in txn.cursor().iternext(values=False)
                if not k.startswith(b'__')
            )
        print(f'[S2PSLMDBDataset] {len(self.keys)} patches from LMDB')

    def _read_lmdb(self, env, key: str, shape: tuple) -> np.ndarray:
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode())
        return np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()

    def __getitem__(self, idx: int) -> Dict[str, object]:
        key = self.keys[idx]

        # The LMDB stores metadata under __shape_<key> entries
        import lmdb
        with self.lq_env.begin(write=False) as txn:
            lq_meta = txn.get(f'__shape_{key}'.encode())
            lq_shape = tuple(map(int, lq_meta.decode().split(',')))
        with self.gt_env.begin(write=False) as txn:
            gt_meta = txn.get(f'__shape_{key}'.encode())
            gt_shape = tuple(map(int, gt_meta.decode().split(',')))

        lr = self._read_lmdb(self.lq_env, key, lq_shape)
        hr = self._read_lmdb(self.gt_env, key, gt_shape)

        if self.do_normalize:
            lr = _percentile_normalize(lr)
            hr = _percentile_normalize(hr)

        if self.gt_size > 0:
            # random crop
            gt_h = gt_w = self.gt_size
            lq_h, lq_w = gt_h // self.scale, gt_w // self.scale
            top = random.randint(0, lr.shape[1] - lq_h)
            left = random.randint(0, lr.shape[2] - lq_w)
            lr = lr[:, top:top + lq_h, left:left + lq_w]
            hr = hr[:, top * self.scale:(top + lq_h) * self.scale,
                     left * self.scale:(left + lq_w) * self.scale]

        if self.do_augment:
            lr, hr = _augment_pair(lr, hr)

        return {
            'lq': torch.from_numpy(np.ascontiguousarray(lr)).float(),
            'gt': torch.from_numpy(np.ascontiguousarray(hr)).float(),
            'lq_path': key,
            'gt_path': key,
        }

    def __len__(self):
        return len(self.keys)
