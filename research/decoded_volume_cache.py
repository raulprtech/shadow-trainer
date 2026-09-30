"""Session-local, source-bound float32 NIfTI decode cache.

The cache is reconstructible but never trusted without a receipt and byte hash.
No historical /tmp cache is opened, migrated or deleted.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import psutil


GIB, MIB = 2**30, 2**20
MAX_DECODE_BYTES = 1536 * MIB


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def validate_resources(expected_bytes=0):
    if shutil.disk_usage('/mnt/c').free < 20 * GIB + expected_bytes + 256 * MIB:
        raise RuntimeError('decoded_cache_disk_reservation')
    if psutil.virtual_memory().available < 1536 * MIB:
        raise RuntimeError('decoded_cache_ram_floor')
    if psutil.swap_memory().used > 192 * MIB:
        raise RuntimeError('decoded_cache_swap_limit')


def materialize(image_nii, source_sha256, root):
    shape = tuple(int(x) for x in image_nii.shape)
    if len(shape) != 3 or any(x < 1 for x in shape):
        raise ValueError('invalid image shape')
    expected_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
    if expected_bytes > MAX_DECODE_BYTES:
        raise RuntimeError('decoded_cache_case_limit')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'{source_sha256}.f32'
    receipt = root / f'{source_sha256}.receipt.json'
    identity = {'schema': 'shadowtrainer.decoded-volume/v2', 'source_sha256': source_sha256,
                'shape': list(shape), 'dtype': 'float32', 'order': 'F',
                'size_bytes': expected_bytes}
    if target.exists() or receipt.exists():
        if not target.is_file() or not receipt.is_file() or target.stat().st_size != expected_bytes:
            raise ValueError('incomplete or incorrectly sized decoded cache')
        row = json.loads(receipt.read_text())
        if row.get('identity') != identity or row.get('decoded_sha256') != digest(target):
            raise ValueError('decoded cache receipt or content mismatch')
        return np.memmap(target, mode='r', dtype=np.float32, shape=shape, order='F')
    validate_resources(expected_bytes)
    fd, partial_name = tempfile.mkstemp(dir=root, prefix='.decode-', suffix='.partial')
    os.close(fd)
    partial = Path(partial_name)
    try:
        # NIfTI stores the first dimension contiguously. Reading slices along
        # the last dimension lets gzip advance through the stream instead of
        # repeatedly seeking through the entire compressed volume.
        mapped = np.memmap(partial, mode='w+', dtype=np.float32, shape=shape, order='F')
        for start in range(0, shape[2], 8):
            validate_resources()
            mapped[:, :, start:start + 8] = np.asarray(
                image_nii.dataobj[:, :, start:start + 8], dtype=np.float32)
        mapped.flush()
        del mapped
        with partial.open('rb') as stream:
            os.fsync(stream.fileno())
        decoded_sha = digest(partial)
        os.link(partial, target)  # exclusive publication; never replace another cache
        record = {'identity': identity, 'decoded_sha256': decoded_sha}
        fd2, receipt_name = tempfile.mkstemp(dir=root, prefix='.receipt-', suffix='.partial')
        try:
            with os.fdopen(fd2, 'w') as stream:
                json.dump(record, stream, sort_keys=True)
                stream.flush(); os.fsync(stream.fileno())
            os.link(receipt_name, receipt)
        finally:
            Path(receipt_name).unlink(missing_ok=True)
    finally:
        partial.unlink(missing_ok=True)
    return np.memmap(target, mode='r', dtype=np.float32, shape=shape, order='F')


def evict(root, source_sha256):
    """Discard only this session's verified, reconstructible decoded image."""
    root = Path(root)
    target = root / f'{source_sha256}.f32'
    receipt = root / f'{source_sha256}.receipt.json'
    if not target.exists() and not receipt.exists():
        return False
    if not target.is_file() or not receipt.is_file():
        raise ValueError('incomplete decoded cache cannot be evicted')
    record = json.loads(receipt.read_text())
    identity = record.get('identity', {})
    if (identity.get('schema') != 'shadowtrainer.decoded-volume/v2'
            or identity.get('source_sha256') != source_sha256
            or identity.get('size_bytes') != target.stat().st_size
            or record.get('decoded_sha256') != digest(target)):
        raise ValueError('decoded cache mismatch before eviction')
    receipt.unlink()
    target.unlink()
    return True
