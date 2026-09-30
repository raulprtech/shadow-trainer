"""Fail-closed receipts for new predictions; never certify legacy artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(checkpoint_sha256, image, label, protocol):
    return {'checkpoint_sha256': checkpoint_sha256, 'image_sha256': sha256(image),
            'label_sha256': sha256(label), 'protocol': protocol}


def receipt_path(prediction):
    return Path(str(prediction) + '.receipt.json')


def seal(prediction, expected):
    destination = receipt_path(prediction)
    if destination.exists():
        raise ValueError('receipt already exists; refuse overwrite')
    payload = {'schema': 'shadowtrainer.prediction-receipt/v1',
               'binding': expected, 'prediction_sha256': sha256(prediction)}
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.receipt-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive publication: another evaluator must not overwrite a receipt.
        os.link(name, destination)
    finally:
        os.unlink(name)


def verify(prediction, expected):
    receipt = receipt_path(prediction)
    if not receipt.is_file():
        raise ValueError('legacy/unsealed prediction: no provenance receipt; refuse reuse')
    data = json.loads(receipt.read_text())
    if data.get('schema') != 'shadowtrainer.prediction-receipt/v1' or data.get('binding') != expected:
        raise ValueError('prediction provenance mismatch')
    if data.get('prediction_sha256') != sha256(prediction):
        raise ValueError('prediction digest mismatch')
    return data
