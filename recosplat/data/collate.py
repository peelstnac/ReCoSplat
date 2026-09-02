"""Batch assembly: collate a list of homogeneous unbatched examples into one batch.

The dataset collates its own batches (the DataLoader runs with `batch_size=None`), so
this only needs to handle the example structure our pipeline produces: nested dicts of
same-shaped tensors, plus strings. Homogeneity within a batch is guaranteed because all
per-step parameters (V, chunk sizes, resolution) are fixed before the batch is filled.
"""

import torch
from torch import Tensor


def collate_examples(examples: list[dict]) -> dict:
    assert len(examples) > 0
    first = examples[0]
    batch: dict = {}
    for key, value in first.items():
        if isinstance(value, Tensor):
            batch[key] = torch.stack([example[key] for example in examples])
        elif isinstance(value, dict):
            batch[key] = collate_examples([example[key] for example in examples])
        elif isinstance(value, str):
            batch[key] = [example[key] for example in examples]
        else:
            raise TypeError(f"Cannot collate key {key!r} of type {type(value)}")
    return batch
