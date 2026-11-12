"""Resume fast-forward by index must yield the batches the old collate-and-discard path yields.

The training loop reseeds the dataloader's generator per (seed, epoch), stamps the epoch into
the collator, then either pulls and discards ``n`` batches (old) or drops their indices before
any example is read (``skip_first_batches``). Both must leave the iterator at the same batch,
with the same masks, for any worker count.
"""

import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from model_foundry.data import _worker_init_fn, skip_first_batches
from model_foundry.data_collators import CausalLMDataCollator, MaskedLMDataCollator


class _Tok:
    pad_token_id, mask_token_id, cls_token_id, sep_token_id, unk_token_id = 0, 1, 2, 3, 4
    bos_token_id = eos_token_id = None

    def __len__(self):
        return 500


class _Rows(Dataset):
    def __init__(self, n=203):
        g = torch.Generator().manual_seed(7)
        self.rows = [torch.randint(5, 500, (int(torch.randint(4, 24, (1,), generator=g)),), generator=g).tolist()
                     for _ in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"input_ids": self.rows[i]}


def _loader(collate, workers):
    return DataLoader(_Rows(), batch_size=8, collate_fn=collate, shuffle=True, generator=torch.Generator(),
                      num_workers=workers, prefetch_factor=2 if workers else None, worker_init_fn=_worker_init_fn)


def _epoch_iter(loader, epoch, skip, fast):
    """What the loop does at the top of a (resumed) epoch."""
    loader.generator.manual_seed(1234 + epoch)
    if hasattr(loader.collate_fn, "set_epoch"):
        loader.collate_fn.set_epoch(epoch)
    if fast and skip:
        assert skip_first_batches(loader, skip)
    it = iter(loader)
    if not fast:
        for _ in range(skip):
            next(it)
    return it


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# Worker processes are spawned on macOS and re-import the package (~30 s each); run those cases on Linux.
WORKERS = [0] if sys.platform == "darwin" else [0, 1, 3]


@pytest.mark.parametrize("workers", WORKERS)
@pytest.mark.parametrize("skip", [1, 5, 17])
@pytest.mark.parametrize("make", [lambda: MaskedLMDataCollator(tokenizer=_Tok(), base_seed=99),
                                  lambda: CausalLMDataCollator(tokenizer=_Tok())])
def test_index_skip_matches_discarding(workers, skip, make):
    old = list(_epoch_iter(_loader(make(), workers), epoch=2, skip=skip, fast=False))
    new = list(_epoch_iter(_loader(make(), workers), epoch=2, skip=skip, fast=True))
    assert len(old) == len(new) == 26 - skip
    assert all(_same(a, b) for a, b in zip(old, new))


def test_later_epochs_are_not_skipped():
    w = WORKERS[-1]
    loader = _loader(MaskedLMDataCollator(tokenizer=_Tok(), base_seed=99), w)
    list(_epoch_iter(loader, epoch=0, skip=4, fast=True))
    after = list(_epoch_iter(loader, epoch=1, skip=0, fast=True))
    fresh = list(_epoch_iter(_loader(MaskedLMDataCollator(tokenizer=_Tok(), base_seed=99), w), 1, 0, False))
    assert len(after) == len(fresh) == 26 and all(_same(a, b) for a, b in zip(after, fresh))


def test_skipping_the_whole_epoch_leaves_nothing():
    loader = _loader(CausalLMDataCollator(tokenizer=_Tok()), 0)
    assert list(_epoch_iter(loader, epoch=0, skip=26, fast=True)) == []
    assert len(loader) == 26


def test_refused_when_masks_come_from_the_ambient_rng():
    loader = _loader(MaskedLMDataCollator(tokenizer=_Tok(), base_seed=None), 0)
    assert not skip_first_batches(loader, 3)
    assert len(list(iter(loader))) == 26


def test_refused_for_an_unknown_collator():
    loader = DataLoader(_Rows(), batch_size=8, collate_fn=lambda x: x, shuffle=True)
    assert not skip_first_batches(loader, 3)
