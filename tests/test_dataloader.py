import gc
import itertools
import weakref

import pytest
from torch.utils.data import Dataset

from src.dataset.dataloader import EndlessDataloader, FixedStepsDataloader


class CountingDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.access_count = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.access_count += 1
        return self.access_count


class Payload:
    pass


class EphemeralDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.refs = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        payload = Payload()
        self.refs.append(weakref.ref(payload))
        return payload


def first_item(batch):
    return batch[0]


def scalar_batches(loader, count=None):
    batches = loader if count is None else itertools.islice(loader, count)
    return [int(batch.item()) for batch in batches]


def test_fixed_steps_reloads_data_instead_of_caching_batches():
    dataset = CountingDataset(size=2)
    loader = FixedStepsDataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        n_batches=5,
    )

    assert scalar_batches(loader) == [1, 2, 3, 4, 5]
    assert dataset.access_count == 5


def test_fixed_steps_does_not_retain_yielded_batches():
    dataset = EphemeralDataset(size=10)
    loader = FixedStepsDataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        n_batches=2,
        collate_fn=first_item,
    )
    iterator = iter(loader)

    batch = next(iterator)
    batch_ref = weakref.ref(batch)
    del batch
    gc.collect()

    assert batch_ref() is None


def test_fixed_steps_rejects_empty_underlying_dataloader():
    loader = FixedStepsDataloader(
        CountingDataset(size=0),
        batch_size=1,
        num_workers=0,
        n_batches=1,
    )

    with pytest.raises(RuntimeError, match="empty underlying dataloader"):
        next(iter(loader))


def test_endless_dataloader_reloads_data_instead_of_caching_batches():
    dataset = CountingDataset(size=2)
    loader = EndlessDataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    assert scalar_batches(loader, count=5) == [1, 2, 3, 4, 5]
    assert dataset.access_count == 5
