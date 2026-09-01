"""Cross-version pickle loading helpers.

NumPy 2 serializes arrays through ``numpy._core``. NumPy 1 environments, such
as the current RR training environment, expose the same objects under
``numpy.core`` and otherwise fail before dataset conversion can inspect the
trajectory.
"""

from __future__ import annotations

import gzip
import lzma
import pickle
from pathlib import Path
from typing import BinaryIO, Union


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 1- or 2-authored arrays in either namespace layout."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_pickle_file(file: BinaryIO):
    return NumpyCompatUnpickler(file).load()


def load_pickle_path(path: Union[Path, str]):
    path = Path(path)
    if path.name.endswith(".pkl.xz") or path.suffix == ".xz":
        opener = lzma.open
    elif path.name.endswith(".pkl.gz") or path.suffix == ".gz":
        opener = gzip.open
    elif path.suffix == ".pkl":
        opener = open
    else:
        raise ValueError(f"Invalid pickle file extension: {path}")
    with opener(path, "rb") as file:
        return load_pickle_file(file)


__all__ = ["NumpyCompatUnpickler", "load_pickle_file", "load_pickle_path"]
