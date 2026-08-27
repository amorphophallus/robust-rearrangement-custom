from copy import deepcopy

import numpy as np
import pytest

from src.dataset.lmdb import (
    DEFAULT_FRAME_COMPRESSION,
    DEFAULT_FRAME_COMPRESSION_LEVEL,
    build_frame_specs,
    pack_frame,
    unpack_frame,
)


def example_images():
    return {
        "color_image1": np.zeros((32, 48, 3), dtype=np.uint8),
        "color_image2": np.full((32, 48, 3), 17, dtype=np.uint8),
        "depth_image1": np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48),
        "depth_image2": np.ones((32, 48), dtype=np.float32),
    }


def test_new_lmdb_datasets_default_to_fast_zstd_compression():
    assert DEFAULT_FRAME_COMPRESSION == "zstd"
    assert DEFAULT_FRAME_COMPRESSION_LEVEL == 1


def test_zstd_frame_round_trip_and_key_selection():
    pytest.importorskip("zstandard")
    images = example_images()
    raw_specs = build_frame_specs(images)
    compressed_specs = deepcopy(raw_specs)
    compressed_specs["compression"] = {"codec": "zstd", "level": 1}

    raw_payload = pack_frame(images, raw_specs)
    compressed_payload = pack_frame(images, compressed_specs)

    assert len(compressed_payload) < len(raw_payload)
    decoded = unpack_frame(compressed_payload, compressed_specs)
    for key, expected in images.items():
        np.testing.assert_array_equal(decoded[key], expected)

    selected = unpack_frame(
        compressed_payload,
        compressed_specs,
        keys=["depth_image2"],
    )
    assert list(selected) == ["depth_image2"]
    np.testing.assert_array_equal(selected["depth_image2"], images["depth_image2"])


def test_unknown_frame_compression_codec_is_rejected():
    images = example_images()
    frame_specs = build_frame_specs(images)
    frame_specs["compression"] = {"codec": "unknown"}

    with pytest.raises(ValueError, match="Unsupported LMDB frame compression codec"):
        pack_frame(images, frame_specs)
    with pytest.raises(ValueError, match="Unsupported LMDB frame compression codec"):
        unpack_frame(b"payload", frame_specs)
