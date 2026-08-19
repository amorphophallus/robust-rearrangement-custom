import pytest

from types import SimpleNamespace

from services.vlm_guidance.native_sft import (
    configure_native_processor,
    parse_native_prediction,
    pixels_to_qwen,
    try_parse_native_prediction,
)


def test_native_processor_uses_left_padding_for_batched_generation():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    configure_native_processor(processor)
    assert processor.tokenizer.padding_side == "left"


def test_parse_native_prediction_accepts_plain_or_prefixed_json():
    expected = ("pick", [100.5, 120.0])
    assert parse_native_prediction(
        '{"skill":"pick","target_point_2d":[100.5,120.0]}'
    ) == expected
    assert parse_native_prediction(
        'answer: {"skill":"pick","target_point_2d":[100.5,120.0]} trailing'
    ) == expected


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"skill":"unknown","target_point_2d":[10,20]}',
        '{"skill":"pick","target_point_2d":null}',
        '{"skill":"pick","target_point_2d":[320,20]}',
    ],
)
def test_parse_native_prediction_rejects_unsafe_output(text):
    with pytest.raises(ValueError):
        parse_native_prediction(text)


def test_pixels_to_qwen_uses_front_image_axes():
    assert pixels_to_qwen([319.0, 239.0]) == [1000.0, 1000.0]


def test_tolerant_parser_retains_skill_when_point_is_null():
    skill, point, error = try_parse_native_prediction(
        '{"skill":"insert","target_point_2d":null}'
    )
    assert skill == "insert"
    assert point is None
    assert error == "target_point_2d is not a two-value list"
