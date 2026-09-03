from scripts import run_vlm_grasp_dit_eval as runner


def test_gpu_quiescence_requires_three_consecutive_recovered_samples(monkeypatch):
    samples = iter([100, 900, 700, 900, 900, 900])
    monkeypatch.setattr(runner, "_gpu_memory_free_mib", lambda: next(samples))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    result = runner._wait_for_gpu_quiescence(
        baseline_free_mib=1000,
        tolerance_mib=200,
        timeout_seconds=60,
    )

    assert result["target_free_mib"] == 800
    assert result["final_free_mib"] == 900
    assert result["sample_count"] == 6
    assert result["minimum_free_mib"] == 100
