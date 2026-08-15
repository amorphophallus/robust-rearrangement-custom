from argparse import Namespace

from scripts.monitor_noise_eval_resources import GIB, _classify


def _args():
    return Namespace(
        warn_available_gib=6.0,
        critical_available_gib=3.0,
        warn_psi=20.0,
        critical_psi=35.0,
        warn_swap_ratio=0.90,
        critical_swap_ratio=0.97,
        warn_disk_free_gib=90.0,
        critical_disk_free_gib=80.0,
    )


def _sample(*, available_gib: float, swap_ratio: float, disk_free_gib: float):
    return {
        "memory": {
            "available_bytes": int(available_gib * GIB),
            "swap_used_ratio": swap_ratio,
        },
        "disk": {"free_bytes": int(disk_free_gib * GIB)},
        "system_pressure": {},
        "user_cgroup": {},
    }


def test_full_stale_swap_does_not_warn_when_memory_is_available():
    severity, reasons = _classify(
        _sample(available_gib=16.0, swap_ratio=1.0, disk_free_gib=101.0),
        _args(),
    )

    assert severity == "ok"
    assert reasons == []


def test_full_swap_is_critical_when_available_memory_is_low():
    severity, reasons = _classify(
        _sample(available_gib=2.0, swap_ratio=1.0, disk_free_gib=101.0),
        _args(),
    )

    assert severity == "critical"
    assert "swap_used=100.0%" in reasons
