from scripts.audit_clean_train_noise_eval import _limit_reported_issues


def test_limit_reported_issues_preserves_count_and_completion_state():
    payload = {
        "issues": ["first", "second", "third"],
        "complete": False,
    }

    limited = _limit_reported_issues(payload, 2)

    assert limited["issues"] == ["first", "second"]
    assert limited["issue_count"] == 3
    assert limited["issues_truncated"] == 1
    assert limited["complete"] is False
    assert payload["issues"] == ["first", "second", "third"]
