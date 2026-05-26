from efficient_llm_serving.benchmarking import summarize_latencies


def test_summarize_latencies_reports_expected_fields():
    summary = summarize_latencies([0.3, 0.1, 0.2])

    assert summary.count == 3
    assert summary.p50_s == 0.2
    assert summary.min_s == 0.1
    assert summary.max_s == 0.3


def test_summarize_latencies_rejects_empty_input():
    import pytest

    with pytest.raises(ValueError):
        summarize_latencies([])
