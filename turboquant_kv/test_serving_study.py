import pytest

from serving_study import TokenClock, summarize
from test_serving_adapter import _tiny_llama


def rows():
    result = []
    for question in range(3):
        for arm in ("native", "packed8_auto"):
            result.append(dict(id=str(question), question_id=str(question), input_sha256=str(question),
                               task="test", context_cap=2048, arm=arm, repeat=0, status="ok",
                               score=1.0 if arm == "native" else 0.5, slo_qualified=True,
                               ttft_s=.2, completion_s=1, inter_token_s=[.1, .2], peak_allocated_bytes=100,
                               peak_incremental_bytes=50, resident_cache_bytes=20))
    return result


def test_timing_callback_counts_generated_tokens_without_prompt():
    torch, model, _ = _tiny_llama()
    clock = TokenClock()
    with torch.inference_mode():
        model.generate(torch.tensor([[1, 7, 9, 4, 3]]), max_new_tokens=4,
                       do_sample=False, streamer=clock)
    assert len(clock.times) == 4
    assert clock.times == sorted(clock.times)


def test_quality_difference_resamples_questions_and_preserves_paired_arms():
    result = summarize(rows())
    assert result["paired_quality"]["packed_minus_native"] == -.5
    assert result["paired_quality"]["question_bootstrap_95_percentile"] == [-.5, -.5]
    assert result["native"]["offered"] == result["native"]["completed"] == 3


def test_failures_stay_in_the_offered_quality_denominator():
    data = rows()
    data[0].update(status="error", score=0.0, slo_qualified=False)
    result = summarize(data)
    assert result["native"]["offered"] == 3
    assert result["native"]["completed"] == result["native"]["slo_qualified"] == 2
    assert result["native"]["mean_score_all_offered"] == 2 / 3


@pytest.mark.parametrize("problem", ["duplicate", "missing", "different_input"])
def test_invalid_pairs_fail_instead_of_silently_changing_comparison(problem):
    data = rows()
    if problem == "duplicate":
        data.append(data[0].copy())
    elif problem == "missing":
        data.pop()
    else:
        data[0]["input_sha256"] = "other"
    with pytest.raises(ValueError):
        summarize(data)


def test_explicit_candidate_pair_keeps_quality_direction():
    data = rows()
    for row in data:
        row["arm"] = "auto" if row["arm"] == "native" else "fused"
    result = summarize(data, arms=("auto", "fused"))
    assert result["paired_quality"]["candidate_minus_baseline"] == -.5
    assert result["auto"]["offered"] == result["fused"]["offered"] == 3
