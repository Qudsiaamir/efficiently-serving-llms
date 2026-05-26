import pytest

from efficient_llm_serving.batching import (
    GenerationRequest,
    chunk_requests,
    filter_incomplete_requests,
)


def test_generation_request_completion_state():
    request = GenerationRequest(prompt="hello", max_new_tokens=2)
    assert not request.complete

    request.generated_token_count = 2
    assert request.complete


def test_chunk_requests_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        chunk_requests([], batch_size=0)


def test_filter_incomplete_requests():
    done = GenerationRequest(prompt="done", max_new_tokens=1, generated_token_count=1)
    pending = GenerationRequest(prompt="pending", max_new_tokens=1)

    assert filter_incomplete_requests([done, pending]) == [pending]
