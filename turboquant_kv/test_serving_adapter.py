from serving_adapter import generate
import pytest


class Model:
    def __init__(self):
        self.config = object()
        self.calls = []

    def generate(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return "native-result"


class Input:
    device = "cpu"


def test_cpu_uses_native_model_cache_path():
    model = Model()
    assert generate(model, Input(), bit_width=8, max_cache_len=100) == "native-result"
    assert len(model.calls) == 1
    assert model.calls[0][1] == {}


def test_invalid_bit_width_is_rejected_before_cache_construction():
    model = Model()
    class CudaInput:
        device = "cuda"
    try:
        generate(model, CudaInput(), bit_width=0, max_cache_len=10)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid bit width accepted")
    assert model.calls == []


def _tiny_llama():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(17)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=None,
    )
    return torch, transformers.LlamaForCausalLM(config).eval(), config


def test_cpu_adapter_matches_tiny_random_llama_native_generation():
    torch, model, _ = _tiny_llama()
    inputs = torch.tensor([[1, 7, 9, 4, 3]])
    with torch.inference_mode():
        expected = model.generate(inputs, max_new_tokens=4, do_sample=False)
        actual = generate(
            model, inputs, bit_width=4, max_cache_len=16,
            max_new_tokens=4, do_sample=False,
        )
    assert torch.equal(actual, expected)


def test_cpu_packed_cache_generation_quality_and_length_controls():
    torch, model, config = _tiny_llama()
    from cache import TQPackedCache

    inputs = torch.tensor([[1, 7, 9, 4, 3]])
    with torch.inference_mode():
        reference = model(inputs, use_cache=True)
        cache = TQPackedCache(
            config, bit_width=4, max_cache_len=16, device="cpu", use_kernel=False,
        )
        packed = model(inputs, use_cache=True, past_key_values=cache)
        rel_logit_rmse = (
            (packed.logits - reference.logits).square().mean().sqrt()
            / reference.logits.square().mean().sqrt()
        ).item()
        assert rel_logit_rmse < 0.15
        assert cache.get_seq_length() == inputs.shape[1]
        assert cache.get_max_length() == 16
        assert cache.get_max_cache_shape() == 16

        generated = model.generate(
            inputs,
            past_key_values=TQPackedCache(
                config, bit_width=4, max_cache_len=16, device="cpu", use_kernel=False,
            ),
            use_cache=True,
            max_new_tokens=4,
            do_sample=False,
        )
    assert generated.shape == (1, inputs.shape[1] + 4)

    bounded = TQPackedCache(
        config, bit_width=4, max_cache_len=2, device="cpu", use_kernel=False,
    )
    keys = torch.zeros(1, 2, 2, 8)
    bounded.update(keys, keys, 0)
    assert bounded.get_seq_length() == 2
    with pytest.raises(RuntimeError, match="max_cache_len=2"):
        bounded.update(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8), 0)
