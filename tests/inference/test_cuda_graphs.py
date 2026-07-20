"""E1 FrameGraphRunner guard logic (CUDA-free). The capture/replay path itself is GPU-only and is
exercised by the Modal bench; here we test the fallback guards and that a disabled runner is a no-op."""

from types import SimpleNamespace

from mira.inference.cuda_graphs import FrameGraphRunner


def _fake_model(n_register_tokens=0, psd_enabled=False):
    return SimpleNamespace(config=SimpleNamespace(n_register_tokens=n_register_tokens, psd_enabled=psd_enabled))


def test_runner_enabled_for_supported_config():
    r = FrameGraphRunner(_fake_model(), n_diffusion_steps=4, noise_level=0.2, schedule_type="linear")
    assert r.disabled is False


def test_runner_disabled_with_register_tokens():
    r = FrameGraphRunner(_fake_model(n_register_tokens=2), 4, 0.2, "linear")
    assert r.disabled is True


def test_runner_enabled_with_psd():
    # Tier B: PSD is supported -- tau_delta is derived from the static delta_ts inside the captured
    # body, so no dynamic input crosses the graph boundary. Bit-exactness is verified GPU-side via
    # `bench_infer_speed --psd --verify-graphs`.
    r = FrameGraphRunner(_fake_model(psd_enabled=True), 4, 0.2, "linear")
    assert r.disabled is False


def test_disabled_runner_run_returns_none():
    # A disabled runner must signal eager fallback (return None) without touching CUDA.
    r = FrameGraphRunner(_fake_model(n_register_tokens=2), 4, 0.2, "linear")
    assert r.run(None, None, None, None) is None
