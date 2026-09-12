import numpy as np
from PIL import Image

from malecns_cache.graph import load_stub
from runtime.decoder import decode_dn_rates, decode_leg_mn
from runtime.encoder import encode_frame
from runtime.lif import LIFNetwork
from train.readout import apply_linear, fit_linear


def test_encoder_center_brighter_than_black():
    black = np.zeros((240, 320, 3), dtype=np.float32)
    bright = np.zeros((240, 320, 3), dtype=np.float32)
    bright[80:160, 120:200] = 1.0
    dark, _ = encode_frame(black, 64)
    lit, _ = encode_frame(bright, 64)
    assert float(lit.mean()) > float(dark.mean())
    assert lit.shape == (64,)


def test_encoder_roundtrip_jpeg(tmp_path):
    image = Image.new("RGB", (320, 240), (200, 80, 40))
    path = tmp_path / "cube.jpg"
    image.save(path, "JPEG")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r1, r8 = encode_frame(rgb, 16, 4)
    assert r1.shape == (16,) and r8.shape == (4,)
    assert float(r1.mean()) > 0.2


def test_lif_shares_weight_storage(tmp_path):
    graph = load_stub(tmp_path)
    brain = LIFNetwork(graph.weights)
    assert brain.weights is graph.weights


def test_lif_chain_and_finite(tmp_path):
    graph = load_stub(tmp_path)
    brain = LIFNetwork(graph.weights, synaptic_gain=2.0, thresh=0.5)
    brain.set_current(np.array([0]), np.array([3.0]))
    brain.step(50)
    assert np.isfinite(brain.v).all()
    assert float(brain.rate_hz[1]) >= 0
    assert float(brain.spikes.sum()) >= 0


def test_contrast_decoder_right_minus_left(tmp_path):
    graph = load_stub(tmp_path)
    signal = np.zeros(graph.n, dtype=np.float32)
    signal[graph.indices("DNp20_R")] = 8
    signal[graph.indices("DNp20_L")] = 1
    from runtime.decoder import decode_contrast

    cmd = decode_contrast(graph, signal, step=4.0)
    assert cmd.dy_mm > 2.0
    assert cmd.keep_level is True


def test_decoder_right_minus_left(tmp_path):
    graph = load_stub(tmp_path)
    rate = np.zeros(graph.n, dtype=np.float32)
    rate[graph.indices("DNp20_R")] = 200
    rate[graph.indices("DNp20_L")] = 10
    cmd = decode_dn_rates(graph, rate, scale=0.05)
    assert cmd.dy_mm > 0
    assert cmd.keep_level is True


def test_leg_mn_decoder_uses_flexor_extensor_contrast(tmp_path):
    graph = load_stub(tmp_path)
    rate = np.zeros(graph.n, dtype=np.float32)
    rate[graph.indices("Ti_extensor")] = 40
    rate[graph.indices("Ti_flexor")] = 5
    cmd = decode_leg_mn(graph, rate, step=4.0)
    assert cmd.dx_mm > 2.0
    assert cmd.keep_level is True


def test_linear_readout_recovers_map():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(6, 4))
    x = rng.normal(size=(40, 6))
    y = x @ true
    hat = fit_linear(x, y, l2=1e-8)
    pred = apply_linear(hat, x)
    assert np.linalg.norm(pred - y) < 1e-4
