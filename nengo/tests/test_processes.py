from __future__ import absolute_import

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.dists import Distribution, Gaussian
from nengo.processes import BrownNoise, WhiteNoise, WhiteSignal


class DistributionMock(Distribution):
    def __init__(self, retval):
        super(Distribution, self).__init__()
        self.retval = retval
        self.sample_calls = []

    def sample(self, n, d, rng=np.random):
        self.sample_calls.append((n, d, rng))
        return np.ones((n, d)) * self.retval


def test_whitenoise(rng):
    dist = DistributionMock(42)
    process = WhiteNoise(dist, scale=False)
    samples = process.run_steps(5, d=2, rng=rng)
    assert np.all(samples == 42 * np.ones((5, 2)))
    assert process.run_steps(5, rng=rng).shape == (5, 1)
    assert process.run_steps(1, d=1, rng=rng).shape == (1, 1)
    assert process.run_steps(2, d=3, rng=rng).shape == (2, 3)


def test_brownnoise(Simulator, seed, plt):
    d = 5000
    t = 0.5
    std = 1.5
    process = BrownNoise(dist=Gaussian(0, std))
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed)
    sim.run(t)
    samples = sim.data[up]

    trange = sim.trange()
    expected_std = std * np.sqrt(trange)
    atol = 3.5 * expected_std / np.sqrt(d)

    plt.subplot(2, 1, 1)
    plt.title("Five Brown noise signals")
    plt.plot(trange, samples[:, :5])
    plt.subplot(2, 1, 2)
    plt.ylabel("Standard deviation")
    plt.plot(trange, np.abs(np.std(samples, axis=1)), label="Actual")
    plt.plot(trange, expected_std, label="Expected")
    plt.legend(loc='best')

    assert np.all(np.abs(np.mean(samples, axis=1)) < atol)
    assert np.all(np.abs(np.std(samples, axis=1) - expected_std) < atol)


def psd(values, dt=0.001):
    freq = npext.rfftfreq(values.shape[0], d=dt)
    power = 2. * np.std(np.abs(np.fft.rfft(
        values, axis=0)), axis=1) / np.sqrt(values.shape[0])
    return freq, power


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_gaussian_whitenoise(Simulator, rms, seed, plt):
    d = 500
    process = WhiteNoise(Gaussian(0., rms), scale=False)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed)
    sim.run(0.3)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two dimensions of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    val_rms = npext.rms(values, axis=0)
    assert np.allclose(val_rms.mean(), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.2)


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_whitesignal_rms(Simulator, rms, seed, plt):
    t = 1.
    d = 500
    process = WhiteSignal(t, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed)
    sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    assert np.allclose(np.std(values), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1], rms, rtol=0.35)


@pytest.mark.parametrize('high', [5, 50])
def test_whitesignal_high(Simulator, high, seed, plt):
    t = 1.
    rms = 0.5
    d = 500
    process = WhiteSignal(t, high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed)
    sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    dt = sim.dt
    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(len(trange), dt) > high] < rms * 0.5)


def test_whitesignal_dt(Simulator, seed, plt):
    t = 1.
    dt = 0.01
    high = 10
    rms = 0.5
    d = 500
    process = WhiteSignal(t, high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed, dt=dt)
    sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(len(trange), dt) > high] < rms * 0.5)


def test_whitesignal_continuity(Simulator, seed, plt):
    """Test that WhiteSignal is continuous over multiple periods."""
    t = 1.
    high = 10
    rms = 0.5
    process = WhiteSignal(t, high=high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=1)
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed)
    sim.run(4 * t)
    dt = sim.dt
    x = sim.data[up]

    plt.plot(sim.trange(), x)

    # tolerances approximated from derivatives of sine wave of highest freq
    safety_factor = 2.
    a, f = np.sqrt(2) * rms, (2 * np.pi * high) * dt
    assert abs(np.diff(x, axis=0)).max() <= safety_factor * a * f
    assert abs(np.diff(x, n=2, axis=0)).max() <= safety_factor**2 * a * f**2


def test_sampling_shape():
    process = WhiteSignal(0.1)
    assert process.run_steps(1).shape == (1, 1)
    assert process.run_steps(5, d=1).shape == (5, 1)
    assert process.run_steps(1, d=2). shape == (1, 2)


def test_reset(Simulator, seed):
    trun = 0.1

    with nengo.Network() as model:
        u = nengo.Node(WhiteNoise(Gaussian(0, 1), scale=False))
        up = nengo.Probe(u)

    sim = Simulator(model, seed=seed)

    sim.run(trun)
    x = np.array(sim.data[up])

    sim.reset()
    sim.run(trun)
    y = np.array(sim.data[up])

    assert x.shape == y.shape
    assert (x == y).all()
