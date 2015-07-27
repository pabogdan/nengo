from __future__ import absolute_import

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.dists import Distribution, Gaussian
from nengo.processes import BrownNoise, FilteredNoise, WhiteNoise, WhiteSignal
from nengo.synapses import Lowpass


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


def test_brownnoise(rng, plt):
    d = 5000
    t = 0.5
    dt = 0.001
    std = 1.5

    process = BrownNoise(dist=Gaussian(0, std))
    samples = process.run(t, d=d, dt=dt, rng=rng)

    trange = process.trange(t, dt=dt)
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
def test_gaussian_whitenoise(rms, rng, plt):
    d = 500
    t = 0.1
    dt = 0.001

    process = WhiteNoise(Gaussian(0., rms))

    values = process.run(t, d=d, dt=dt, rng=rng)
    freq, val_psd = psd(values)

    trange = process.trange(t, dt=dt)
    plt.subplot(2, 1, 1)
    plt.title("First two dimensions of white noise process, rms=%.1f" % rms)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')

    val_rms = npext.rms(values, axis=0) * np.sqrt(dt)
    assert np.allclose(val_rms.mean(), rms, rtol=0.02)
    assert np.allclose(val_psd[1:-1] * np.sqrt(dt), rms, rtol=0.2)


@pytest.mark.parametrize('rms', [0.5, 1, 100])
def test_whitesignal_rms(rms, rng, plt):
    d = 500
    t = 1
    dt = 0.001

    process = WhiteSignal(t, rms=rms)
    values = process.run(t, d=d, dt=dt, rng=rng)
    freq, val_psd = psd(values)

    trange = process.trange(t, dt=dt)
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
def test_whitesignal_high(high, rng, plt):
    rms = 0.5
    d = 500
    t = 1
    dt = 0.001

    process = WhiteSignal(t, high, rms=rms)
    values = process.run(t, d=d, dt=dt, rng=rng)
    freq, val_psd = psd(values)

    trange = process.trange(t, dt=dt)
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(t, dt) > high] < rms * 0.5)


def test_whitesignal_dt(rng, plt):
    rms = 0.5
    high = 10
    d = 500
    t = 1
    dt = 0.01

    process = WhiteSignal(t, high, rms=rms)
    values = process.run(t, d=d, dt=dt, rng=rng)
    freq, val_psd = psd(values, dt=dt)

    trange = process.trange(t, dt=dt)
    plt.subplot(2, 1, 1)
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle='steps')
    plt.xlim(right=high * 2.0)

    assert np.allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(t, dt) > high] < rms * 0.5)


def test_whitesignal_continuity(rng, plt):
    """Test that WhiteSignal is continuous over multiple periods."""
    rms = 0.5
    high = 10
    dt = 0.001
    t = 1
    process = WhiteSignal(t, high=high, rms=rms)
    x = process.run(4 * t, d=1, dt=dt, rng=rng)

    plt.plot(process.ntrange(len(x), dt=dt), x)

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


def test_reset(seed):
    trun = 0.1

    with nengo.Network() as model:
        u = nengo.Node(WhiteNoise(Gaussian(0, 1), scale=False))
        up = nengo.Probe(u)

    sim = nengo.Simulator(model, seed=seed)

    sim.run(trun)
    x = np.array(sim.data[up])

    sim.reset()
    sim.run(trun)
    y = np.array(sim.data[up])

    assert (x == y).all()


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = WhiteNoise(dist=Gaussian(0.3, 0.2))
    b = WhiteNoise(dist=Gaussian(0.3, 0.2))
    c = FilteredNoise(dist=Gaussian(0.3, 0.2), synapse=Lowpass(0.02))

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely

    with pytest.raises(ValueError):
        a.dist = Gaussian(0.3, 0.5)  # test that dist param is frozen
    with pytest.raises(ValueError):
        a.dist.std = 0.4  # test that dist object is frozen


def test_seed(Simulator, seed):
    with nengo.Network() as model:
        a = nengo.Node(WhiteSignal(0.1, high=100, seed=seed))
        b = nengo.Node(WhiteSignal(0.1, high=100, seed=seed+1))
        c = nengo.Node(WhiteSignal(0.1, high=100))
        d = nengo.Node(WhiteNoise(seed=seed))
        e = nengo.Node(WhiteNoise())
        ap = nengo.Probe(a)
        bp = nengo.Probe(b)
        cp = nengo.Probe(c)
        dp = nengo.Probe(d)
        ep = nengo.Probe(e)

    sim1 = nengo.Simulator(model)
    sim1.run(0.1)

    sim2 = nengo.Simulator(model)
    sim2.run(0.1)

    tols = dict(atol=1e-7, rtol=1e-4)
    assert np.allclose(sim1.data[ap], sim2.data[ap], **tols)
    assert np.allclose(sim1.data[bp], sim2.data[bp], **tols)
    assert not np.allclose(sim1.data[cp], sim2.data[cp], **tols)
    assert not np.allclose(sim1.data[ap], sim1.data[bp], **tols)
    assert np.allclose(sim1.data[dp], sim2.data[dp], **tols)
    assert not np.allclose(sim1.data[ep], sim2.data[ep], **tols)


def test_present_input(Simulator, rng):
    n = 5
    c, ni, nj = 3, 8, 8
    images = rng.normal(size=(n, c, ni, nj))
    pres_time = 0.1

    model = nengo.Network()
    with model:
        u = nengo.Node(nengo.processes.PresentInput(images, pres_time))
        up = nengo.Probe(u)

    sim = Simulator(model)
    sim.run(1.0)
    t = sim.trange()
    i = np.floor(t / pres_time) % n
    y = sim.data[up].reshape(len(t), c, ni, nj)
    for ii, image in zip(i, y):
        assert np.allclose(image, images[ii], rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize('local', [False, True])
def test_conv2(local, Simulator, rng):
    f = 4
    c = 2
    ni, nj = 30, 32
    si, sj = 5, 3

    fshape = (f, ni, nj, c, si, sj) if local else (f, c, si, sj)
    filters = rng.normal(size=fshape)
    biases = rng.normal(size=f)
    image = rng.normal(size=(c, ni, nj))

    result = np.zeros((f, ni, nj))
    result += biases.reshape(-1, 1, 1)
    si2 = (si - 1) / 2
    sj2 = (sj - 1) / 2
    for i in range(ni):
        for j in range(nj):
            i0, i1 = i - si2, i + si2 + 1
            j0, j1 = j - sj2, j + sj2 + 1
            sli = slice(max(-i0, 0), min(ni + si - i1, si))
            slj = slice(max(-j0, 0), min(nj + sj - j1, sj))
            w = (filters[:, i, j, :, sli, slj] if local else
                 filters[:, :, sli, slj])
            xij = image[:, max(i0, 0):min(i1, ni), max(j0, 0):min(j1, nj)]
            result[:, i, j] += np.dot(xij.ravel(), w.reshape(f, -1).T)

    model = nengo.Network()
    with model:
        u = nengo.Node(image.ravel())
        v = nengo.Node(nengo.processes.Conv2((c, ni, nj), filters, biases))
        nengo.Connection(u, v, synapse=None)
        vp = nengo.Probe(v)

    sim = Simulator(model)
    sim.run_steps(3)
    y = sim.data[vp][-1].reshape((f, ni, nj))
    assert np.allclose(result, y, rtol=1e-4, atol=1e-7)
