import numpy as np
import pytest

import nengo
from nengo.utils.compat import range
from nengo.utils.functions import HilbertCurve
from nengo.utils.numpy import maxint, rmse


def test_sine_waves(Simulator, plt, seed):
    radius = 2
    dim = 5
    product = nengo.networks.Product(
        200, dim, radius, net=nengo.Network(seed=seed))

    func_A = lambda t: np.sqrt(radius)*np.sin(np.arange(1, dim+1)*2*np.pi*t)
    func_B = lambda t: np.sqrt(radius)*np.sin(np.arange(dim, 0, -1)*2*np.pi*t)
    with product:
        input_A = nengo.Node(func_A)
        input_B = nengo.Node(func_B)
        nengo.Connection(input_A, product.A)
        nengo.Connection(input_B, product.B)
        p = nengo.Probe(product.output, synapse=0.005)

    sim = Simulator(product)
    sim.run(1.0)

    t = sim.trange()
    AB = np.asarray(list(map(func_A, t))) * np.asarray(list(map(func_B, t)))
    delay = 0.013
    offset = np.where(t >= delay)[0]

    for i in range(dim):
        plt.subplot(dim+1, 1, i+1)
        plt.plot(t + delay, AB[:, i])
        plt.plot(t, sim.data[p][:, i])
    plt.xlim(right=t[-1])

    assert rmse(AB[:len(offset), :], sim.data[p][offset, :]) < 0.2


@pytest.mark.benchmark
@pytest.mark.slow
def test_product_benchmark(analytics, seed):
    n_trials = 50
    hc = HilbertCurve(n=4)  # Increase n to cover the input space more densly
    duration = 5.           # Simulation duration (s)
    wait_duration = 0.5     # Duration (s) to wait in the beginning to have a
                            # stable representation
    n_neurons = 100
    n_eval_points = 1000

    rng = np.random.RandomState(seed)

    def stimulus_fn(t):
        return np.squeeze(hc(t / duration).T * 2 - 1)

    def run_trial():
        model = nengo.Network(seed=rng.randint(maxint))
        with model:
            model.config[nengo.Ensemble].n_eval_points = n_eval_points

            stimulus = nengo.Node(
                output=lambda t: stimulus_fn(max(0., t - wait_duration)),
                size_out=2)

            product_net = nengo.networks.Product(n_neurons, 1)
            nengo.Connection(stimulus[0], product_net.A)
            nengo.Connection(stimulus[1], product_net.B)
            probe_test = nengo.Probe(product_net.output)

            ens_direct = nengo.Ensemble(1, dimensions=2, neuron_type=nengo.Direct())
            result_direct = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens_direct)
            nengo.Connection(
                ens_direct, result_direct, function=lambda x: x[0] * x[1],
                synapse=None)
            probe_direct = nengo.Probe(result_direct)

        sim = nengo.Simulator(model)
        sim.run(duration + wait_duration, progress_bar=False)

        selection = sim.trange() > wait_duration
        test = sim.data[probe_test][selection]
        direct = sim.data[probe_direct][selection]
        return rmse(test, direct)

    error_data = [run_trial() for i in range(n_trials)]
    analytics.add_data(
        'error', error_data, "Multiplication RMSE. Shape: n_trials")
