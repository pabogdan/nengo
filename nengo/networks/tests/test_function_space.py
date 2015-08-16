import numpy as np

import nengo

from nengo.networks.function_space import FS_Ensemble
from nengo.utils.function_space import gaussian, SVD_Function_Space
from nengo.utils.distributions import Uniform


def test_FS_ensemble(Simulator, nl, plt):

    # parameters
    n_neurons = 2000  # number of neurons
    domain_dim = 1
    dist_args = [Uniform(-1, 1)]
    base_func = gaussian

    # function space object
    FS = SVD_Function_Space(base_func, domain_dim, dist_args,
                            n_functions=n_neurons, n_basis=20)

    # test input is a gaussian bumps function
    # generate a bunch of gaussian functions
    input_func = FS.sample_comb(4, dist_args)
    # radius to use for ensemble
    f_radius = np.linalg.norm(FS.signal_coeffs(input_func))

    # evaluation points are gaussian bumps functions
    n_eval_points = 400
    eval_points = FS.sample_eval_points(n_eval_points, 4, dist_args)

    encoders = FS.encoder_coeffs()
    net = FS_Ensemble(FS, eval_points=eval_points, radius=f_radius,
                      encoders=encoders)

    with net:
        input = nengo.Node(output=input_func)
        nengo.Connection(input, net.FS_input)
        func_probe = nengo.Probe(net.FS_output, synapse=0.1)

    sim = Simulator(net)
    sim.run(2)

    reconstruction = sim.data[func_probe][400]
    true_f = input_func

    plt.saveas = "func_repr.pdf"

    plt.plot(FS.domain, reconstruction, label='model_f')
    plt.plot(FS.domain, true_f, label='true_f')
    plt.legend(loc='best')

    assert np.allclose(true_f, reconstruction, atol=0.2)
