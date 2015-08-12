import numpy as np

import nengo

from nengo.utils.function_space import *
from nengo.utils.distributions import Uniform

sigma = 0.2


def gaussian(points, center):
    return np.exp(-(points - center)**2 / (2 * sigma ** 2))


def test_function_repr(Simulator, nl, plt):

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
    gaussians = generate_functions(base_func, 4, *dist_args)
    # evaluate them on the domain and add up
    input_func = np.sum([func(FS.domain) for func in gaussians], axis=0)

    # evaluation points are gaussian bumps functions
    n_eval_points = 400
    funcs = []
    for _ in range(n_eval_points):
        gaussians = generate_functions(base_func, 4, *dist_args)
        funcs.append(np.sum([func(FS.domain)
                             for func in gaussians], axis=0))
    eval_points = FS.signal_coeffs(np.array(funcs).T)

    # vector space coefficients
    signal_coeffs = FS.signal_coeffs(input_func)
    encoders = FS.encoder_coeffs()

    f_radius = np.linalg.norm(signal_coeffs)  # radius to use for ensemble

    with nengo.Network() as model:
        # represents the function
        f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
                           encoders=encoders, radius=f_radius,
                           eval_points=eval_points, label='f')
        signal = nengo.Node(output=signal_coeffs)
        nengo.Connection(signal, f)

        probe_f = nengo.Probe(f, synapse=0.1)

    sim = Simulator(model)
    sim.run(6)

    reconstruction = FS.reconstruct(sim.data[probe_f][400])
    true_f = input_func.flatten()

    plt.saveas = "func_repr.pdf"

    plt.plot(FS.domain, reconstruction, label='model_f')
    plt.plot(FS.domain, true_f, label='true_f')
    plt.legend(loc='best')

    assert np.allclose(true_f, reconstruction, atol=0.2)


def test_fourier_basis(plt):
    """Testing fourier basis, not in neurons"""

    # parameters
    domain_dim = 1

    FS = Fourier(domain_dim)

    # test input is a gaussian bumps function
    # generate a bunch of gaussian functions
    gaussians = generate_functions(gaussian, 4, Uniform(-1, 1))
    # evaluate them on the domain and add up
    true = np.sum([func(FS.domain) for func in gaussians], axis=0)
    model = FS.reconstruct(FS.signal_coeffs(true))

    plt.figure('Testing Fourier Basis')
    plt.plot(FS.domain, true, label='Function')
    plt.plot(FS.domain, model, label='reconstruction')
    plt.legend(loc='best')
    plt.savefig('utils.test_function_space.test_fourier_basis.pdf')

    # clip ends because of Gibbs phenomenon
    assert np.allclose(true[200:-200], model[200:-200], atol=0.2)


def function_gen_eval(plt):
    """Plot the output of function generation to check if it works"""
    plt.plot(function_values(generate_functions(gaussian, 20, Uniform(-1, 1)),
                             uniform_cube(1, 1, 0.001)),
             label='function_values')


def uniform_cube():
    """Plot the output of domain point generation to check if it works"""
    points = uniform_cube(2, 1, 0.1)
    plt.scatter(points[0, :], points[1, :], label='points')
