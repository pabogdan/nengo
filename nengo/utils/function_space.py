from __future__ import absolute_import

import numpy as np

from nengo.utils.numpy import array
from nengo.utils.distributions import Uniform


"""TODO: -Finish fourier class"""

def generate_functions(function, n, *arg_dists):
    """
    Parameters:
    -----------

    function: callable,
       a real-valued function to be used as a basis, ex. gaussian

    n: int,
       number of functions to generate

    arg_dists: instances of nengo distributions
       distributions to sample arguments (eg. mean of a gaussian function) from

    Returns:
    --------
    a list of callable functions that have some parameters fixed
    based on the distributions given
    """

    # get argument samples to make different functions
    arg_samples = np.array([arg_dist.sample(n) for arg_dist in arg_dists]).T

    functions = []
    for i in range(n):
        def func(points, i=i):
            args = [points]
            args.extend(arg_samples[i])
            # since function outputs are 1D, flatten
            return function(*args).flatten()
        functions.append(func)

    return functions


def uniform_cube(domain_dim, radius=1, d=0.001):
    """Returns uniformly spaced points in a hypercube.

    The hypercube is defined by the given radius and dimension.

    Parameters:
    ----------
    domain_dim: int
       the dimension of the domain

    radius: float, optional
       2 * radius is the length of a side of the hypercube

    d: float, optional
       the discretization spacing (a small float)

    Returns:
    -------
    ndarray of shape (domain_dim, radius/d)

    """

    if domain_dim == 1:
        domain_points = np.arange(-radius, radius, d)
        domain_points = array(domain_points, min_dims=2)
    else:
        axis = np.arange(-radius, radius, d)
        # uniformly spaced points of a hypercube in the domain
        grid = np.meshgrid(*[axis for _ in range(domain_dim)])
        domain_points = np.vstack(map(np.ravel, grid))
    return domain_points


def function_values(functions, points):
    """The values of the function on ``points``.

    Returns:
    --------
    ndarray of shape (n_points, n_functions).
    """

    values = np.empty((len(points), len(functions)))
    for i, function in enumerate(functions):
        values[:, i] = function(points).flatten()
    return values


class Function_Space(object):
    """A helper class for using function spaces in nengo.

    Parameters:
    -----------

    domain_dim: int,
      The dimension of the domain on which the function space is defined

    n_basis: int, optional
      Number of orthonormal basis functions to use

    d: float, optional
       the discretization factor (used in spacing the domain points)

    radius: float, optional
       2 * radius is the length of a side of the hypercube of the domain

    n_functions: int, optional
      Number of functions used to tile the space.
    """

    def __init__(self, domain_dim, n_basis=20, d=0.001, radius=1,
                 n_functions=200):

        self.domain = uniform_cube(domain_dim, radius, d)
        self.n_basis = n_basis
        self.n_functions = n_functions

    def get_basis(self):
        raise NotImplementedError("Must be implemented by subclasses")

    def reconstruct(self, coefficients):
        raise NotImplementedError("Must be implemented by subclasses")

    def encoder_coeffs(self):
        raise NotImplementedError("Must be implemented by subclasses")

    def signal_coeffs(self, signal):
        raise NotImplementedError("Must be implemented by subclasses")


def gaussian(points, center):
    return np.exp(-(points - center)**2 / (2 * 0.2 ** 2))


class SVD_Function_Space(Function_Space):
    """A function space subclass where the basis is derived from the SVD.

    A base function is used to tile the space with a bunch of functions. The
    SVD is then used to compute a basis from those functions.

    Parameters:
    -----------

    fn: callable,
      The function that will be used for tiling the space.

    dist_args: list of nengo Distributions
       The distributions to sample functions from.

    """

    def __init__(self, fn=gaussian, domain_dim=1, dist_args=[Uniform(-1, 1)],
                 n_functions=200, n_basis=20, d=0.001, radius=1):

        super(SVD_Function_Space, self).__init__(domain_dim, n_basis, d,
                                                 radius, n_functions)

        self.n_functions = n_functions

        self.base_func = fn

        self.fns = function_values(generate_functions(fn, n_functions,
                                                      *dist_args),
                                   self.domain)

        self.dx = d ** self.domain.shape[1]  # volume element for integration
        self.n_basis = n_basis

        # basis must be orthonormal
        self.U, self.S, V = np.linalg.svd(self.fns)
        self.basis = self.U[:, :self.n_basis] / np.sqrt(self.dx)

    def select_top_basis(self, n_basis):
        self.n_basis = n_basis
        self.basis = self.U[:, :n_basis] / np.sqrt(self.dx)

    def get_basis(self):
        return self.basis

    def singular_values(self):
        return self.S

    def reconstruct(self, coefficients):
        """Linear combination of the basis functions"""
        return np.dot(self.basis, coefficients)

    def encoder_coeffs(self):
        """Project encoder functions onto basis to get encoder coefficients."""
        return self.signal_coeffs(self.fns)

    def signal_coeffs(self, signal):
        """Project a given signal onto basis to get signal coefficients.
           Size returned is (n_signals, n_basis)"""
        return np.dot(signal.T, self.basis) * self.dx

    def sample_comb(self, k, dist_args):
        """Create a sample linear combination by summing k random variations
        of the initial function used for tiling"""
        functions = generate_functions(self.base_func, 4, *dist_args)
        # evaluate them on the domain and add up
        sample_input = np.sum([func(self.domain) for func in functions],
                              axis=0)
        return sample_input

    def sample_eval_points(self, n_eval_points, k, dist_args):
        """Create some evalutations points as linear combinations of k
        random variations of the initial function used for tiling"""
        funcs = []
        for _ in range(n_eval_points):
            funcs.append(self.sample_comb(k, dist_args))
        return self.signal_coeffs(np.array(funcs).T)

# class Fourier(Function_Space):
#     """A function space subclass that uses the Fourier basis."""

#     def __init__(self, domain_dim, n_basis=20, d=0.001, radius=1):

#         super(Fourier, self).__init__(domain_dim, n_basis, d, radius)

#     def reconstruct(self, coefficients):
#         """inverse fourier transform"""
#         return np.fft.irfft(coefficients, len(self.domain))

#     def encoder_coeffs(self):
#         """Apply the fourier transform to the encoder functions."""
#         return self.signal_coeffs(self.fns)

#     def signal_coeffs(self, signal):
#         """Apply the Discrete fourier transform to the signal"""
#         # throw out higher frequency coefficients
#         return np.fft.rfft(signal.T)[:self.n_basis]
