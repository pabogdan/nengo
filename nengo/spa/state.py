import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.networks.ensemblearray import EnsembleArray


class State(Module):
    """A module capable of representing a single vector, with optional memory.

    This is a minimal SPA module, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the vector
    subdimensions : int
        Size of the individual ensembles making up the vector. Must divide
        evenly into dimensions
    neurons_per_dimensions : int
        Number of neurons in an ensemble will be this*subdimensions
    float : float
        Gain of feedback connection. Set to 1.0 for perfect memory. Other
        non-zero values will create a decaying memory
    feedback_synapse : float, nengo.Synapse
        The synapse on the feedback connection
    vocab : Vocabulary, optional
        The vocabulary to use to interpret this vector
    """

    def __init__(self, dimensions, subdimensions=16, neurons_per_dimension=50,
                 feedback=0.0, feedback_synapse=0.01, vocab=None, label=None,
                 seed=None, add_to_container=None):
        super(State, self).__init__(label, seed, add_to_container)

        if vocab is None:
            # use the default one for this dimensionality
            vocab = dimensions
        elif vocab.dimensions != dimensions:
            raise ValueError('Dimensionality of given vocabulary (%d) does not'
                             'match dimensionality of buffer (%d)' %
                             (vocab.dimensions, dimensions))

        # Subdimensions should be at most the number of dimensions
        subdimensions = min(dimensions, subdimensions)

        if dimensions % subdimensions != 0:
            raise ValueError('Number of dimensions(%d) must be divisible by '
                             'subdimensions(%d)' % (dimensions, subdimensions))

        with self:
            self.input = nengo.Node(size_in=dimensions)
            self.output = nengo.Node(size_in=dimensions)
            self.state_ensembles = EnsembleArray(
                neurons_per_dimension * subdimensions,
                dimensions // subdimensions,
                ens_dimensions=subdimensions,
                radius=np.sqrt(float(subdimensions) / dimensions),
                label='state')

        self.inputs = dict(default=(self.input, vocab))
        self.outputs = dict(default=(self.output, vocab))

        with self:
            nengo.Connection(self.input, self.state_ensembles.input,
                             synapse=None)

            if feedback != 0.0 and feedback is not None:
                nengo.Connection(self.state_ensembles.output,
                                 self.state_ensembles.input,
                                 transform=feedback, synapse=feedback_synapse)
