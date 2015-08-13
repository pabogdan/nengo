import nengo
from nengo.utils.function_space import Function_Space
from nengo.utils.network import with_self


class FS_Network(nengo.Network):
    """ A function space network.

    This network is meant to simplify working with function spaces, and
    essentially encapsulates an ensemble.  A function space object, which
    defines the functions space (see utils/function_space.py), is required.
    This object can also be used to compute appropriate evalutation points and
    the radius for the ensemble.
    """

    def __init__(self, FS, **ens_kwargs):
        if isinstance(FS, Function_Space):
               self.FS = FS
        else:
            raise ValueError("FS argument must be an object of type"
                                " ``Function_Space``")

        with self:
            self.FS_ens = nengo.Ensemble(dimensions=FS.n_basis,
                                         n_neurons=FS.n_functions,
                                         encoders=FS.encoder_coeffs(),
                                         **ens_kwargs)

    @with_self
    def add_input_node(self, node_output):
        self.input_node = nengo.Node(output=self.FS.signal_coeffs(node_output))
        nengo.Connection(self.input_node, self.FS_ens)

    def probe(self, synapse=0.1):
        self.probe = nengo.Probe(self.FS_ens, synapse=synapse)

    def reconstruct(self, sim, t):
        return self.FS.reconstruct(sim.data[self.probe][round(t/sim.dt)])

    def connect_to(self, nengo_object):
        nengo.Connection(self.FS_ens, nengo_object)
