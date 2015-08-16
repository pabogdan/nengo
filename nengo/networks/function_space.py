import nengo
from nengo.utils.function_space import Function_Space


def FS_Ensemble(FS, net=None, **ens_kwargs):
    """A function space ensemble"""

    if not isinstance(FS, Function_Space):
        raise ValueError("FS argument must be an object of type"
                         " ``Function_Space``")

    # define these to ignore time argument
    def output1(t, x):
        return FS.signal_coeffs(x)

    def output2(t, x):
        return FS.reconstruct(x)

    if net is None:
        net = nengo.Network(label="Function Space")
    with net:
        net.FS_input = nengo.Node(size_in=FS.n_points,
                                  output=output1, label='FS_input')
        net.FS_ens = nengo.Ensemble(n_neurons=FS.n_functions,
                                    dimensions=FS.n_basis, **ens_kwargs)
        nengo.Connection(net.FS_input, net.FS_ens, synapse=None)
        net.FS_output = nengo.Node(size_in=FS.n_basis,
                                   size_out=FS.n_points, output=output2,
                                   label='FS_output')
        nengo.Connection(net.FS_ens, net.FS_output, synapse=None)
    return net
