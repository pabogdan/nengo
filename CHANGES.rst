***************
Release History
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - API changes
   - Improvements
   - Behavioural changes
   - Bugfixes
   - Documentation

2.1.0 (unreleased)
==================

**API changes**

- Spiking ``LIF`` neuron models now accept an additional argument,
  ``min_voltage``. Voltages are clipped such that they do not drop below
  this value (previously, this was fixed at 0).
  (`#666 <https://github.com/nengo/nengo/pull/666>`_)
- ``Process`` objects can now be passed directly as node outputs,
  making them easier to use. The ``Process`` interface is also improved
  and is currently the same as the ``Synapse`` interface. However,
  further improvements are pending, and the current implementation
  SHOULD NOT BE RELEASED!
  (`#652 <https://github.com/nengo/nengo/pull/652>`_)
- The ``PES`` learning rule no longer accepts a connection as an argument.
  Instead, error information is transmitted by making a connection to the
  learning rule object (e.g.,
  ``nengo.Connection(error_ensemble, connection.learning_rule)``.
  (`#344 <https://github.com/nengo/nengo/issues/344>`_,
  `#642 <https://github.com/nengo/nengo/pull/642>`_)
- The ``modulatory`` attribute has been removed from ``nengo.Connection``.
  This was only used for learning rules to this point, and has been removed
  in favor of connecting directly to the learning rule.
  (`#642 <https://github.com/nengo/nengo/pull/642>`_)
- Connection weights can now be probed with ``nengo.Probe(conn, 'weights')``,
  and these are always the weights that will change with learning
  regardless of the type of connection. Previously, either ``decoders`` or
  ``transform`` may have changed depending on the type of connection;
  it is now no longer possible to probe ``decoders`` or ``transform``.
  (`#729 <https://github.com/nengo/nengo/pull/729>`_)

**Behavioural changes**

- The sign on the ``PES`` learning rule's error has been flipped to conform
  with most learning rules, in which error is minimized. The error should be
  ``actual - target``. (`#642 <https://github.com/nengo/nengo/pull/642>`_)
- The ``PES`` rule's learning rate is invariant to the number of neurons
  in the presynaptic population. The effective speed of learning should now
  be unaffected by changes in the size of the presynaptic population.
  Existing learning networks may need to be updated; to achieve identical
  behavior, scale the learning rate by ``pre.n_neurons / 100``.
  (`#643 <https://github.com/nengo/nengo/issues/643>`_)
- The ``probeable`` attribute of all Nengo objects is now implemented
  as a property, rather than a configurable parameter.
  (`#671 <https://github.com/nengo/nengo/pull/671>`_)
- Node functions receive ``x`` as a copied NumPy array (instead of a readonly
  view).
  (`#716 <https://github.com/nengo/nengo/issues/716>`_,
  `#722 <https://github.com/nengo/nengo/pull/722>`_)
- The SPA Compare module produces a scalar output (instead of a specific
  vector).
  (`#775 <https://github.com/nengo/nengo/issues/775>`_,
  `#782 <https://github.com/nengo/nengo/pull/782>`_)

**Improvements**

- Added ``PES.pre_tau`` attribute, which sets the time constant on a lowpass
  filter of the presynaptic activity.
  (`#643 <https://github.com/nengo/nengo/issues/643>`_)
- ``EnsembleArray.add_output`` now accepts a list of functions
  to be computed by each ensemble.
  (`#562 <https://github.com/nengo/nengo/issues/562>`_,
  `#580 <https://github.com/nengo/nengo/pull/580>`_)
- Added ``SqrtBeta`` distribution, which describes the distribution
  of semantic pointer elements.
  (`#414 <https://github.com/nengo/nengo/issues/414>`_,
  `#430 <https://github.com/nengo/nengo/pull/430>`_)
- Added ``Triangle`` synapse, which filters with a triangular FIR filter.
  (`#660 <https://github.com/nengo/nengo/pull/660>`_)
- Added ``utils.connection.eval_point_decoding`` function, which
  provides a connection's static decoding of a list of evaluation points.
  (`#700 <https://github.com/nengo/nengo/pull/700>`_)
- Resetting the Simulator now resets all Processes, meaning the
  injected random signals and noise are identical between runs,
  unless the seed is changed (which can be done through
  ``Simulator.reset``).
  (`#582 <https://github.com/nengo/nengo/pull/582>`_,
  `#616 <https://github.com/nengo/nengo/pull/616>`_,
  `#652 <https://github.com/nengo/nengo/pull/652>`_)
- An exception is raised if SPA modules are not properly assigned to an SPA
  attribute.
  (`#730 <https://github.com/nengo/nengo/issues/730>`_,
  `#791 <https://github.com/nengo/nengo/pull/791>`_)

**Bug fixes**

- Fixed issue where setting ``Connection.seed`` through the constructor had
  no effect. (`#724 <https://github.com/nengo/nengo/issues/725>`_)
- Fixed issue in which learning connections could not be sliced.
  (`#632 <https://github.com/nengo/nengo/issues/632>`_)
- Fixed issue when probing scalar transforms.
  (`#667 <https://github.com/nengo/nengo/issues/667>`_,
  `#671 <https://github.com/nengo/nengo/pull/671>`_)
- Fix for SPA actions that route to a module with multiple inputs.
  (`#714 <https://github.com/nengo/nengo/pull/714>`_)

2.0.1 (January 27, 2015)
========================

**Behavioural changes**

- Node functions receive ``t`` as a float (instead of a NumPy scalar)
  and ``x`` as a readonly NumPy array (instead of a writeable array).
  (`#626 <https://github.com/nengo/nengo/issues/626>`_,
  `#628 <https://github.com/nengo/nengo/pull/628>`_)

**Improvements**

- ``rasterplot`` works with 0 neurons, and generates much smaller PDFs.
  (`#601 <https://github.com/nengo/nengo/pull/601>`_)

**Bug fixes**

- Fix compatibility with NumPy 1.6.
  (`#627 <https://github.com/nengo/nengo/pull/627>`_)

2.0.0 (January 15, 2015)
========================

Initial release of Nengo 2.0!
Supports Python 2.6+ and 3.3+.
Thanks to all of the contributors for making this possible!
