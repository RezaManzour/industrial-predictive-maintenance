"""Feature engineering for AI4I 2020 (implemented in Phase 2).

Planned physics-motivated features, all computed row-wise so they cannot leak
information across splits:

- ``temp_diff``  = Process temperature [K] - Air temperature [K]
- ``power_w``    = Torque [Nm] * Rotational speed [rpm] * 2*pi / 60
- ``strain``     = Tool wear [min] * Torque [Nm]
- ``Type``       one-hot (L/M/H) for tree and linear models; integer-encoded
  for the PyTorch embedding in Phase 3.

Excluded on purpose: ``UDI``, ``Product ID`` and failure mode flags
TWF/HDF/PWF/OSF/RNF, which are components of the target and would leak it.
"""
