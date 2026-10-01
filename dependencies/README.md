# CacheGS fVDB dependency

Read-only inspection on zxcpu2 found fVDB 0.0.1 loaded from the user's Python
3.10 site packages. Its build source is `/home/zyl/XCube/openvdb/fvdb`, at
OpenVDB commit `7c04e6dc75c9d2aba49a3fa31495efab31c62879`, with the two tracked
modifications captured in `fvdb-local.patch`.

The patch fixes build directories/header search paths and removes a redundant
ScalarType caster for this PyTorch setup. It applies at the OpenVDB repository
root. This is dependency provenance, not evidence that a clean install has been
reproduced by the consolidation task. Use a compatible built environment or
rebuild that source with its documented prerequisites before running CacheGS.

No remote package, build or CUDA initialization was changed during inspection.
No generic `pip install fvdb` substitution is made. The historical ProxyGS
compatibility backend does not use this CacheGS model dependency.
