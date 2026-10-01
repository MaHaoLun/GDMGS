# Third-party notices

`upstream/ProxyGS/` preserves the complete local Proxy-GS-eac937e8 source tree
and its third-party licenses. Its native anchor construction/query directory
is overlaid from the later GDM-GS joint-query experiment's retained source.

`system/` contains source derived from that model/renderer plus GDM-GS research
implementations. The inherited research-use license is preserved in
`system/LICENSE.md`; the original repository's Apache license does not override
these terms. Source copyright headers and bundled third-party license files
remain in place.

The staged gsplat intersection extension retains its own LICENSE and source
provenance under `system/staged_isect_src/`. PyTorch, gsplat, NumPy, SciPy,
nvdiffrast, torch-scatter and other installed dependencies keep their licenses.
Their private precompiled binaries and trained datasets/checkpoints are not
included. `docs/source_inventory.json` maps the full source snapshot and all
integration changes.

`upstream/CacheGS/` preserves the GDMGS_Codebase source snapshot read from
zxcpu2, including its inherited research-use license. `system/batch.py`,
`system/raster_backend.py`, and `system/full_bundle_cache.py` are source-derived
transport components. Original research implementations under `research/`
retain the source-family licenses. fVDB is an external dependency; its observed
source commit and local build patch are recorded under `dependencies/`.
