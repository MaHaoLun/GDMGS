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
