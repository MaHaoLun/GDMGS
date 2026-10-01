# GDM-GS stages A and B

This is an isolated extension of the existing CacheGS source. The default
entrypoints retain the legacy renderer and cache. `train.py` and the checkpoint
format remain unchanged. The GDM-GS option currently runs fresh FoV selection,
explicit anchor-ID decoding and rasterization. Mesh/BVH/ORI, the new cache and
lookahead scheduling are later stages.

## Runtime and outputs

On zxcpu2 use the existing `/ssddata/lun/miniconda3/envs/render` environment.
The verified import baseline is Python 3.10.14, Torch 2.4.0+cu124 and
gsplat 1.4.0+pt24cu124 with the installed fVDB build. No new mesh dependency is
required. `environment.yml` and `req.txt` retain the original environment inputs;
the validation report records the actual environment used.

Set `CUDA_VISIBLE_DEVICES` before launching Python. `--device cuda:0` refers to
the first device in that visible set; the entrypoints no longer select another
physical GPU during import. Use an output root outside the checkpoint directory
for every project run. Omitting it intentionally retains the original CLI output
location for compatibility.

```sh
CUDA_VISIBLE_DEVICES=1 CACHE_ENABLE=0 PRECOMP_INDICES_PATH= \
  /ssddata/lun/miniconda3/envs/render/bin/python render.py \
  -m /path/to/checkpoint-run --iteration 40000 \
  --output-root /ssddata/lun/gdmgs_artifacts/scene/iteration_40000/legacy

CUDA_VISIBLE_DEVICES=1 CACHE_ENABLE=0 PRECOMP_INDICES_PATH= \
  /ssddata/lun/miniconda3/envs/render/bin/python render.py \
  -m /path/to/checkpoint-run --iteration 40000 --pipeline gdmgs \
  --output-root /ssddata/lun/gdmgs_artifacts/scene/iteration_40000/gdmgs_fresh
```

`render2.py` keeps separate train/test modes and their skip flags.
`precompute_cache.py --output-root ...` writes the original precompute format
outside the checkpoint. `metrics.py -m <render-output-root>` reads the rendered
images; its optional `--output-root` redirects metric result files.

## Explicit selections and ownership

Create `gdmgs.pipeline.FreshPipeline` from a dedicated, loaded evaluation model
after its coarse intervals have been initialized. Its `render(...,
selected_ids=ids)` accepts unique, one-dimensional integer IDs in exact PLY row
space, preserving the supplied order. An explicit selection bypasses FoV,
precompute and cache selection. Without one, pure pose-local FoV produces the
selection. Conflicting legacy cache/precompute settings raise an error.

The optional `BundleMetadata` records each request (including zero-row results),
row owner, original offset slot, count and prefix offsets. The legacy Gaussian
tuple remains seven items. Geometry tensors and ownership use the same opacity
filter. Raster inputs must have matching row lengths.

Sessions bind the current loaded rows and decoder parameters without changing
training `requires_grad` flags. Parameter updates, row replacement, LoD setting
changes or closing the session invalidate reuse. This is an in-memory inference
contract: do not mutate tensor storage through `.data`, foreign pointers or
concurrent training. Open a new session after replacing a model. Camera and
materialization identity are explicit fields and session tokens; no content
hashes or checksum passes are used.

## Validation scope

The A/B report distinguishes all 13 actual checkpoint loads from bounded
fixed-frame compatibility comparisons. Temporary training fixtures test forward,
backward, statistics, optimizer updates and saving/loading independently of the
read-only pretrained checkpoints. Cross-pose bundle reuse is a quality precheck;
it does not claim the later cache quality criterion has passed.

Run the unit suite in the rendering environment with `python -m pytest`.
Real-data tools under `tools/gdmgs/` expose their arguments with `--help`.
The accepted architecture and owner plan are in the workspace's
`docs/adr/0007-gdmgs-ab-isolation-and-fresh-interface.md` and
`docs/gdmgs_ab_implementation_plan.md`.
