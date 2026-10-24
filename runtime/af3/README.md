# Build the AF3 runtime used by this campaign

The existing AF3 adapters are included under `vendor/af3`; the upstream engine
is installed separately. The compatible engine is [`kierandidi/alphafold3` commit
`8d3601151d5bf7525d8591583915d2a8471442a5`](https://github.com/kierandidi/alphafold3/commit/8d3601151d5bf7525d8591583915d2a8471442a5).
The build recipe downloads that exact source revision, including its Python
model, featurisation, compiled-extension source, and dependency lockfile, then
installs the runtime. This package contains the recipe and setup documentation,
not the upstream engine source, model parameters, or a prebuilt environment.

Use **`runtime/af3/af3-open-2026-08-19.def`** for this campaign. That file was
extracted, unchanged, from the actual configured Digs image using read-only
`apptainer inspect --deffile`. Its labels were extracted with
`apptainer inspect --labels`. The upstream source at that revision has its own
`apptainer/af3-open.def` pointing to an **older engine commit**; use the inspected
recipe supplied here to reproduce the selected image.

## Build and configure

On a Linux x86-64 machine with Apptainer and permission to build images, run from
the package root:

```bash
apptainer build /absolute/path/af3-open.sif \
  runtime/af3/af3-open-2026-08-19.def
```

On Digs, image builds belong on an allocated compute node, not the login host.
This is a substantial software build requiring network access and working disk
space. The recipe downloads the engine at the pinned commit, system packages,
Python dependencies, native-library sources, and chemical-component resources.
It does not download model parameters.

The recipe declares:

| Component | Version or source |
| --- | --- |
| Base image | `nvidia/cuda:12.6.3-base-ubuntu24.04` |
| Engine | `kierandidi/alphafold3`, commit `8d3601151d5bf7525d8591583915d2a8471442a5` |
| Python | 3.12 |
| Package installer | `uv` 0.9.24, using `uv sync --frozen --no-editable` |
| JAX | 0.10.2, CUDA 12 |
| Haiku | 0.0.16 |
| RDKit | 2025.9.4 |
| Tokamax | 0.0.12 |
| PyTorch | CPU 2.13.0 for the optional upstream parameter-conversion utility |

Set `af3_image` in your site JSON to the built `.sif`, and set `af3_model_dir`
to your authorized AF3 parameter directory. The CPU environment and NetMHCpan
paths remain separate settings. Follow [the campaign setup](../../README.md)
to prepare and launch a run. The search adapter uses the AF3 parameters selected
in that directory; the existence of OpenFold3 support in the engine does not
authorize substituting a different model family in this experiment.

## Provenance and validation limits

[`PROVENANCE.json`](PROVENANCE.json) records the inspected shared image path,
size, build labels, exact Git commit/tree, and the inspected upstream
source-archive SHA-256. The matching upstream source was reviewed to check the
required APIs and dependency versions, but is not included in this package.
The engine source and recipe are Apache-2.0; the engine's original licenses and
separate parameter/output terms are obtained with the source during the build.

The observed image was 3,909,484,544 bytes. Its complete SHA-256 was not computed,
and the image itself is not included. An older adjacent
`artifacts/af3-open.manifest.json` refers to another image and was not used as
the identity of this dated runtime.

The source and embedded recipe were inspected without launching the image.
This local packaging step did **not** rebuild the image or run GPU inference.
The recipe pins the engine and Python dependency lockfile, but retains a base
image tag and live system-package/data downloads. Rebuilding is therefore not
a claim of bit-for-bit identity with the existing image. Validate an actual
GPU prediction on the target host before relying on a rebuilt runtime.

For the exact single-sequence protocol, acceptance gates, and output locations,
see [the folding guide](../../docs/AF3.md).
