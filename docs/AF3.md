# AF3 folding: code, execution, and outputs

This package contains the folding code used by the original LF campaign. It
automatically folds shortlisted sequences and admits passing structures to the
next search population. It is a separate workflow from the CPU-only Gradio app,
which stops at an external-folding queue.

The existing AF3 adapter and worker are included here, together with the build
recipe extracted from the original container. The upstream `alphafold3` engine
is an external dependency: the recipe downloads its exact compatible revision
and builds the runtime environment. The configured AF3 container supplies that
engine, compiled extensions, and CUDA/JAX environment at runtime; parameters
come from a separate authorized installation. Copying this folder alone does
not build those dependencies or make AF3 run on a Mac.

## Where the code lives

| File | Responsibility |
| --- | --- |
| `models.py`, `AF3` | Converts selected sequences into AF3 manifests, sends requests, and checks returned structures. |
| `models.py`, `parse_fold` | Validates sequence/residue identity and applies the original confidence and geometry gates. |
| `run120.py`, `Experiment120.fold` | Sends four designs per batch and journals completed batches for restart. |
| `vendor/af3/af3_daemon.py` | Owns a local Unix socket and one persistent GPU worker. |
| `vendor/af3/af3_apptainer.py` | Constructs the Apptainer invocation and bind mounts. |
| `vendor/af3/run_alphafold_batch.py` | Loads parameters once, prepares inputs, and runs each manifest. |
| `vendor/af3/af3_runtime.py` | Adapts the engine's model runner, CIF inputs, and output files. |
| `runtime/af3/` | Exact inspected image-build recipe, labels, verified external engine revision, and build instructions. |
| `job.sh` | Starts the GPU worker and CPU search, arranges temporary directories, and performs cleanup. |
| `site.example.json` | Supplies machine-specific installation paths and Slurm resources. |

The worker code is copied from RFProteina `v0.0.2`. Source hashes and original
locations are recorded in `SOURCE_PROVENANCE.json`; no live import from that
repository is required. The external engine is the `kierandidi/alphafold3` fork
at commit `8d3601151d5bf7525d8591583915d2a8471442a5`, identified by the actual
container's embedded recipe and labels. Its separate provenance is in
`runtime/af3/PROVENANCE.json`.

## Exact folding protocol

- The selected input chain is normalized to **chain A, residues 1..L**. This
  campaign folds one protein chain; other chains and ligands are not context.
- Each candidate receives its proposed sequence on the parent backbone as a
  temporary PDB, then a complete polymer mmCIF description. That CIF is used to
  construct the AF3 input. The parent coordinates are **not supplied as an AF3
  template**: the adapter explicitly sets an empty template list.
- Paired and unpaired MSAs are empty strings. This is the existing single-sequence
  folding protocol; no sequence database search or template search is run.
- Every prediction uses **five recycles and one diffusion sample**, with a seed
  recorded in the fold result. Batch seeds derive from the run seed, round, and
  position in the shortlist. The existing launcher selects `cudnn` attention;
  the runtime pins Tokamax GLU and layer-normalization operations to XLA.
- A result must contain exactly one model, one chain A, the expected residue
  numbering, the exact proposed sequence, and unchanged fixed residues.
- The gate is **mean CA pLDDT >= 90** and **CA RMSD <= 2 Å** after rigid alignment
  to the original normalized input structure. The RMSD reference is the root
  structure, not the latest parent. Confidence values must be finite and within
  0..100.
- Only passing structures can become search parents. The campaign performs two
  additional independent folds of the final champion and records both outcomes.
  It does not silently replace the champion when a repeat fails.

These are the original protocol choices. Changing samples, recycles, inputs,
gates, model family, or parameters changes the experiment; they are not mere
installation settings.

## What happens to files

1. `job.sh` creates a private `/tmp/lf-portable.*` directory. It contains the
   daemon socket, model compilation cache, and per-call folding files. The CPU
   process and AF3 container see the same absolute paths through bind mounts.
2. `AF3.fold` writes a temporary `fold_*/manifest.json`. Each manifest item has a
   name, inline CIF, output directory, seed, and explicit empty MSAs.
3. The CPU process sends `{"cmd":"fold","manifest":"...","save_full_outputs":false}`
   to the local Unix socket. The daemon passes the manifest path to the resident
   worker. Requests execute serially on one GPU; input preparation for the next
   design overlaps the current GPU prediction.
4. The worker writes `<name>_model.cif` and
   `<name>_summary_confidences.json`. Full confidence arrays are temporarily
   written to compute summaries, then removed. This launcher does not request
   retained per-sample output directories.
5. The CPU adapter reads and validates those files, records metrics and the
   summary confidence, and keeps a PDB string for passing results. The temporary
   fold directory is removed after the call, including failed structures.
6. Completed four-design batches are saved in `pending_folds.json.gz`; completed
   rounds go into `initial.json.gz` or `round_*/results.json.gz` and the population
   checkpoint `state.json.gz`. `live.json` reports batch progress.
   `af3_daemon.log` contains model-loading, compilation, and folding diagnostics.
   `champion.pdb`, `champion.json`, and `confirmation.json.gz` are final outputs.

The worker remains warm within a job. The compilation cache is inside the job's
temporary directory and is removed at cleanup, so a new allocation can pay
model-loading and compilation costs again. A socket existing, or a successful
`ping`, means the daemon is listening; it does not prove the model is loaded.

## Dependencies and setup

The existing execution path requires Linux, a supported NVIDIA GPU and driver,
Apptainer with GPU passthrough, a compatible AF3 image, and authorized access to
the selected parameters. The host Python used by the daemon must support Python
3.10 syntax; the CPU search can use the configured CPU container or a separate
environment with `requirements.txt` installed.

The checked-in Digs configuration points to:

| Setting | Existing location |
| --- | --- |
| `af3_image` | `/projects/ml/rfd4-proteina/latent_data/af3/apptainer_images/af3-open-2026-08-19.sif` |
| `af3_model_dir` | `/net/databases/alphafold/` |
| `cpu_container` | `/net/software/containers/universal.sif` |
| `netmhcpan` | `/net/software/netmhcpan/netMHCpan-4.1` |

These are site examples, not bundled files. Check access from your own account.
`af3_model_dir` means the **parameter directory**, despite the site's
`/net/databases/` prefix. This folding protocol does not use the large sequence
or template-search databases. The container must still supply the engine's
chemical-component resources and compiled extensions.

Copy `site.example.json` to a personal site file, edit paths and resources, then
pass `--site-config your_site.json` to `lf.py doctor` and `lf.py init` as shown in
the [main README](../README.md). `doctor` checks dependency paths; it is not a GPU
inference test. `lf.py submit` uses Slurm. Initialization itself submits nothing.

On another Linux GPU workstation, supply equivalent paths and a compatible
image. The existing batch script can run directly with
`CUDA_VISIBLE_DEVICES=0 bash job.sh /absolute/prepared/run`, after preparing the
run against those local paths. It still launches AF3 through Apptainer. The
current script requires one numeric GPU identifier; it does not accept a
comma-separated device list or a GPU UUID. Direct Linux execution has not been
GPU-validated as part of this local packaging step.

A generic AF3 image is not guaranteed to work: the batch wrapper uses
fork-specific featurisation arguments (`fix_standalone_glycans`, `of3_weights`),
and the runtime imports Tokamax, including private operation settings. Preserve
the compatible image for reproduction or build the supplied exact recipe using
the [runtime setup guide](../runtime/af3/README.md). Use the recipe under
`runtime/af3/`, not the older recipe in the upstream source at that revision.
The adapter header's `sokrypton/alphafold3` commit
`d43d4971e68a2b88551288a670a397be427067ce` describes the adapter's origin, not the
engine revision in this dated container. A rebuilt image still needs a GPU
validation run; no rebuild or GPU inference was performed for this local port.

The wrapper also contains an OpenFold3 parameter-detection branch. It is not
selected by the LF site example and is not a license to substitute different
weights when reproducing the campaign.

## Mac and sharing boundaries

This folder can be stored, read, and edited locally on macOS. The original AF3
execution path requests the CUDA JAX backend and uses `apptainer --nv`; there is
no Apple GPU, Metal, or CPU AF3 path in this package. Use a compatible Linux GPU
machine for the complete campaign. The existing CPU-only Mac Gradio app remains
a separate application.

Retain `THIRD_PARTY_NOTICES.txt`, `vendor/af3/LICENSE`, and file headers when
sharing the source. The RFProteina wrappers carry their retained BSD notice;
`af3_runtime.py` retains its Apache-2.0 header and upstream attribution. The
external engine has its own Apache-2.0 license and separate parameter/output
terms, obtained with the engine when the runtime is built. The upstream engine
source, NetMHCpan installation, AF3 parameters, and container images are not
redistributed here. No broad license is asserted for the original project code
or bundled MPNN checkpoint; their original terms still apply.
