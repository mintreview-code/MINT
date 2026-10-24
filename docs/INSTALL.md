# Installation and deployment

## Machines and environments

Full inference uses Linux x86-64, Bash 4+, host `/usr/bin/python3` version 3.10+
for the AF3 daemon, Apptainer with NVIDIA passthrough, and one compatible NVIDIA
GPU. ProteinMPNN runs on CPU; AF3 runs on GPU. The default allocation is 8 CPUs,
30 GiB RAM and one GPU, but larger proteins may require more GPU memory.

There are three separate Python environments:

1. **Control host**: `lf.py`, `prepare.py`, and the AF3 socket daemon. Setup uses
   the standard library. Gradio additionally needs `requirements-ui.txt`.
2. **CPU inference**: the `cpu_python` environment or configured CPU container.
   Install `requirements.txt` for a native Linux environment.
3. **AF3 inference**: the compatible AF3 Apptainer image supplies the pinned
   engine, JAX/CUDA, compiled extensions, and dependencies. It uses the bundled
   `vendor/af3/` adapter. Installing the CPU requirements alone does not install AF3.

The upstream AF3 engine source is not bundled. The recipe under `runtime/af3/`
fetches its exact pinned revision when building the image. The worker imports
the installed engine inside that image and uses the included `vendor/af3/` adapter.

The current Mac can read the source, run unit tests, and display the Gradio
interface. Full execution is not supported natively on macOS or an emulated
Linux container on Apple Silicon. The earlier CPU-only Mac application remains
in the parent folder and retains its native MPNN + Docker NetMHCpan mode.

## Digs

Copy the package to your account and copy `configs/digs.example.json` to
`site.local.json`. The shared paths are the original known dependency locations:

| Configuration | Meaning |
| --- | --- |
| `cpu_container` | Shared CPU image; launcher expects `python3.9` inside it |
| `cpu_python` | Optional absolute path to a CPU Python environment; overrides the CPU image |
| `netmhcpan` | Installation root containing `Linux_x86_64/bin/netMHCpan` and its trained data |
| `af3_image` | Compatible AF3 Apptainer `.sif` image |
| `af3_model_dir` | Directory containing the authorized AF3 model parameters |
| `partition`, `gres`, `constraint`, `account` | Your site's Slurm scheduling settings |
| `cpus`, `memory`, `time_limit` | Requested resources; defaults 8, 30G, 48:00:00 |

Run `python3 lf.py doctor --site-config site.local.json` to check paths are
readable. This check does **not** validate CUDA, imports, model weights, or a
successful fold. The exact image recipe/source pin is documented in `AF3.md`.

Prepare with `prepare.py`, inspect the saved `config.json`, then submit with
`lf.py submit`. That single job runs both CPU search and GPU folding. No code
depends on the original author's home directory. The historical campaign's
personal CPU sandbox was a deployment choice; supply a working environment
accessible to your account.

## Another Linux machine

Create a CPU environment using your site's Python installation:

```bash
python3 -m venv /absolute/path/to/cpu-env
/absolute/path/to/cpu-env/bin/python -m pip install -r requirements.txt
cp configs/linux.example.json site.local.json
```

Edit every placeholder path. For a CPU-only PyTorch build, use the PyTorch
CPU wheel index appropriate to your environment. The requirements are compatible
ranges inherited from the project, not a newly validated lockfile. Exact
stochastic results also depend on library versions and hardware. Preserve your
environment lock or container identity with results.

Install NetMHCpan 4.1 through its authorized distribution and complete its data
installation. The project invokes the Linux binary from the expected directory
layout; a downloaded launcher alone is insufficient. Do not place its licensed
archive or trained data inside a source release.

Use the matching AF3 runtime recipe and authorized parameters described in
[AF3.md](AF3.md). A generic stock AF3 image is not an interchangeable dependency
for this adapter: the pinned fork has additional APIs and Tokamax dependencies.

After `doctor` and run preparation, either submit to your Slurm scheduler or
run directly on a workstation:

```bash
CUDA_VISIBLE_DEVICES=0 bash job.sh /absolute/path/to/prepared-run
```

The current launcher requires a numeric GPU ID and host `/usr/bin/python3`.
It starts one persistent AF3 worker, caps numerical-library threads, and removes
its own temporary directory at exit. JAX compilation cache is temporary per
allocation, so the first fold in each allocation may compile again. Do not
set `LF_EXPERIMENT_BASE`; it is a retained optional hook for an external campaign
monitor, not needed by this package.

The optional native NetMHCpan hook needs a supported Linux x86-64 AVX2 CPU and
the exact supported predictor binary. `prepare.py` retains the campaign's
`--native-mode required` default. `--native-mode off` uses the original stock
predictor; `auto` attempts the hook when supported and otherwise falls back.
The Mac's earlier compatibility checks did not validate that hook under
emulation, so do not infer compatibility from the presence of a `.so` file.

If a kernel rebuild is needed, run `native/build.sh` on an appropriate Linux
compute node before preparing runs, because the binary is fingerprinted.

## Gradio locally or through SSH

Install `requirements-ui.txt` in the Python environment used for the UI and
run `python app.py --port 7861`. The app only orchestrates preparation, explicit
Slurm submission, and status reads; it does not execute model inference on a
login node.

For an app hosted on Digs, keep it bound to localhost and forward the port from
your Mac using your own account:

```bash
ssh -L 7861:127.0.0.1:7861 YOUR_ACCOUNT@digs.ipd.uw.edu
```

Then browse to `http://127.0.0.1:7861`. If the app is on a compute node, use your
site's permitted tunnel to that node instead. You must run the app where the
configured files and scheduler commands are accessible. It is a local trusted
user tool, not a multi-user hosted service.

## Move or share

Use `tools/release.py` to make the source ZIP. The destination account still
needs its own dependency access and edited site configuration. Keep run data
separate. Formatting or editing files after preparation invalidates protected
source checks; initialize fresh runs after code changes.

The bundled test suite can verify policy consistency without model access.
Run the real CPU smoke test and then a small real GPU smoke run in the target
environment before treating that environment as validated. See [TESTING.md](TESTING.md).
