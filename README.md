# MINT

## What is included

- Fast Search, exact rank cache, original optional AVX2 scoring accelerator,
  ProteinMPNN implementation and checkpoint, and the 25-allele panel.
- AF3 request adapter, persistent GPU worker, model runner, output writer, and
  the compatible image's build recipe. The exact external engine revision and
  setup instructions are recorded under `runtime/af3/`.
- The enriched LuxSit `2.pdb` example and its 43 fixed residues.
- Gradio controls, command-line setup, Slurm launcher, tests, and documentation.

NetMHCpan's licensed installation/data, AF3 model parameters, prebuilt container
images, the upstream AF3 engine, and historical run caches are **external**.
Obtain access under their respective terms. This backend requires Linux x86-64,
Apptainer, and an NVIDIA GPU. See
[AF3 setup and compatibility](docs/AF3.md) and [third-party notices](THIRD_PARTY_NOTICES.txt).

## Start here

| Goal | Read or run |
| --- | --- |
| Understand the algorithm and find the code | [Workflow and code map](docs/WORKFLOW.md) |
| Configure another computer | [Installation and deployment](docs/INSTALL.md) |
| Understand AF3 inputs, outputs, and runtime | [AF3 guide](docs/AF3.md) |
| Run checks and understand validation limits | [Testing](docs/TESTING.md) |
| Review exactly what changed in this port | [Porting record](docs/PORTING.json) |
| Inspect historical source identities | [Source provenance](SOURCE_PROVENANCE.json) |

Commands below run **from this folder on the execution host**. Digs preparation
and submission are lightweight login-node operations; model execution and
scientific tests belong on compute nodes.

## Run the LuxSit example on Digs

Copy this folder to your own account. Check your account can read the shared
dependencies in `configs/digs.example.json`, then make a local configuration:

```bash
cp configs/digs.example.json site.local.json
# Edit resource/account settings and paths if needed.
python3 lf.py doctor --site-config site.local.json

# Prepare only: no job is launched.
python3 prepare.py --site-config site.local.json \
  --out /net/scratch/"$USER"/luxsit_fastsearch_af3

# Inspect config.json, then preview and submit one GPU job.
python3 lf.py submit --run /net/scratch/"$USER"/luxsit_fastsearch_af3 --dry-run
python3 lf.py submit --run /net/scratch/"$USER"/luxsit_fastsearch_af3
```

`prepare.py` defaults to the bundled LuxSit seed, deimmunization, and its 43
fixed positions. Add `--direction immunize` for the opposite direction and
choose a different output directory. A custom `--pdb` gets **no automatic
LuxSit fixed positions**; use `--chain`, `--fixed`, `--fixed-file`, or
`--fixed-indices` explicitly.

```bash
python3 prepare.py --site-config site.local.json \
  --pdb /path/to/protein.pdb --chain B --fixed 'B:40,B:45A' \
  --direction deimmunize --seed 123 --out /path/to/new-run
```

For a small real-model integration run, add `--smoke`. This still performs real
scoring and AF3 folding. A first AF3 batch includes model loading and compilation.
It is not an instant UI demo.

## Gradio interface

Install `requirements-ui.txt` in an environment on the execution host, then:

```bash
python app.py
```

Open [localhost:7861](http://127.0.0.1:7861). The interface offers preparation,
configuration review, explicit Slurm submission/resumption, and progress
inspection. **Prepare run** does not submit. The UI uses the same `prepare.py`
and `lf.py` entry points as the commands above.

When it runs remotely, forward port 7861 over SSH and open it in your local
browser; see [deployment instructions](docs/INSTALL.md). Dependency checks run
where the app runs. Opening the UI on your Mac does not give it access to Digs
files or GPU resources. It binds only to localhost and does not create a public
Gradio share link. Port 7861 leaves the existing CPU app's port 7860 available.

## Run on a Linux GPU workstation without Slurm

Configure `configs/linux.example.json` as described in [INSTALL.md](docs/INSTALL.md),
prepare a run with that configuration, then launch on the selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash job.sh /absolute/path/to/new-run
```

This invokes the same original daemon and `run120.py`. Only one process should
run a given directory at a time. The current launcher accepts one numeric GPU
index; GPU UUIDs and MIG UUIDs are not supported. Do not run this command on a
cluster login host.

## Preserved campaign settings

`prepare.py` retains the original 120-round campaign budget and population
settings, with the revised weighted objective and joint temperature 0.4. It prepares **one run**, with scratch deletion disabled.
It does not recreate the original forty-lineage scheduler.

| Setting | Campaign preset |
| --- | --- |
| Initial MPNN draws / folding shortlist | 512 / 96 slots, plus root if absent; fallback folds if none pass |
| Initial and immigrant MPNN temperatures | 0.4 |
| Joint-proposal temperature | 0.4 for all eight 2/4/8-site attempts |
| Objective | Sum of cumulative EL rank counts at 0.25%, 0.50%, …, 2.00% |
| Search rounds / population size | Up to 120 / 24 |
| Parents / child folds per round | Up to 8 / 48 |
| Per-allocation search budget | 42 hours, then checkpoint/pause |
| AF3 | Empty MSAs, no templates, 5 recycles, 1 diffusion sample |
| Fold acceptance | Mean CA pLDDT ≥90 and root-aligned CA RMSD ≤2 Å |
| Default random seed | 26051011; configurable per run |
| Scoring accelerator | `required`; use `--native-mode off` for stock scoring |

The original campaign's individual lineages had their own seeds and inputs;
this single-run example is not a claim to reproduce an entire lineage cohort.
Use the original seeds and environment when reproducing a particular run.

The lower-level `lf.py init` remains available with an **80-round default budget**:
80 rounds, weighted scoring, and temperature 0.4. Its `search_hours`
field is not the stopping budget used by the `run120.py` launcher;
`allocation_search_hours` is. Prefer `prepare.py` for the campaign preset.
The current scientific policy and historical distinctions are described in
[WORKFLOW.md](docs/WORKFLOW.md).

## Progress, results, and restart

Every run owns its files; a new run starts with an empty predictor cache.

| File | Purpose |
| --- | --- |
| `config.json`, `input.pdb`, `residue_mapping.json`, `fixed_residues.txt` | Captured settings, normalized input, and original residue IDs |
| `live.json`, `slurm-*.log`, `af3_daemon.log` | Progress and runtime logs |
| `state.json.gz`, `cache.sqlite` | Last committed search state and exact rank cache |
| `pending_folds.json.gz`, `resume_accounting.json`, `paused.json` | Partial-fold reuse and allocation-boundary restart |
| `round_*/results.json.gz`, `summary.json`, `progress.csv` | Candidate and progress records |
| `champion.pdb`, `champion.json`, `archive.fasta` | Best passing structure and retained sequence archive |
| `confirmation.json.gz`, `complete.json`, `failure.json` | Final confirmation, completion, or error details |

Resume an interrupted submitted run with:

```bash
python3 lf.py submit --run /absolute/path/to/run --resume
```

Keep the package and normalized input unchanged for a prepared run: source
fingerprints are checked. This scoring revision has new source fingerprints;
old prepared runs must continue using their original package. Never overwrite
an existing run to start again. The archive/cleanup utility is opt-in and is
disabled by `prepare.py`.

## Check and share the package

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python tools/release.py --check
python tools/release.py --output dist/FastSearchAF3-source.zip
```

The release tool includes only the explicit source layout and rejects symlinks;
it excludes run data, private installations, environments, and generated
archives. The ZIP contains a top-level `FastSearchAF3/` directory and a checksum
manifest. A `.sha256` file accompanies it. Read the third-party notices before
redistributing components beyond the intended project.

Local tests exercise policy, input handling, checkpoints, setup, and mocked AF3
transport. **No fresh GPU fold was performed to package this source.** Historical
`VALIDATION.json` is preserved upstream evidence, not a new GPU validation.
