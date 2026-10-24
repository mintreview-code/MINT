# Portable LF immunization / deimmunization

This folder packages the algorithm used by `lf_deimmunize` and `lf_immunize`.
It includes the LF search, exact peptide cache and scoring logic, ProteinMPNN
code and checkpoint, accelerated native kernel, and the persistent AF3 adapter.
It does not depend on the original MHC_project tree or its author's account.

Copy the whole folder to your own DIGS account. The large CPU/AF3 container
images, AF3 parameters, and NetMHCpan installation remain shared dependencies.
Their locations are configurable in `site.example.json`; no large databases
or historical prediction caches are included.

The distributable `lf_portable.tar` contains the package, example input and
checkpoint, without development validation directories or run data. Verify it
with `sha256sum -c lf_portable.tar.sha256`, then extract it with
`tar -xf lf_portable.tar` in its destination directory. `PACKAGE_MANIFEST.json`
records checksums of the distributed files. The archive is uncompressed and
small; no cluster container images or AF3 parameters are embedded.

## Quick start on DIGS

Run these commands from this folder using the host's Python 3. Preparing a run
does not submit it. The explicit `submit` command submits one GPU job.

```bash
python3 lf.py doctor

python3 lf.py init \
  --pdb /path/to/your_protein.pdb \
  --chain A \
  --fixed 'A:10,A:25,A:42A' \
  --direction deimmunize \
  --out /net/scratch/"$USER"/my_lf_deimmunize

python3 lf.py submit --run /net/scratch/"$USER"/my_lf_deimmunize --dry-run
python3 lf.py submit --run /net/scratch/"$USER"/my_lf_deimmunize
```

Use residue identifiers that actually exist in your PDB. Omit `--fixed` when
no positions should be frozen. **There is no automatic LuxSit fixed-residue
list for arbitrary input proteins.** For the opposite direction, use
`--direction immunize` with a different output directory.

`launch` accepts the same arguments as `init` and immediately submits the
prepared run. `init` followed by `submit` is useful for inspecting the config.

To reproduce the original LF starting structure and 43 fixed positions:

```bash
python3 lf.py init \
  --pdb examples/luxsit_seed.pdb \
  --fixed-file examples/luxsit_fixed.txt \
  --direction deimmunize \
  --out /net/scratch/"$USER"/luxsit_lf_deimmunize
python3 lf.py submit --run /net/scratch/"$USER"/luxsit_lf_deimmunize
```

The example PDB is the enriched seed `2.pdb` used by the LF runs, rather than
the different original seed used by some other project benchmarks.

## Input structures and fixed residues

The search accepts a **complete, contiguous canonical protein chain of
arbitrary supported length**, rather than being restricted to LuxSit's 117
residues. Length is ultimately limited by AF3 and the selected GPU's memory.
The input parser requires at least 11 residues so all three peptide lengths
can be evaluated.

For a PDB with multiple chains, choose one using `--chain`. This is a monomer
workflow: the selected chain is scored, redesigned and folded independently;
other chains and ligands are not folding context. Multi-chain co-design,
ligand-aware design, missing-backbone reconstruction and noncanonical residue
design are not implemented.

The selected chain is normalized internally to chain A, residues 1..L.
`residue_mapping.json` maps each internal index back to its original chain,
residue number and insertion code. Output structures use the internal numbering.
Input and mapping files are copied into the run directory, so the original
PDB need not remain available after preparation.

Supported ways to freeze positions:

* `--fixed 'B:40,B:45A'`: original PDB identifiers, including insertion codes.
* `--fixed-file fixed.txt`: the same identifiers separated by whitespace or
  commas; `#` comments are supported in this file. Legacy `A40` identifiers
  also work.
* `--fixed-indices '1,6,9-10'`: one-based indices within the selected chain.

Missing backbone atoms, unknown fixed identifiers, and physical chain breaks
are rejected instead of silently constructing peptide windows across gaps.
The first PDB model is used. Alternate locations are resolved deterministically,
preferring blank then A conformers. Waters/ligands are not included.

## Shared software and another account

Default locations are in `site.example.json`. They point to shared cluster
resources, not a personal home directory:

| Dependency | Default |
|---|---|
| CPU Python libraries | `/net/software/containers/universal.sif` |
| NetMHCpan 4.1 | `/net/software/netmhcpan/netMHCpan-4.1` |
| AF3 image | Shared dated RFProteina AF3 image under `/projects/ml/rfd4-proteina/latent_data/af3/apptainer_images/` |
| AF3 parameters | `/net/databases/alphafold/` |
| MPNN checkpoint | Bundled `assets/protein_mpnn.pt` |

You need access to those installations under your own account. To use other
locations or resource settings, copy the configuration and pass it at init:

```bash
cp site.example.json my_site.json
# Edit paths, partition, account, constraint, CPUs, memory, or time_limit.
python3 lf.py doctor --site-config my_site.json
python3 lf.py init --site-config my_site.json \
  --pdb /path/input.pdb --direction immunize \
  --out /net/scratch/"$USER"/my_lf_immunize
```

The site config is captured in the run's `config.json`. An optional
`cpu_python` path selects your own Python environment instead of the CPU
container; its packages are listed in `requirements.txt`. The folded-model
runtime still uses the configured AF3 container.

The vendored AF3 adapter uses the same warm-worker protocol, five recycles,
one diffusion sample, and empty MSAs as the original LF experiment. The
wrapper is included; the RFD4 repository is neither required nor modified.

The accelerated NetMHCpan hook is restricted to its verified native-engine
checksum and AVX2 CPUs. `--native-mode auto` falls back to the stock forward
pass when the hook is unavailable; `required` fails rather than falling back;
`off` always uses stock prediction. Scoring still requires the supported
NetMHCpan 4.1 installation/layout. Cache provenance includes predictor assets
and whether the hook was used.

## Search behavior retained from LF

Default settings are the original LF settings:

1. Generate 512 MPNN founders at temperatures 0.4, 0.7 and 1.0. Score and
   shortlist up to 96 for folding, also considering the supplied seed.
2. Search the champion plus rotating parents, up to eight parents per round.
3. Enumerate single substitutions at peptide-map hotspots; propose up to
   eight MPNN-supported position pairs with three alternatives per site;
   add eight joint MPNN proposals targeting 2, 4 or 8 hotspot positions.
4. Optimize the number of peptide–allele occurrences with EL rank <2% directly.
   Deimmunization minimizes it; immunization maximizes it. The sum of the
   cumulative <0.25% and <0.5% counts is the tie-breaker, not a separate
   acceptance constraint. That sum counts <0.25% occurrences twice.
5. Permit parent-relative worsening of `max(2, ceil(0.02 * parent_count))` for
   ordinary exploratory children. Every fifth round adds 16 fresh MPNN samples
   from the original structure; these immigrants use the separate exploratory
   route without that parent-relative screening requirement.
6. Fold up to 48 children, prioritizing up to 36 non-worsening designs and
   selecting diverse extras. Require pLDDT >=90 and root-aligned CA RMSD <=2 A.
7. Update the elite archive and diverse population, each up to 24 sequences.
   Diversity is Hamming distance, not structural or peptide-set distance.

The existing rotating-parent and selection policies are preserved, including
their details: the parent offset advances by eight while selecting seven
alternatives, so a static list is not guaranteed complete round-robin coverage;
the population score-slack restriction applies to its diversity-filling phase.

Caching, within-request deduplication, affected-window scoring, ordered allele
waves and exact early rejection are retained. All counts represent complete
9/10/11-mer scans across the selected allele panel even when unchanged windows
reuse parent contributions. Partial early-rejection counts are not final counts.

This package is the LF population search, **not the separate 200-candidate
ablation benchmark and not original kappa whole-sequence guidance**.

## Budgets, restarts, and files

Default resources: one GPU, eight CPUs, 30 GiB, 48-hour Slurm limit. The
search stops after 80 rounds, 44 elapsed hours, or the objective endpoint,
whichever occurs first. It then audits the champion with the stock predictor
and performs two confirmation folds. Larger proteins may need a larger GPU,
more memory, and different budgets configured in the site file.

CLI budget options include `--rounds`, `--search-hours`, `--initial-designs`,
`--initial-folds`, `--folds-per-round`, `--beam`, and `--seed`. A JSON allele
list can replace the bundled 25-allele panel via `--alleles-file`.

Each new run starts with an empty prediction cache. An interrupted run can
resume its own checkpoint with:

```bash
python3 lf.py submit --run /net/scratch/"$USER"/my_lf_deimmunize --resume
```

Resume retains the original cumulative budget; it does not silently restart
80 rounds or 44 hours. An already-running or completed job is not resubmitted.
Keep package code and the normalized input unchanged after preparing a run;
their checksums are verified. Preparing a new run is preferable after editing
the package. Prepared runs can be relocated alongside the unchanged package;
regenerate the submission command from their new location.

Main outputs:

* `config.json`, `residue_mapping.json`, `fixed_residues.txt`: reproducibility.
* `live.json`, `state.json.gz`: current progress and restart state.
* `round_*/results.json.gz`, `summary.json`: candidates, folding outcomes, metrics.
* `cache.sqlite`, `provenance.json`: this run's ranks and predictor identity.
* `champion.pdb`, `champion.json`, `archive.fasta`: final best design and archive.
* `progress.csv`, `below_2.png`, `strong_sum.png`: measured progress.
* `confirmation.json.gz`, `complete.json`: final audit and repeated-fold outcomes.

Failed folding structures and per-call intermediates are temporary. Passing
structures are bundled in state records, with the final champion exported.
The CPU metric is consumed design user+system time; AF3 and final validation
are excluded. Wall time and allocated CPU-hours are different quantities.

## Validation and small tests

`VALIDATION.json` records tests actually performed while packaging. Unit tests
cover input normalization, fixed positions, arbitrary lengths, exact delta
scoring in both directions, cache behavior, original LF policy parity, and
variable-length AF3 input/output formatting. CPU integration exercises real
MPNN and stock-versus-accelerated NetMHCpan ranks using shared software.

After preparing a run, the following executes on an existing CPU allocation
and does not submit jobs or use a GPU:

```bash
bash job.sh /path/to/prepared_run unit-tests
bash job.sh /path/to/prepared_run cpu-smoke
```

For a small real GPU integration run, prepare a separate directory with
`--smoke`, then explicitly submit it. It still uses actual MPNN, NetMHCpan and
AF3, so an unfavorable input may have no design passing the strict folding
gate. GPU folding was not rerun merely to create this package; protocol and
variable-length conversion tests are not a substitute for real GPU validation.

`SOURCE_PROVENANCE.json` identifies copied/extracted sources. Third-party
notices accompany the vendored adapters. The trained MPNN checkpoint and its
architecture are unchanged. Different library versions/hardware can change
stochastic trajectories even when the search policy is identical.
