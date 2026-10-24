# How the full Fast Search package works

This is the complete CPU search plus GPU AF3 workflow copied from the original
campaign package. The original 2026-10-14 readability pass changed formatting and comments only;
the parsed Python syntax trees of the eight reformatted scientific modules are
identical to that historical source snapshot. See [readability-check.json](readability-check.json).
The 2026-10-18 revision deliberately changes scoring and joint temperature;
that historical AST-equivalence claim does not apply to the revised files.
See [SCORING_REVISION.json](SCORING_REVISION.json) for current evidence.

The full workflow is intended for a Linux Slurm allocation with one supported
NVIDIA GPU and the configured CPU, NetMHCpan, and AF3 installations. Copying its
code to a Mac does not make AF3 a Mac CPU application. The existing CPU-only
Gradio application in the parent folder remains a separate application.

## Where to start reading

Read `prepare.py`, `lf.py`, `job.sh`, `run120.py`, and then `search.py`, in that order.
`prepare.py` is the new, small single-run preparation wrapper. It applies the
120-round budget plus revised weighted scoring and temperature 0.4 to `lf.initialize`.
It does not add a second search implementation.
`job.sh` launches **`run120.py`**, whose `Experiment120` extends the algorithm in
`search.Experiment` with allocation-aware pause/resume and fold journals.
Running `search.py` directly uses the base experiment's different time-budget
behavior and is not the normal packaged launch path.

| File or directory | Responsibility |
| --- | --- |
| `prepare.py` | Prepare one independent run with weighted scoring and joint temperature 0.4; never submit it. |
| `app.py` | Optional Gradio controls for the same preparation helper, existing Slurm submission, and saved progress records. |
| `lf.py` | Read site settings, normalize inputs, create immutable run provenance, check dependencies, and submit through Slurm. |
| `configs/` | Digs and generic Linux installation paths and Slurm resources. Copy the appropriate example to `site.local.json`. |
| `job.sh` | Prepare job-local scratch, start AF3, run the CPU search, forward the advance signal, and clean up temporary job resources. |
| `run120.py` | Resume the campaign across allocations, retain completed AF3 batches, and commit complete rounds. |
| `search.py` | Initial population, single/pair/joint proposals, parent rotation, fold selection, population/archive updates, and final confirmation. |
| `objective.py` | Eight cumulative thresholds, weighted score, and explicit legacy broad2 mode. |
| `core.py` | Predictor thread pool, exact SQLite rank cache, objective ordering, diversity selection, and mutation-local early rejection. |
| `mapper.py` | NetMHCpan process wrapper, predictor fingerprint, peptide windows, full rank maps, and single-substitution enumeration. |
| `models.py` | CPU ProteinMPNN adapter, AF3 socket client, fold parsing, and structural acceptance gates. |
| `geometry.py` | PDB normalization, residue mapping, fixed-position checks, backbone serialization, and aligned CA RMSD. |
| `mpnn_helpers.py` | Model construction and features for the unchanged vendored ProteinMPNN sampler. |
| `vendor/mpnn/` | Bundled original model and feature implementations. |
| `vendor/af3/` | Included AF3 daemon, Apptainer command builder, resident inference driver, and runtime helpers. See [AF3.md](AF3.md). |
| `runtime/af3/` | Exact inspected image-build recipe, external engine revision, and installation instructions. |
| `native/` | NetMHCpan AVX2 hook, original C source, and build script. The predictor installation remains separate. |
| `assets/` | Original ProteinMPNN checkpoint and default 25-allele panel. |
| `examples/` | LuxSit high-confidence sample `2.pdb` and its 43 fixed residues. |
| `archive_results.py` | Optional verified export and deletion of an explicitly owned scratch run after successful completion. |
| `tests/` | Existing policy, geometry, checkpoint, cleanup, and real-model smoke checks. See [TESTING.md](TESTING.md). |

## Preparing and launching a run

Use `prepare.py` for the campaign settings documented below. Its default input
is the bundled LuxSit sample, with the bundled 43 fixed positions. A custom PDB
does not inherit those fixed positions; specify its own fixed-residue input.
The optional `app.py` interface, on port 7861 by default, calls this same helper.
Preparation, status refresh, and command preview do not submit jobs; the
explicit submission action calls the existing `lf.submit` function.

Dependency checks, filesystem paths, and Slurm commands are evaluated on the
machine running the Python process. To prepare and submit on Digs through a
browser, run the interface on an appropriate Digs allocation and access it
through an SSH tunnel. Simply opening the interface on a Mac does not make
Digs installation paths available or run commands remotely.

1. The underlying `lf.py init` reads `site.example.json`, applies any `--site-config` overrides,
   and checks that the named dependency paths exist and are readable. This
   check does not load AF3 or prove GPU compatibility.
2. `geometry.normalize` selects one protein chain, retains N/CA/C/O backbone
   atoms, and renumbers it to chain A, residues 1 through L. Missing backbone
   atoms, noncanonical residues, and physical chain breaks are rejected.
3. Fixed residues expressed using original PDB identifiers are translated into
   the new one-based indices. `residue_mapping.json` records that translation.
4. Initialization creates a new run directory containing `input.pdb`,
   `config.json`, `residue_mapping.json`, and `fixed_residues.txt`. Configuration
   records input hashes and hashes of protected code, model, and kernel files.
   `init` does not submit work; `launch` prepares and submits it.
5. `lf.py submit --dry-run` prints the Slurm command. A real submission runs
   `job.sh`, allocates the configured CPU and GPU resources, and records
   `submission.json` plus append-only `submissions.jsonl`.
6. `job.sh` creates private `/tmp/lf-portable.*` scratch, including the rank
   database, AF3 socket, and JAX compilation cache. It starts a resident AF3
   process through the configured Apptainer image. CPU work uses either
   `cpu_python` or the configured CPU image; ProteinMPNN uses CPU in both cases.

Keep prepared runs with the exact package that prepared them. Editing protected
source or replacing model/kernel files causes submission or resume to fail its
hash checks. Formatting also changes file hashes and parser provenance: old
prepared runs cannot be transferred to the newly formatted package unchanged.

## Initialization: creating the first usable population

1. Score the input sequence across all configured alleles and all 9-, 10-, and
   11-residue windows. For the 117-residue LuxSit example and 25 alleles, this
   is 324 windows × 25 alleles = 8,100 window–allele occurrences.
2. Sample `initial_designs` ProteinMPNN draws at all mutable positions, using
   the configured initial temperature schedule and deterministic seeds.
   Cysteine and unknown amino acids are omitted from sampled alternatives;
   fixed identities are preserved. Identical sequences are deduplicated.
3. Score distinct missing peptide strings across batches of up to 32 records.
   The cache stores ranks by predictor fingerprint, peptide string, and allele.
   Repeated windows still count at each sequence position in the objective;
   caching avoids recalculating their identical predictor scores.
4. Select up to `initial_folds` records, reserving one quarter of those slots
   for sequence diversity. If the original input is absent, append it, so the
   first shortlist may contain `initial_folds + 1` entries.
5. Fold the shortlist with AF3 in batches of four. If none pass, try remaining
   scored draws in further groups of up to 16 until a passing fold is found or
   the available draws are exhausted. Initialization fails if no parent passes.
6. Build two sets of up to `beam` passing records: the **archive** preserves
   the best scores, while the **population** also reserves diversity slots.
   Commit round zero to `state.json.gz`.

A starting PDB is not a previously scored population. A new run must start with
an empty prediction cache. Only the same run may restore its own saved cache.

## One search round

1. Merge and deduplicate population and archive. Always use the best parent,
   then rotate through other parents up to `parents_per_round`.
2. Build each parent's full cached peptide map. Mutable positions covered by
   rank <2% windows form the hotspot ordering. Immunization also includes
   mutable positions with zero current coverage so it can create new hits.
3. Generate single substitutions, proactive pairs, and joint MPNN proposals.
   Single alternatives exclude the original residue and cysteine. Each selected
   pair uses up to three MPNN alternatives per position with marginal probability
   at least 0.01. Joint proposals cycle through 2, 4, and 8 hotspot positions
   at temperature 0.4 for all eight attempts. Pair-alternative marginal
   probabilities are still queried at temperature 1.0; that separate step is unchanged. Duplicate sequences are removed.
4. Rescore only peptide windows overlapping changed positions. Alleles are
   processed in groups of five, starting with those having the largest parent
   contribution to the selected objective: the eight-threshold cumulative sum
   in weighted8 mode, or the <2% count in legacy broad2 mode. Allele names break
   ties deterministically. Bounds can reject a candidate before all groups finish; acceptance
   requires complete counts. The parent-relative worsening allowance is
   `max(2, ceil(worsening_fraction * parent_objective_score))`.
5. Every `immigrant_interval` rounds, generate `immigrants` fresh MPNN draws
   from the original input backbone. Immigrants use their separate temperature
   schedule when supplied and enter the exploratory pool.
6. Reserve up to three quarters of `folds_per_round` for non-worsening
   candidates; fill the remaining slots from the other eligible candidates
   with diversity selection. Previously folded sequences are excluded as
   implemented in `one_round`.
7. Before AF3, recompute each selected candidate's complete map and require
   its counts to equal the mutation-local counts. Fold the selected sequences.
8. Update the archive from previous archive members and newly passing folds.
   The champion's primary objective cannot worsen. Update the separate
   population with diversity inside score slack
   `max(10, ceil(0.1 * champion_objective_score))`, and checkpoint the completed round.

The primary objective is **the sum of cumulative peptide–allele occurrence
counts at EL %Rank thresholds 0.25%, 0.50%, 0.75%, 1.00%, 1.25%, 1.50%,
1.75%, and 2.00%**, using strict inequalities. A rank of 0.2 contributes eight;
a rank of 1.8 contributes one. Deimmunization minimizes the score and
immunization maximizes it. This score governs screening, ranking, exploration,
diversity allowances, retention, independent audit, and endpoint checks. The tie-breaker is `strong_sum`, the sum of the cumulative
rank <0.25% and rank <0.5% counts, followed by record ID. A rank <0.25%
occurrence contributes twice to `strong_sum`; these are not disjoint bins.
There is no separate staged optimization of the stricter thresholds.

## What AF3 does here

The client serializes each candidate using its parent backbone with candidate
residue names, then submits a CIF manifest to the resident AF3 worker. This
request supplies **empty paired and unpaired MSAs and no template field**.
The runtime explicitly clears templates when none are supplied; the serialized
parent coordinates do not constrain the AF3 fold.

The unchanged adapter uses five recycles and one diffusion sample per design.
It accepts a fold only when the sequence and fixed residues match, mean CA
pLDDT is **at least 90**, and aligned CA RMSD against the **original root** is
**at most 2 Å**. Passing folds supply the backbone used by future MPNN proposals.
See [AF3.md](AF3.md) for runtime dependencies, protocol, and output details.

## Historical CLI defaults versus the 120-round campaign

The copied `lf.py` CLI defaults are not a complete preset for the later campaign.
Its preparation script at
`peptide_scan_investigation/lf_pool120_plan_20261007/prepare.py` applies additional
settings after `lf.initialize`. The comparison below was checked directly
against that source. It describes the original production campaign, not its
small smoke runs.

The table below records the **historical 2026-10-14 package and source campaign**,
not the current scoring/temperature defaults. Current `prepare.py` uses weighted8
and initial, immigrant, and joint temperature schedules `[0.4]`; it retains the
120-round budget and the other portable per-run choices described below.
The historical `prepare.py` applied the production settings in the right-hand
column, with portable per-run choices: a user-supplied site file, the
bundled sample unless another input is given, the default 25-allele panel,
cleanup disabled, and seed `26051011` unless explicitly overridden. It adds a
campaign-preset label instead of impersonating a historical lineage ID. The
site file controls resource paths and constraints; the wrapper does not force
the old private CPU-image path or A4000 constraint.

| Setting | Bare `lf.py init` defaults | Original 120-round campaign preparation |
| --- | --- | --- |
| Requested rounds | 80 | 120 |
| Initial MPNN draws | 512 | 512 |
| Initial fold slots | 96, plus root if absent | Same |
| Initial temperatures | Falls back to `[0.4, 0.7, 1.0]` | `[0.4]` |
| Joint MPNN temperatures | `[0.4, 0.7, 1.0]` | Same |
| Immigrant temperatures | Falls back to `[0.4, 0.7, 1.0]` | `[0.4]` |
| Beam / parents per round | 24 / 8 | Same |
| Position pairs / joint proposals per parent | 8 / 8 | Same |
| Folds per round | 48 | Same |
| Immigrant interval / count | Every 5 rounds / 16 | Same |
| Worsening fraction | 0.02 | Same |
| Predictor workers with 8 CPUs | 7 | Same |
| Native predictor mode | `auto` | `required` |
| Stored `search_hours` | 44 | 42 |
| `allocation_search_hours` | Not set; `run120.py` falls back to 42 | Explicitly 42 |
| Slurm time limit / memory | 48 hours / 30 GiB | Same |
| GPU constraint | Empty in the example site | `A4000` |
| CPU environment | Example universal image | Campaign-specific `vep_fold_sandbox` path |
| Random seed | `26051011` | Per-lineage seed from the campaign plan |
| Scratch export and cleanup | Disabled | Enabled with explicit lineage-owned paths |

**The normal launcher uses `run120.py`, whose allocation limit overrides the
base class's `search_hours` check.** It requests a pause when the current
allocation reaches `allocation_search_hours` (42 hours if absent), or when it
receives the Slurm advance signal. It then resumes toward the requested round
count in a later allocation. Changing only `--search-hours` does not change
this launcher's allocation boundary. Similarly, bare `lf.py init --smoke` sets
`search_hours=1.5` but does not set `allocation_search_hours`; the original
campaign explicitly set both for its smoke configurations.

`prepare.py --smoke` retains the original small-test sizes, with weighted scoring and temperature 0.4:
four initial draws, five initial fold slots, one round, beam two, one parent,
at most two mutable hotspot positions, one pair, two joint proposals, two
folds per round, and no immigrants. It sets both time fields to 1.5 hours
and uses seed `202610071` unless explicitly overridden. AF3 remains real;
this is a smaller integration run, not a substitute folding implementation.

The production campaign prepared forty lineage/direction entries. Its
lineage coordinator is separate from the per-run scientific package. Running
`prepare.py` prepares one independent run. Bare `lf.py init` does not recreate
that cohort, select its per-lineage inputs, or apply its post-initialization
overrides automatically.

## Checkpoints, pauses, completion, and files

| Path inside a run | Meaning |
| --- | --- |
| `config.json`, `input.pdb`, `residue_mapping.json` | Frozen settings and normalized input with the original-to-normalized residue map. |
| `provenance.json` | Predictor asset fingerprint, acceleration status, and whether this run restored its own cache. |
| `root.json` | Initial sequence's score summary. |
| `initial.json.gz` | Initial sampled records and fold outcomes. |
| `live.json` | Latest coarse scoring or folding progress; not a complete checkpoint. |
| `state.json.gz` | Last fully committed round, archive, population, history, and accounting. |
| `cache.sqlite` | Snapshot of this run's exact predictor cache; active writes occur in job-local scratch. |
| `pending_folds.json.gz` | Completed AF3 batches for an unfinished round, keyed by its shortlist signature. |
| `resume_accounting.json`, `paused.json` | Accumulated timing and allocation-pause status. |
| `round_###/results.json.gz`, `summary.json` | Candidate screening, fold outcomes, and completed-round summary. |
| `af3_daemon.log`, `slurm-*.log` | AF3-worker and overall job logs. |
| `failure.json` | Recorded exception and timing if execution fails. |
| `confirmation.json.gz` | Two independently seeded AF3 confirmation folds of the champion. |
| `champion.pdb`, `champion.json`, `archive.fasta` | Best accepted structure, its metrics, and archived sequences. |
| `progress.csv`, `objective_score.png`, `below_2.png`, `strong_sum.png` | Final score history and plots. |
| `complete.json` | Final audit, confirmation pass count, stop reason, and timing. |

At an allocation pause, the current rank cache and accounting are saved. An
unfinished round is regenerated deterministically on resume; completed AF3
batches are reused only when their ordered sequence/ID/parent-PDB signature
matches exactly. Completed rounds clear the pending-fold journal.

Use the same run directory and original package to resume. The existing
`lf.py submit --resume` wrapper also requires `state.json.gz`, so a pause before
the first committed population needs inspection even if a fold journal exists;
do not delete the journal or silently prepare a replacement run.

Completion performs a fresh, unaccelerated original-NetMHCpan audit with no
rank-cache reuse, comparing all eight cumulative counts, then two independently
seeded AF3 repeats. `objective_audit.json` records the audited objective.
The immunization endpoint is eight times the number of window–allele occurrences;
merely putting every occurrence below 2% does not reach that endpoint. It records all
outcomes and does not silently replace a poorly reproduced champion. Reaching
the objective endpoint stops early only after both confirmation repeats pass;
the requested-round limit can complete with failed confirmation outcomes, which
remain visible in `complete.json` and `confirmation.json.gz`.

`archive_results.py` acts only when cleanup was explicitly configured. After
verifying terminal results, ownership, paths, and every copied file hash, it
exports scientific results to the requested archive and removes that owned
scratch run. The rebuildable rank cache is excluded. Without that opt-in, run
directories remain in place; `job.sh` still removes its own temporary `/tmp`
directory when the job exits.

## Deployment settings versus scientific policy

For another machine or cluster, start with a copy of `site.example.json`.
Set the NetMHCpan installation, AF3 image/model directory, CPU image or Python
environment, Slurm partition/account/constraint, CPU count, memory, and wall
time there. The sample paths describe the original environment; they are not
download instructions or bundled software. Keep user-specific site files out
of a shared source archive.

Scientific policy lives in `search.py`, `core.py`, `mapper.py`, `models.py`,
`mpnn_helpers.py`, the vendored models, and the prepared run configuration.
Changes to proposals, temperatures, weights, allele panels, objective/ranking,
early rejection, cache semantics, population/diversity, AF3 settings or gates,
and campaign budgets require an explicitly reviewed decision. Do not treat
these as performance-only deployment edits or change them during packaging.
Several values are implemented directly in code; editing a descriptive
`config.json` field such as a gate or recycle label alone does not change those
implementations.

The AVX2 hook changes predictor execution, so its original engine checksum and
CPU checks must remain intact. `auto` uses the hook when supported and otherwise
uses stock prediction; `required` fails if the verified hook is unavailable;
`off` uses stock prediction. A new run is required when the scorer fingerprint
changes. Keep algorithm edits separate from deployment and verify equivalent
scientific outputs before adopting a new execution backend.
