# Testing and validation

The fast test suite uses fake predictor scores, fake MPNN samples, and a mocked
AF3 request. It can run on a Mac without Docker, a licensed NetMHCpan installation,
or an NVIDIA GPU. Passing it establishes input handling and search-policy
consistency; it does not establish that AF3 or NetMHCpan can run on that machine.

## Run the unit tests

From the package root, with `requirements-ui.txt` installed (this includes the CPU
dependencies and Gradio, which the UI boundary test imports):

```sh
python -B -m unittest discover -s tests -p 'test_*.py' -v -b
```

The suite includes the historical 43 tests plus nine weighted-score regressions and one weighted cleanup-endpoint test:

| File | Tests | What they check |
| --- | ---: | --- |
| `tests/test_portable.py` | 15 | PDB chain/numbering normalization, the LuxSit fixture and 43 fixed positions, invalid inputs, rank thresholds, deduplicated cache completion, exact mutation-local counts, original proposal/selection policy parity in both directions, and AF3 manifest/confidence parsing with a mocked fold. |
| `tests/test_campaign.py` | 8 | Allocation pause behavior, separate temperature settings, fold-journal reuse, rejection of a changed pending shortlist, and avoiding finalization when paused. |
| `tests/test_cleanup.py` | 15 | Verified archival before scratch deletion, retention of paused/failed runs, missing or corrupted results, insufficient space, export failures, ownership checks, and symlink rejection. |
| `tests/test_weighted.py` | 9 | Strict threshold boundaries, 1,920 delta/full-scan comparisons across modes/directions/allowances, repeated-window multiplicity, objective-based allele order and deterministic ties, weighted ranking and diversity, eight joint calls at 0.4, full weighted endpoints, independent-audit mismatch detection, and successful final report export. |
| `tests/test_packaging.py` | 6 | Exact campaign/smoke preparation presets, bundled versus custom fixed positions, explicit overrides, existing-run preservation, and UI preparation without job submission. |

The packaging tests write real temporary run configurations and normalize real
PDB inputs. They mock external installation availability and intercept job
submission/process launch in the UI test. They do not invoke a predictor or
start Gradio's server. The UI test checks both an explicit seed and a blank seed;
a blank seed uses the same preset default as the CLI (`202610071` for smoke).

`tests/reference_policy.py` contains the frozen policy excerpts used by the parity
tests. It is a test reference, not a second implementation used by the search.
The policy-parity fixture is the 117-residue LuxSit example; separate tests also
exercise a shorter chain and residue mapping.

Individual suites can be run with:

```sh
python -B tests/test_portable.py
python -B tests/test_campaign.py
python -B tests/test_cleanup.py
python -B tests/test_packaging.py
```

The `test_campaign.py` entry point also includes the cleanup tests; discovery
runs each test case once.

## Historical checks performed on 2026-10-14

The original baseline and the readability-formatted copy were checked on
2026-10-14 using the existing local Python environment. No GPU inference, real
NetMHCpan scoring, container rebuild, or cluster job was started for these checks.

| Check | Result |
| --- | --- |
| Original baseline unit suite | 37 passed in 2.915 seconds. |
| Formatted copy, before adding packaging tests | Original 37 tests passed in 2.836 seconds. |
| Full suite with preparation/UI tests | 43 passed in 4.086 seconds, including blank UI seed coverage. |
| Temperature-test robustness | Accepted AST-equivalent reformatting; rejected altered initial/immigrant temperature keys and a changed joint-temperature index. |
| Python syntax | All current Python files parsed successfully. |
| Python behavior-preserving formatting check | Parsed ASTs of all 17 original runtime Python files matched the original baseline. |
| Shell syntax | `bash -n job.sh native/build.sh` passed. |

The original temperature test compared literal source strings and initially
failed when readability formatting changed quotes and spaces. Its assertions now
compare Python AST structure. It still checks separate initial and immigrant
temperature configuration and `config['temperatures'][index % 3]` in joint
proposals. This was a test robustness change; runtime algorithms were not edited
to make it pass. The final full-suite result above includes this revised test.

The environment used for these checks was Python 3.12.4, NumPy 2.5.4,
PyTorch 2.14.1, ProDy 2.6.1, Biopython 1.88, and Gradio 6.30.0. `matplotlib` was absent from that
existing environment. It is listed in `requirements.txt` and is needed for final
search plots and `lf.py doctor --inside`, but the unit suite does not import it.
These observed versions are a test record, not a pinned compatibility guarantee.

## Revised scoring validation

See `SCORING_REVISION.json` for the 2026-10-18 source hashes and historical test
results, and its follow-up revision record for the 2026-10-23 allele-order change.
Historical tests remain evidence for their original version.
The current code uses a schedule-length-based joint temperature index, so `[0.4]`
works for every attempt. Broad2 parity checks remain available, while the default
prepared run uses weighted8. The original frozen reference policy is unmodified.

## Real model integration tests

`tests/cpu_smoke.py` performs real ProteinMPNN sampling, compares accelerated
and stock NetMHCpan predictions, and verifies that the exact rank cache starts
empty and reuses completed predictions. It writes `cpu_smoke_passed.json` to a
prepared run directory. It does not perform any folds or submit jobs by itself.

On a configured Linux compute node, with a prepared run:

```sh
bash job.sh /absolute/path/to/prepared-run cpu-smoke
```

This is an integration command, not a Mac unit-test command. Run it inside a CPU
allocation with access to the configured NetMHCpan installation and CPU
environment. The checked-in accelerator is Linux x86-64/AVX2 code. The full
campaign backend also uses Linux paths and `/proc/cpuinfo`; this source package
does not add a native macOS scoring backend.

The campaign folding smoke path is `prepare.py --smoke` followed by an explicit
submission of the prepared run. Despite the small search sizes, that path uses
real ProteinMPNN, NetMHCpan, and AF3 and requires a properly configured NVIDIA
GPU allocation. See the main README for preparation and submission commands.

## Historical evidence, not a new GPU validation

The imported `VALIDATION.json` records earlier upstream checks dated 2026-10-06:

- Fifteen portable tests passed in the shared CPU container.
- Real CPU integrations covered the 117-residue deimmunization fixture and a
  23-residue immunization fixture, each comparing 100 stock/accelerated
  peptide–allele pairs and checking cache reuse.
- Relocation preparation and submission dry-run passed. One relocated full-suite
  rerun could not start Python because Apptainer encountered a host resource error.
- **GPU integration was not executed for that packaging validation.**

Those records are preserved as upstream evidence. They are not fresh results
from this Mac or a claim that the copied package has just completed an AF3 fold.
The AF3 adapter, warm-worker daemon, batch runner, and runtime adapter code are
included, along with the inspected build recipe in `runtime/af3`. The upstream
engine is external, as requested. Its source was inspected to verify the
matching revision, but its own tests were not executed. JAX/CUDA, the engine and
compiled extensions, compatible model parameters, and a working container image
are required for real folds. See [the runtime guide](../runtime/af3/README.md).
