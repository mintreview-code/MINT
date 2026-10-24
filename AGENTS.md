# FastSearchAF3

The user has authorized including the original AF3 integration in this new
package. The existing CPU app in the parent folder remains separate.

Do not change the search or folding algorithms without explicit user approval.
This includes proposals, scoring, early rejection, thresholds, ranking,
diversity, population updates, MPNN sampling, temperatures, model weights,
AF3 input construction, recycles, samples, confidence metrics, and fold gates.

The original reference is lf_pool120_plan_20261007/package on Digs.
On 2026-10-18 the user explicitly authorized porting the weighted8 scoring
policy and using temperature 0.4 for all joint proposals. The weighted source
is lf_shared8x5_weighted300_p95_r1_20261016/package; only its scoring policy
is adopted. Retain this package's 90/2 fold gates and 120-round preset.
See docs/SCORING_REVISION.json for hashes and validation. Further algorithm
changes still require explicit user approval.
On 2026-10-23 the user authorized using the selected objective for allele
processing order too. Weighted8 orders by each allele's eight-threshold sum;
legacy broad2 retains its <2% ordering. This is a local revision, not fresh
verification of the running Digs source. See the follow-up revision record in
docs/SCORING_REVISION.json.
Preserve SOURCE_PROVENANCE.json, upstream references, vendored notices, policy
tests, and the behavior-equivalence evidence in docs/PORTING.json.

Keep code simple. Prefer thin UI/setup wrappers around existing entry points.
No synthetic fold metrics or structures passed off as predictions.
Do not redistribute NetMHCpan, AF3 model parameters, private data, container
images, virtual environments, credentials, or run caches. Use tools/release.py
for an allowlisted source archive. New files must be added explicitly to the
release layout. Refresh the manifest after deliberate edits.

Never perform scientific compute or compilation on a cluster login host.
Prepared runs fingerprint source code; initialize new runs after edits.
