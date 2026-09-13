# Venue strategy — September 2026

## Recommended order

1. **EuroMLSys 2027** — strongest fit for an early systems contribution about
   resource-aware ML, storage, recovery, and edge retraining. The 2027 call is
   not yet frozen; the recent six-page format is compatible with a focused
   version after Stage38 and the baseline experiments.
2. **IEEE IPAS 2027, Edge Deployment of AI special session** — pragmatic
   applied target. The currently published paper deadline is 10 November 2026,
   leaving time after Circuito 14 for repeated physical experiments.
3. **MLSys 2027** — highest-ambition systems target. Submission closes
   30 October 2026. Attempt only if Stage38 passes or yields a strong negative
   result, the runtime is evaluated against executable baselines, and repeated
   multi-workload results are ready.
4. **CVC 2027** — acceptable fallback when framed around resource-bounded 3D
   vision training. Round 2 closes 1 October 2026, which conflicts with the
   Circuito 14 package and leaves too little time for the missing evidence.
5. **MIDL/MICCAI** — separate medical paper only after held-out segmentation
   evaluation. The current NIfTI systems run cannot support that manuscript.

## Go/no-go decision

On completion of the physical matrix:

- **MLSys go:** at least three workload families, an executable baseline,
  repeated cold/warm measurements, recovery equivalence, and a defensible
  systems-level advantage or negative finding.
- **EuroMLSys/IPAS go:** complete artifact, strong safety/recovery evidence,
  and at least one convincing capacity or bounded-storage experiment; broad
  performance superiority is not required.
- **CVC go:** only if a complete anonymized paper exists before the deadline
  without weakening Circuito 14 and the registration cost is acceptable.
- **No-go:** only short independent runs, unmatched cache warmth, missing
  baseline, or an unresolved integrity/equivalence failure presented as speedup.

The same technical manuscript must not be simultaneously submitted to archival
venues. A non-archival workshop version is considered only after reading the
selected venue's current overlap policy. Posting a preprint and disclosing AI
assistance also follow the final venue's current rules.

## Current manuscript status

The v0.1 draft has a complete narrative, design, methodology, honest initial
results, related work, limitations, and next experiments. It is not ready for
submission because it lacks repeated comparative experiments, a framework
baseline, Stage38 closure, full bibliography verification, author review, and
venue-specific formatting.

Official pages used for the strategy:

- MLSys 2027 CFP: https://mlsys.org/Conferences/2027/CallForResearchPapers
- EuroMLSys: https://euromlsys.eu/
- IEEE IPAS 2027 event: https://events.vtools.ieee.org/m/563249
- CVC 2027 CFP: https://saiconference.com/CVC/CallforPapers
- MIDL 2027 scope: https://2027.midl.io/aims-and-scope
