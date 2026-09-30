# Private GitHub development repository

The upload requested on 30 September 2026 preserves the existing Git history
and adds the current product source, experimental Python supervisors, tests,
aggregate technical documentation and dependency lock.

Repository: https://github.com/raulprtech/shadow-trainer

The repository is private. This upload is a source backup and development
handoff, not a release or a Circuito14 submission. Existing authorship metadata
and historical configuration paths are retained. No new license or statement
of exclusive ownership is introduced by this upload.

## Local artifacts excluded

- Virtual environments, caches and the `research/workspace/` experiment tree.
- Images/datasets, medical volumes and model checkpoints.
- Credential files and local environment configuration.
- New detailed STU-Net case-level receipts and internal chat handoffs.
- The 29 September application packet, which remains in the thesis vault.

Existing versioned manifests and historical aggregate experiment summaries
remain as provenance; they are not the underlying datasets. Tracked files are
not retroactively excluded by `.gitignore`, so the tracked tree and reachable
history are inspected before pushing as well as new additions.

## Validation

The upload preparation checks candidate paths, sizes and common credential
signatures across candidate contents and reachable historical Git blobs.
This is an upload hygiene check, not a repository-wide security assessment.
CPU tests are run with CUDA devices hidden, without starting a GPU experiment.
On 30 September, the dedicated MVP environment passed 108 CPU tests in
14.01 seconds; one scientific evaluator module was skipped because SciPy is
not installed in that minimal environment. Its eight unittest cases were
then run separately in the existing scientific environment and all passed
in 5.078 seconds. That environment has no pytest installed, so unittest was
used directly. Both runs hid CUDA devices and used synthetic fixtures;
no GPU experiment or dataset download was started.

These checks cover the selected tests, not the known edge cases recorded in
later runtime audits. Source selection and history scanning found no matches
for the credential signatures checked. The private local upload inventory
contains file sizes and hashes; it is not added to Git.

The default execution strategy remains `sync`; the long STU-Net prefetch gate
is pending. The README identifies known robustness limitations. Dataset/model
downloads and real STU-Net training depend on separately authorized local
assets and laboratory integrations; cloning this repository does not provide
those assets or external access tokens.
