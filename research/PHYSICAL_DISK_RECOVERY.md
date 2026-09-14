# Physical disk recovery gate for the paired matrix

Recorded on 2026-09-14. This is an operational recovery procedure, not an
experiment result.

## Authoritative observations

- The only registered distribution is `Ubuntu`, running as WSL2.
- The backing file is
  `C:\Users\raul_\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx`.
- Its physical file length was 127,232,114,688 bytes (118.49 GiB).
- The ext4 filesystem reported 263,940,717 blocks, 244,608,162 free blocks,
  and a 4,096-byte block size: approximately 79.19 GB were in use.
- C: reported 3,910,651,904 free bytes in the final inventory.
- Exact SHA-256 grouping of historical files larger than 20 MiB found only
  3,901,612,065 bytes of duplicate content. Deduplicating the evidence would
  not reach the runtime floor and is therefore rejected.
- The gap between the VHDX file length and allocated ext4 content is large
  enough that offline VHDX compaction is the preferred non-deleting recovery.

No historical run, cache, checkpoint, or dataset should be deleted for this
operation. Do not use `wsl --unregister`.

## Procedure requiring the Windows host

This cannot be completed from the active Codex task because shutting down WSL
terminates that task. First finish or close every WSL and Docker workload.

1. In Ubuntu, ask ext4 to discard unused blocks:

       sudo fstrim -av

2. Exit Ubuntu. In an elevated Windows PowerShell, stop WSL:

       wsl --shutdown

3. Still elevated, start DiskPart:

       diskpart

4. At the `DISKPART>` prompt, select the exact detached dynamic disk and
   compact it:

       select vdisk file="C:\Users\raul_\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
       detail vdisk
       compact vdisk
       exit

5. Check the result in PowerShell before reopening Ubuntu:

       Get-Item "C:\Users\raul_\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx" | Select-Object Length
       Get-PSDrive C | Select-Object Used,Free

Microsoft documents that dynamically expanding VHD files grow but do not
automatically shrink, and that `compact vdisk` requires a selected VHD that is
detached or attached read-only:

- https://learn.microsoft.com/en-us/windows/wsl/disk-space
- https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk

## Re-entry gate

After Ubuntu restarts, do not launch a workload first. Run:

    cd /home/raulprtech/shadow-trainer
    PYTHONPATH=src /home/raulprtech/clinical_core/.venv/bin/python \
      research/run_pair_matrix.py --preflight-only

The physical matrix may start only when this command exits successfully with
at least 21,474,836,480 free bytes on `/mnt/c`. Use a new output directory and
never reduce the floor to force admission.
