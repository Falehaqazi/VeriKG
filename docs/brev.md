# Running VeriKG on Brev

The Launchable gives a GPU VM with the repository cloned, everything
installed, the prebuilt graph downloaded and the app running on port 8501.

## Deploy

Open the VeriKG Launchable link, pick the instance and click **Deploy
Launchable**. When the instance is ready, open the secure link for port 8501.
Setup takes about 5 minutes; the app log is `~/VeriKG/verikg_app.log`.

Optional setup values:

| Name | Effect |
| --- | --- |
| `NVIDIA_API_KEY` | Lets a NIM model plan queries in the app (without it, a keyword planner is used) |
| `BUILD_FROM_SOURCE` | `1` pulls Pharos and rebuilds every artifact instead of downloading them (about 1.5 h on one GPU) |

## Create the Launchable (maintainer)

In the Brev console, **Launchables > Create Launchable**:

1. **Files and runtime**: Git repository `https://github.com/Falehaqazi/VeriKG`, VM mode.
2. **Setup script**: paste
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   bash "$HOME/VeriKG/brev/setup.sh"
   ```
3. **Networking**: expose port `8501` as a secure link named `verikg-app`. Jupyter on 8888 can stay on.
4. **Compute**: any single GPU (an L4 or T4 is enough; the app itself runs on CPU).
5. **Setup values**: `NVIDIA_API_KEY` (optional, secret), `BUILD_FROM_SOURCE` (optional, default `0`).
6. Name it **VeriKG**, set view access to anyone with the link, create, and copy the link into the README.

## From a terminal on the instance

```bash
cd ~/VeriKG
python3 -m kgc ask --question "Which approved drugs act on kinases linked to breast cancer?"
python3 -m kgc gnn --split scaffold        # re-run the link prediction comparison on the GPU
```
