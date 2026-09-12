# Local experiment artifacts

`$MALECNS_HOME` (`/Users/monomyth/code/data/malecns`) holds **downloads and derived connectome files only**: GCS feathers, `source.lock.json`, prepared CSR, soma bin.

This directory is what the grok fly-brain controller **writes**:

| Path | Contents |
|---|---|
| `checkpoints/rebot-pickup/` | KC→MBON Learn weights, phase-2 readout, eval JSON |
| `datasets/` | Teacher / pick logs this project recorded |
| `live/activity.bin` | Overlay activity (also dual-written to the lab IPC dir) |

Override with `FLYBRAIN_DATA`. Do not put these files back in the shared MaleCNS cache.
