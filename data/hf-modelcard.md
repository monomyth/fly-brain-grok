---
license: mit
library_name: numpy
tags:
  - neuroscience
  - connectomics
  - malecns
  - robotics
  - grok
task_categories:
  - reinforcement-learning
---

# fly-brain-grok

Distilled **MaleCNS crop LIF gains** (`g`) plus a 12-parameter **DN→TCP unpacker** (`U`) for the ReBot B601-DM kinematic arm in [monomyth/rebot-motion-lab](https://github.com/monomyth/rebot-motion-lab/tree/fly-brain-grok) (`fly-brain-grok` branch).

This is **not** a Transformers checkpoint. Load with `numpy.load`.

Code and run instructions: [github.com/monomyth/fly-brain-grok](https://github.com/monomyth/fly-brain-grok)

## Files

| File | What |
|---|---|
| `g-distill.npz` | `g` (per-class synaptic gains) and `u` (U params including `w_contact`) |
| `g-distill.json` | Same vectors + `g_hash` `7bca9074a252951b` |
| `hop-probe.json` | Crop identity: 59,740 neurons, 267 scored DNs |
| `malecns-v1.0-crop-v1-optic-rich-mancType-DNfl-DNxl.npz` | Optic+DN CSR crop (optional; skip full graph prepare) |
| `eval-size20.json` | Last 20 mm live episode summary |

## Load

```python
import numpy as np
blob = np.load("g-distill.npz")
g, u = blob["g"], blob["u"]  # u[-1] is w_contact
```

Pass `g-distill.npz` to `controller/scripts/run_dn_bus.py --gains ...`.

## Provenance

MaleCNS / MANC types from the public Janelia male CNS connectome. Overlay / k-NN teachers are **not** stored here. Live `loop-da2`: `fly_picked` (dn_bus, cube z 111 mm, hold 2 s). `da_learned` is not claimed for that run (`mbon_gate` 1.0). Cleanup opens the gripper then Folds (lab routing).
