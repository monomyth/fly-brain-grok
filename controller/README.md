# MaleCNS controller

External Python process for the grok ReBot lab. No connectome code lives in `rebot-motion-lab-grok`.

`fly_picked` / `da_learned` are sealed in `arm/score.py`. Overlay / k-NN is `lab_picked` only.

Launching the simulator and running a simulation are **separate**:

```sh
export MALECNS_HOME="${MALECNS_HOME:-$HOME/data/malecns}"   # or /data/malecns on a GPU box
export FLYBRAIN_DATA="${FLYBRAIN_DATA:-$PWD/../data}"
cd controller
source .venv/bin/activate

# 1. App (Dock icon)
bash ../rebot-motion-lab-grok/scripts/wrap-debug-app.sh   # after a Swift rebuild
python scripts/open_lab.py
# or: open "../rebot-motion-lab-grok/dist/ReBot Motion Lab Grok.app"

# 2. Simulation against that open window
python scripts/run_dn_bus.py --no-overlay --size 20 --ticks 120 --hold 2 \
  --tag eval-size20 --gains "$FLYBRAIN_DATA/checkpoints/rebot-pickup/g-distill.npz"
python scripts/run_teacher.py         # scripted pick; lab_picked only
python scripts/run_policy.py          # attach only; does not launch or move the cube
```

Other:

```sh
python -m malecns_cache.lockfile
python -m malecns_cache.prepare
python -m pytest
python scripts/run_silence.py
python scripts/train_readout.py
```

Prepared graph: **166,700** neurons, **25,582,938** signed edges (`prepared/malecns-v1.0-graph.npz`).
