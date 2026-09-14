# fly-brain-grok

MaleCNS as an **external** controller for the [ReBot B601-DM](https://github.com/monomyth/rebot-motion-lab/tree/fly-brain-grok) kinematic arm. Pictures go into optical neurons. A scored descending-neuron bus commands tool-frame Δx/Δy/Δz and grip. The lab unpacks that with IK. There is no connectome code inside Swift.

| | |
|---|---|
| Code (this repo) | https://github.com/monomyth/fly-brain-grok |
| Distilled weights + optic crop | https://huggingface.co/monomyth/fly-brain-grok |
| Simulator branch | https://github.com/monomyth/rebot-motion-lab/tree/fly-brain-grok |

![Pipeline](docs/preview/infographic-working.jpg)

![Pick sequence](docs/preview/pick-demo.gif)

![Approach](docs/preview/01-approach.jpg)
![Pinch](docs/preview/02-pinch.jpg)
![Hold](docs/preview/03-hold.jpg)

## What works

Last live 20 mm episode (`loop-da2`): `acting_map=dn_bus` on 29/29 ticks, `fly_picked`, cube z 111 mm, `tcp_level`, hold 2.0 s. Attach is a pinch of the **solid** (corner or edge is enough). Overlay / k-NN / `|dz|` rewrite are not this path. Prior `eval-size20`: attach tick 19, cube z 121 mm, hold 2.8 s.

- Crop LIF: 59,740 neurons, 150 steps/tick, Front + Gripper JPEG (`apply:false`)
- Scored DN bus: DNfl, DNxl, DNa01, DNa02, DNp01 (abort), MDN, DNp07, DNp10
- After pinch, Δz comes from distilled `w_contact × scored DN Hz`, not a sign flip of the pre-grasp mix
- 20 mm table cube: fingertips down (TCP is still the fingertip). keep_level hits link5 on the floor below ~42 mm

## What does not work yet

![Gaps](docs/preview/infographic-not-working.jpg)

- `da_learned` is not claimed for `loop-da2` (that run logged `mbon_gate` 1.0). A new `--no-overlay` episode must gate from Kenyon current and move KC→MBON on the terminal pulse
- T1 motor-neuron rates are logged, not the joystick
- Live plant is the optic crop, not all ~166k MaleCNS cells
- Physical B601 is joint-space CAN, not MCP TCP. Transfer plan: [docs/hardware-test-plan.md](docs/hardware-test-plan.md); simulator gaps: [docs/sim-to-real-todo.md](docs/sim-to-real-todo.md)

## Run

Needs macOS 14+, the `fly-brain-grok` simulator branch, Python 3.11+, and MaleCNS downloads in `$MALECNS_HOME` (not this repo).

```sh
git clone https://github.com/monomyth/fly-brain-grok.git
cd fly-brain-grok
git clone -b fly-brain-grok https://github.com/monomyth/rebot-motion-lab.git rebot-motion-lab-grok

python3 -m venv controller/.venv
source controller/.venv/bin/activate
pip install -e controller huggingface_hub numpy scipy pillow pyarrow

export MALECNS_HOME="${MALECNS_HOME:-$HOME/data/malecns}"  # or /data/malecns on a GPU box
export FLYBRAIN_DATA="$PWD/data"

# weights (tiny) + optional optic crop (~19 MB)
# If the Hub repo is empty, upload once with a *write* token:
#   huggingface-cli login
#   python scripts/publish_hf.py
huggingface-cli download monomyth/fly-brain-grok --local-dir "$FLYBRAIN_DATA/hf-fly-brain-grok"
cp "$FLYBRAIN_DATA/hf-fly-brain-grok/g-distill.npz" "$FLYBRAIN_DATA/checkpoints/rebot-pickup/"

# 1. Open the Grok lab (Dock icon). Do not use a different ReBot app.
bash rebot-motion-lab-grok/scripts/wrap-debug-app.sh
python controller/scripts/open_lab.py

# 2. Drive it from MaleCNS (app already open)
python controller/scripts/run_dn_bus.py --no-overlay --size 20 --ticks 120 --hold 2 \
  --tag eval-size20 --gains "$FLYBRAIN_DATA/checkpoints/rebot-pickup/g-distill.npz"
```

Connectome prepare (once, large): `python controller/scripts/prepare_graph.py` then crop via `controller/arm/crop.py`. The Hugging Face `*.npz` crop skips a lot of that if you only want to replay the distilled controller.

## Layout

| Path | Role |
|---|---|
| `controller/` | MaleCNS crop LIF, DN bus, U unpack, MCP client |
| `docs/preview/` | Infographics and pick stills |
| `docs/malecns-arm-control-plan.md` | Architecture |
| `data/checkpoints/rebot-pickup/` | Local eval JSON (weights on Hugging Face) |

Lab IPC name: `rebot-motionlab-grok-<uid>`. Capture uses `apply:false` so it does not steal the live camera.
