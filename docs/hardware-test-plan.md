# B601 test, train, demo plan (MaleCNS DN-bus)

For the grok fly-brain controller: pictures → crop LIF → scored DN bus → tool-frame Δx/Δy/Δz + grip. Hardware host is isengard (ReBot B601-DM, Gemini 305 gripper, Gemini 336L overview). Simulator gaps: [sim-to-real-todo.md](sim-to-real-todo.md).

This is a proposal. No MaleCNS-on-B601 trial has been run. Do not treat ACT cube-to-cup training loss, teleop success, or Mac `fly_picked` as physical DN-bus success.

## Transfer verdict

**Not a weight copy.** Distilled `g` maps DN rates onto millimetre TCP deltas for a kinematic MCP lab. The B601 driver maps seven `.pos` degrees onto Damiao CAN (`/dev/ttyACM0`, 921600 baud). There is no `rebot_servo_tcp`, no `gripper_mm`, no `cube.attached`.

What can transfer later, after adapters:

| Layer | Realistic? | Why |
|---|---|---|
| Optic crop on real JPEG | Maybe | Same 305/336L pair the lab pretends to render. Domain gap (lighting, 848×480, mounts) is large. |
| DN bus + U unpack | Only behind new IK | U emits mm deltas. Hardware needs a bounded Cartesian servo that the lab does not provide. |
| `da_learned` / KC→MBON | After the plant exists | Terminal DA is independent of IK, but useless if the arm never picks. |
| ACT cube-to-cup checkpoint | Different task | Joint imitation at 15 Hz, not DN-bus. Baseline only, not this model. |

Human teleop **has** put a cube in a cup (2026-09-06, 60°/s arm / 180°/s gripper). Autonomous ACT rollouts (2026-09-07) ran three 10 s segments and did not record a scored placement; `/reset` failed (`Return would exceed 30 seconds`). Dual-camera set: 5 episodes / 2053 frames at ~14.5 Hz. Demonstration labels on the 50-episode set are not policy success rates.

## Tools (do not run unless a phase says so)

Motor-active: `lerobot-calibrate`, `lerobot-teleoperate`, `lerobot-record`, `lerobot-rollout`, `record-dual-five`, `run-dual-policy`, `open-gripper`, `close-gripper`. `connect()` enables motors even with `calibrate=False`.

Camera-only: Orbbec `dual_rgb_bridge` / `start-dual-cameras` (opens devices).

Read-only: source, logs, `meta/info.json`, saved JPEG/PNG under `~/robotics/logs`.

Herdr session `grok-fly-brain` on isengard is for agents/logs. It is **not** an e-stop. `/stop` on a policy launcher returns toward folded-zero and may leave torque on.

## Phase 0 — host and exclusive ownership

1. Confirm nobody else owns `/dev/ttyACM0` or the Orbbec devices.
2. Hardware e-stop / support the arm. Herdr detach and SSH drop are not stops.
3. Memory-subsystem failures were logged 2026-09-06 with no repair record. Do not qualify a policy on this host until that is closed or the controller moves.
4. Observation-only path that does **not** call follower `connect()`. Cameras via the Orbbec venv; joints only if a future read-only bus exists.

Exit: written ownership checklist. No motion.

## Phase 1 — cameras and encoder (no arm)

1. Start dual publishers; grab wrist + overview JPEG; record serials (305 `CV2L761000FA`, 336L `CPC6463000PZ`).
2. Measure actual FPS, JPEG size, and receipt age (ZMQ uses receive time, not exposure).
3. Crop/rectify 848×480 → the optic encoder size without stretching. Freeze a rig id.
4. Run hop-probe / `step_vision(drive_kc=False)` on real table vs black. Pass = scored DN Hz moves with the cube in the real image.

Exit: `reports/hw-cam-hop.json` with mean DN Hz cube vs black. Still no `fly_picked`.

## Phase 2 — measure the plant (supervised, slow)

Calibrate zeros the usual way (hand-fold, close gripper, Enter). Then, with support and 15–30°/s caps:

1. Jaw curve: command gripper through 0 → −270° and back; measure opening mm (ruler/caliper). Save open and close tables.
2. TCP: tape a tool origin; command known joint poses; record mm in a table frame. Fit FK. IK must keep_level or fingers_down as explicit flags, not lab defaults assumed true.
3. Table height and a keep-out box. Software clip is not a certified limit (`joint_limits` in the driver are clip bounds).

Exit: `data/calibration/b601-v1.json` (new identity). Simulator SIM-001/002 consume this file.

## Phase 3 — Cartesian gateway (shadow then tiny moves)

New adapter, not MCP JSON on the serial port:

`observe` → validate joints + two JPEGs (age, finite, names) → crop LIF → `unpack` → IK to seven degrees → **log** the goal.

Shadow: motors disabled or held, commands not sent. Inspect jumps, missing `wrist_yaw`, mm/degree mixups.

Armed: one axis, millimetres, far above the table. Then open-gripper hover. Never send incomplete `.pos` dicts (driver fills `wrist_yaw=0`).

Exit: N logged shadow episodes; M slow hover moves with measured TCP error.

## Phase 4 — pick task definition

Same gates as the lab, but measured, not welded:

- Grasp: independent evidence (not “jaws closed”). Until contact sensors exist, use a human key + cube-height from overview depth/RGB, labeled estimated.
- Lift: cube center ≥ 100 mm above table for ≥ 2 s.
- Level: tool or cube face within 5° (measure; do not use sim `tcp_level`).
- Release: scripted open, same as lab cleanup — still lab routing, not `fly_picked`.

Compare against teleop and ACT only on this task, not cube-into-cup.

Collect ~whole-episode demos at 15 Hz (wrist+overview JPEG, joints, commanded TCP, IK joints). Split by session, not frames. Keep failures.

## Phase 5 — train on the real contract

1. Domain-randomize lab cameras to the measured rig (SIM-005–007) and re-distill U if the encoder changes.
2. Optional: few-shot operant on real terminal R only after Phase 3 is safe.
3. Freeze g, calibration hash, period, and evaluator before scoring.

## Phase 6 — supervised demo

Predeclare poses. Count every attempt. Record both cameras + DN rates. Say if release is scripted. Do not show Mac `loop-da2` as the hardware result.

Proposed gate for a **supervised** demo (not unattended use): 20 attempts, ≥19 complete successes, zero out-of-box commands, zero e-stop interventions. Change the number when Phase 2 envelope is known.

## Where Phase 0–1 run

USB and the Orbbec SDK only exist on isengard. SSH from this Mac is how a process is **started** there, not a second way to open `/dev/ttyACM0` or the Geminis.

| Work | Where | Why |
|---|---|---|
| Phase 0 ownership (`lsusb`, `fuser`, process list) | **isengard process** (invoked over SSH) | Device nodes are local to that kernel |
| Phase 1 JPEG grab | **isengard**, `orbbec/.venv` + `scripts/hw_orbbec_grab.py` | pyorbbecsdk talks to USB; grab then **exits** (no standing ZMQ publisher) |
| Crop LIF hop-probe | **this repo on the Mac** | MaleCNS crop (~59k) lives here; two JPEGs are cheap to copy; isengard has no fly-brain tree |
| Herdr pane | Optional log tail | Not an e-stop; not required for Phase 0–1 |

Do not run `herdr --remote` from an agent for this. Do not `connect()` the follower. `--from-dir` replays saved `wrist.jpg`/`overview.jpg` without cameras.

First live grab (2026-09-13): serials matched (305 `CV2L761000FA`, 336L `CPC6463000PZ`). Wrist ~13.5 FPS, overview ~15.8 FPS (pipeline start included). Overview was underexposed. Stub crop hop was green (`cube_dn≈14` vs black 0) — that is not the 59k plant. Real crop: R1 Hz ≈ 60, **scored DN Hz = 0**, `ok=false`, `fly_picked=false`. Photoreceptors saw the table; the DN bus did not.

```sh
# from controller/
python scripts/hw_phase01.py --phase 0 --host isengard.local --out ../reports
python scripts/hw_phase01.py --phase 01 --host isengard.local --out ../reports
python scripts/hw_phase01.py --phase 1 --from-dir ../reports/hw-phase1-frames --stub
```

## Immediate next step

Phase 0–1: `scripts/hw_phase01.py` (cameras only). SIM-005/006 crop is in `rebot_adapter/camera_contract.py`.
