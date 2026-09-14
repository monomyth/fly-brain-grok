# Simulator gaps vs the isengard B601

Backlog for **this** kinematic MaleCNS lab (DN bus → tool-frame Δx/Δy/Δz + grip). Not a claim that transfer works.

Checked 2026-09-13 on isengard without enabling motors: `/dev/ttyACM0` (Damiao CAN bridge), `/dev/ttyUSB0` (StarArm102 leader), Gemini 305 `2bc5:0840` and Gemini 336L `2bc5:0807` on USB, no robot-control process running. Codex in Herdr session `grok-fly-brain` (agent `b601`) wrote `/tmp/grok-b601-ops.md` from the `~/robotics` tree. Driver facts below were also read from `lerobot/src/lerobot/robots/rebot_b601_follower/`.

## How far off the current model is

`g-distill.npz` + `run_dn_bus.py` talks to **ReBotMCP**: `rebot_servo_tcp` (mm, `keep_level` / `fingers_down`), `rebot_set_gripper` (opening mm), `rebot_capture_view` 160×120 JPEG Front+Gripper, and a boolean `cube.attached`. The physical follower accepts **seven absolute joint degrees** on CAN. There is no MCP, no TCP IK, no jaw-mm map, no attach flag. Sending `{dx_mm, dy_mm, dz_mm, dgrip_mm}` into `send_action` drops those keys and can command `wrist_yaw.pos = 0`.

Drop-in transfer is not realistic. The optic crop can ingest real JPEGs after a camera contract. The plant cannot.

## Simulator work

Offline hardware-profile: `controller/rebot_adapter/b601.py` + `camera_contract.py`. Default live path is still Mac MCP TCP (`run_dn_bus.py`, FakeRobot, Front/Gripper 160×120). This plant does not open `/dev/ttyACM0` or call `enable_all()`.

- [x] **SIM-001 Cartesian vs joints.** Offline B601 plant with CAN order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper`. `HardwareAdapter` round-trips TCP Δmm ↔ those seven names via URDF FK / DLS IK. Mac MCP TCP path unchanged. **Still needed:** Phase 2 measured FK table (`data/calibration/b601-v1.json`).
- [x] **SIM-002 Gripper units.** Hardware aperture 0° shut → −270° open; lab 0–90 mm. Versioned **fake-v0** linear curve (`data/calibration/b601-fake-v0.json`, `measured: false`). Mixing mm and degrees raises `UnitMixError` and does not clip a positive mm value to 0° closed. FORCE_POS ratio 0.07 is recorded, unused until measured. **Still needed:** caliper open/close hysteresis.
- [x] **SIM-003 No magic attach.** Hardware-profile cube has `grasp_estimated` / `grasp_label="estimated"` only — no `attached` oracle. Mac lab FakeRobot AABB attach unchanged.
- [x] **SIM-004 keep_level / floor.** `send_action` moves joints independently (no implicit keep_level). Wrist-flex unlevels the tool; TCP below table clearance is a collision and is not applied. IK `keep_level` / `fingers_down` are explicit adapter flags. Home is not collision-free. **Still needed:** measured table height / keep-out.
- [x] **SIM-005 Camera contract.** Hardware capture is 848×480. `crop_rectify_to_encoder` center-crops 16:9→4:3 then scales to 160×120 (no stretch). Lab Front/Gripper 160×120 `apply:false` unchanged. **Still needed:** measured rig crop or retrain on real aspect.
- [x] **SIM-006 Names and mounts.** Hardware names `wrist`/`overview`; lab `Gripper`/`Front`. Missing or swapped names raise `CameraContractError` (no first-feed fallback). `capture_rgbs` refuses a mismatched `camera` field.
- [x] **SIM-007 Time.** Timed frames log `capture_time`, `receive_time`, `command_time`, `age_s`. Plant can inject 100–500 ms age and two-camera skew. Timestamps are still host-side, not exposure.
- [x] **SIM-008 Missing feedback → zero.** `present_pos(strict=False)` paints 0° on a dropped joint; `get_joint_positions(strict=True)` raises. Adapter withholds rather than commanding the painted folded-zero. Incomplete `.pos` dicts are not zero-filled on the adapter path.
- [x] **SIM-009 connect() enables torque.** `connect` / `arm` / `command` / `hold` / `home` are separate. `connect()` does not enable torque or move. Home interpolates all joints (gripper shut) at 20°/s and is not a collision-free plan. No `enable_all()` call.
- [x] **SIM-010 Reset is motion.** Hardware-profile `reset`/`home` drive joints toward zero and do **not** respawn the cube. Lab MCP `rebot_set_cube` / cleanup still may.
- [x] **SIM-011 Dynamics.** Minimal gravity, table rest, and tilt/opening slip. Boolean attach is not the hold oracle. Not MuJoCo; not identified compliance.
- [x] **SIM-012 Cadence.** Frozen `COMMAND_HZ=2`, `CAMERA_HZ=15` on `PlantClock` / `iter_schedule`. Camera and command events are independent; one LIF tick is not one camera frame.
- [x] **SIM-013 Live photometry.** Plant overview mean ≈ 7/255, wrist ≈ 72/255. Orange cube is painted on **wrist only** (live `front_r1` was 0). Default CropLIF gain 1.5 is silent on the real 59k crop; `hop_search` walks to 1.62. Stub crop still hops at 1.5. **Still needed:** 336L exposure, measured rig crop.

## Not simulator bugs (leave them)

- T1 MN rates stay a log, not the joystick.
- Live MaleCNS plant stays the optic crop, not 166k cells.
- Overlay / k-NN stay `lab_picked`.
