# Simulator gaps vs the isengard B601

Backlog for **this** kinematic MaleCNS lab (DN bus → tool-frame Δx/Δy/Δz + grip). Not a claim that transfer works.

Checked 2026-09-13 on isengard without enabling motors: `/dev/ttyACM0` (Damiao CAN bridge), `/dev/ttyUSB0` (StarArm102 leader), Gemini 305 `2bc5:0840` and Gemini 336L `2bc5:0807` on USB, no robot-control process running. Codex in Herdr session `grok-fly-brain` (agent `b601`) wrote `/tmp/grok-b601-ops.md` from the `~/robotics` tree. Driver facts below were also read from `lerobot/src/lerobot/robots/rebot_b601_follower/`.

## How far off the current model is

`g-distill.npz` + `run_dn_bus.py` talks to **ReBotMCP**: `rebot_servo_tcp` (mm, `keep_level` / `fingers_down`), `rebot_set_gripper` (opening mm), `rebot_capture_view` 160×120 JPEG Front+Gripper, and a boolean `cube.attached`. The physical follower accepts **seven absolute joint degrees** on CAN. There is no MCP, no TCP IK, no jaw-mm map, no attach flag. Sending `{dx_mm, dy_mm, dz_mm, dgrip_mm}` into `send_action` drops those keys and can command `wrist_yaw.pos = 0`.

Drop-in transfer is not realistic. The optic crop can ingest real JPEGs after a camera contract. The plant cannot.

## Simulator work

- [ ] **SIM-001 Cartesian vs joints.** Add a B601 joint plant (shoulder_pan … gripper) with the same names and CAN order. Keep the MCP TCP path as the Mac lab. A hardware adapter must IK/FK; the sim should round-trip the same seven channels so we can test the adapter offline.
- [ ] **SIM-002 Gripper units.** Hardware: closed = `0°`, open toward **−270°**, FORCE_POS ratio 0.07. Lab: 0–90 mm opening. Measure a real aperture curve (open and close, hysteresis). Until then, a versioned fake curve in the sim that **fails closed** if a caller mixes mm and degrees.
- [ ] **SIM-003 No magic attach.** Lab weld is pads-in-AABB. Hardware has no contact boolean. Stop using `attached` as an oracle in any plant meant to match the robot. Drive pinch/lift from images + optional estimated grasp, labeled as estimated.
- [ ] **SIM-004 keep_level / floor.** Lab IK holds tool orientation and clamps TCP z (~42 mm fingers_down). Hardware `send_action` moves joints independently. Simulate unlevel wrist, table collision, and a missing orientation constraint.
- [ ] **SIM-005 Camera contract.** Real dual stack: 305 wrist + 336L overview, **848×480** YUYV, ZMQ JPEG, 15 Hz recorded (requested 30). Lab: Front/Gripper **160×120**, `apply:false`. Do not stretch 16:9 into 4:3. Add a 848×480 capture path and a documented crop/rectify to the optic encoder, or retrain the crop on the real aspect.
- [ ] **SIM-006 Names and mounts.** Lab `Front`/`Gripper` ≠ hardware `overview`/`wrist`. Wrong-name fallback on the ZMQ camera can silently swap feeds. Fail if the expected name is missing.
- [ ] **SIM-007 Time.** Hardware timestamps are host publish/receive, not exposure. Dataset time is `frame_index/fps`. Lab ticks are LIF-bound (150 steps) and much slower than 15 Hz. Log capture vs receive vs command time in the sim; inject 100–500 ms age and two-camera skew.
- [ ] **SIM-008 Missing feedback → zero.** Follower `_present_pos(strict=False)` substitutes `0.0` when state is missing. `get_joint_positions` is strict. Sim MCP never lies that way. Add a fault: drop a joint, paint 0°, and require the controller to withhold commands (today it would treat folded-zero as real).
- [ ] **SIM-009 connect() enables torque.** Lab MCP connect does not move the arm. Hardware `connect()` → `enable_all()`. Model connect / arm / command / hold / home as separate ops. Home interpolates to all-zero (gripper shut) at ~20°/s and is not collision-free.
- [ ] **SIM-010 Reset is motion.** Lab reset respawns the cube. Hardware `/reset` and recording teardown drive joints to calibrated zero. No object teleport. Sim hardware-profile must not respawn on reset.
- [ ] **SIM-011 Dynamics.** Lab is kinematic; cube attach is Boolean. Need gravity, compliance, slip, and a table. Cube-into-cup teleop success is not a 20 mm level-hold oracle.
- [ ] **SIM-012 Cadence.** Dual demos ran ~14.5 Hz; ACT chunks were hundreds of ms. Crop LIF is slower still. Freeze a policy period (propose 2 Hz command, 15 Hz cameras) and train/eval at that period. Do not assume one sim tick = one camera frame.

## Not simulator bugs (leave them)

- T1 MN rates stay a log, not the joystick.
- Live MaleCNS plant stays the optic crop, not 166k cells.
- Overlay / k-NN stay `lab_picked`.
