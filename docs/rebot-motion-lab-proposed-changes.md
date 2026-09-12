# Proposed changes to rebot-motion-lab (for MaleCNS cube pickup)

Hand this to Codex as a scoped change list. **Do not put the fly connectome, LIF, or Brian2 inside this repo.** The lab stays a kinematic B601-DM simulator. The fly-brain project is an external MCP/servo client.

Target app version: **1.6**. Keep the v1.5 motion rules in `HANDOFF.md` / `AGENTS.md` (immediate sliders, no `ManualMotion`, no mid-drag `AppModel` publishes).

## Goal

Support this experiment without turning the lab into a physics engine:

1. A cube sits on the ground plane.
2. Front / Top cameras show arm + cube (presets already exist).
3. An external controller (MaleCNS adapter in `fly-brain`) can see the scene, move the arm, close the gripper, **kinematically pick up** the cube, lift it, and **hold the cube’s top face parallel to the ground**.

Stock 1.5 cannot do (1) or (3). `(2)` is already `rebot_set_view` `Front` / `Top`. Missing for the fly loop: RGB frames, cube state, grasp attach, tool orientation, and a servo path that does not wait on `MotionPlayer`.

## Non-goals

- No MuJoCo / RealityKit rigid-body dynamics, friction, or payload mass.
- No self-collision, cube-vs-link collision, or bouncing.
- No hardware / CAN.
- No MaleCNS, FlyWire, or neural model in this repository.
- Do not restore slider lag or bind SwiftUI `Slider` to a republishing `ObservableObject`.
- Do not make `RobotScene` observe `AppModel` for pose.
- Web remains display-only (no MCP). Cube + cameras should still match visually.

## Architecture (keep)

```
external client (Grok / Codex / fly-brain)
        │  MCP stdio → ReBotMCP → Unix socket
        ▼
   AppModel ──applyPose──► RobotViewport (RealityKit)
        │
        ├── FloorConstraint (robot meshes vs plane)  [keep]
        └── SceneObjects + GraspAttach               [add]
```

Control modes:

| Mode | Use | Path |
|---|---|---|
| `scripted` (default, 1.5) | Presets, IK, sequences, MCP pose tools | `MotionPlayer` quintic, 60°/s |
| `servo` (new) | Closed-loop fly / external controller | Same path as sliders: `applyInteractive` / `applyPose` immediately, floor-limited |

Overlapping scripted commands still reject. Servo replaces the current pose in-place (like a slider).

---

## PR 1 — Scene cube (visual + state)

**Why:** Codex can “insert a cube on the ground” and Grok can look at it.

### Behavior

- One scene object, id `cube`, axis-aligned box, default **40 × 40 × 40 mm**, color distinct from the arm (e.g. `#c45c26`).
- Default pose: center on the floor in front of the base, **not intersecting the folded arm**. Suggested: `x = 280 mm, y = 0, z = 20 mm` (center; bottom on `FloorConstraint.height`).
- Cube is a RealityKit `ModelEntity` sibling of the robot under `world`, **not** parented to a link until grasp (PR 2).
- Reset (`⌘R` / `rebot_playback reset`) returns the cube to its **spawn pose**, unattached.
- Slider / playback FPS: cube transform updates in `applyPose` / the existing frame callback. Do not `@Publish` cube pose every tick.

### MCP

| Tool | Purpose |
|---|---|
| `rebot_set_cube` | Place or resize. Args: `x_mm`, `y_mm`, `z_mm` (center), optional `size_mm` (scalar or `{x,y,z}`), optional `yaw_deg`. Omit fields = leave unchanged. `present: false` hides/removes. |
| `rebot_get_state` | Add `objects.cube`: `{present, attached, size_mm, center_mm, yaw_deg, top_normal}` |

Reject poses that put the cube below the floor (bottom face `< FloorConstraint.height`). Allow `z` so the cube sits **on** the plane.

### Files (native)

- `Sources/RobotCore/SceneObject.swift` (pure data + floor clamp; unit-testable)
- `Sources/ReBotMotionLab/RobotViewport.swift` — add/remove/update cube entity
- `Sources/ReBotMotionLab/AppModel.swift` — cube state; reset restores spawn
- `Sources/ReBotMotionLab/MCPControl.swift` + `ControlCatalog.swift`
- `Tests/RobotCoreTests/SceneObjectTests.swift`
- `scripts/test-mcp.py` — set cube, read state, reset restores

### Web

- `web/app/robot-scene.tsx` — same default cube and `options.cube`
- No MCP.

### Tests

- Default cube bottom is on the plane, center z = half height.
- `rebot_set_cube` below the floor is rejected; pose unchanged.
- Reset restores spawn; cube `attached == false`.
- Smoke slider harness: cube present, `main_model_notifications` still low, 0 lagging frames.

---

## PR 2 — Kinematic grasp (pickup without physics)

**Why:** Closing the gripper must make the cube follow the TCP. Otherwise “pick up” is theater.

### Grasp rule (deterministic, documented)

Cube **attaches** when all are true:

1. Cube `present` and not already attached.
2. Cube center is inside the **grasp AABB** in `end_link` / fingertip frame.
3. Gripper is **closing** (new `gripper_mm` < previous) and opening ≤ **grasp_close_mm** (default **25 mm**; tunable constant).

Cube **detaches** when:

- Gripper opening ≥ **grasp_release_mm** (default **40 mm**), or
- `rebot_set_cube` / reset / `attached: false`.

While attached:

- Cube world pose = `end_link` transform × **grasp offset** captured at attach time (so the cube does not snap to TCP origin).
- Update that pose inside `RobotViewport.applyPose` / `advance`, same as fingers — **not** via SwiftUI.

### Grasp AABB (v1)

In the closed-fingertip / `end_link` frame (TCP is already “closed fingertip center”):

- X/Y: ±(cube_half + 8 mm) of TCP
- Z: ±(cube_half + 5 mm)

Exact numbers can be calibrated against the finger STL; put them in `RobotCore` as named constants. Add a debug overlay flag later if needed; not required for v1.

### Floor while held

Extend floor limiting so the **cube AABB** also stays above the plane when attached. If a lift/swing would push the cube through the floor, clamp like robot meshes (stop at contact, status `Stopped at base plane`).

Do **not** collide cube vs links in v1. Document that the arm can pass through an unattached cube.

### MCP / state

- `objects.cube.attached: bool`
- `objects.cube.center_mm` always **world**
- `rebot_set_gripper` / `rebot_move_joints` / sliders all go through the same attach/detach function
- Optional `rebot_set_cube` field `attached: false` to force drop

### Tests

- Open gripper, cube in grasp volume, close to 20 mm → `attached == true`; TCP +100 mm Z → cube center rises ~100 mm.
- Open to 50 mm → `attached == false`; cube stays at drop pose (on or above floor; if below, clamp onto plane).
- Cube outside AABB, close gripper → not attached.
- Held cube commanded through the floor → motion stops; cube still attached; `minimum` height accounts for cube.
- Sequence playback: waypoint close/open attaches/releases.

### Files

- `Sources/RobotCore/Grasp.swift`
- `Sources/RobotCore/FloorConstraint.swift` — optional cube hull when attached
- `Sources/ReBotMotionLab/RobotViewport.swift` — reparent cube entity to `end_link` on attach, back to `world` on detach
- `Tests/RobotCoreTests/GraspTests.swift`

---

## PR 3 — Tool orientation + “hold parallel to ground”

**Why:** 1.5 IK is position-only (`ControlCatalog`: “Orientation is unconstrained”). Holding a cube level is a wrist constraint.

### State

Add to `rebot_get_state`:

```json
"tcp_mm": {"x": ..., "y": ..., "z": ...},
"tcp_rpy_deg": {"roll": ..., "pitch": ..., "yaw": ...},
"tcp_level": true
```

Define `tcp_level` as: the tool’s grasp axis keeps the **attached cube top face** within **5° of world +Z**. If unattached, `tcp_level` means the gripper approach axis is within 5° of world −Z (vertical pinch), which is the same offset used at spawn.

Use `end_link` rotation from existing `Kinematics.transforms`.

### IK

Keep `rebot_move_to_position` as position-only (compat).

Add:

| Tool | Purpose |
|---|---|
| `rebot_move_to_pose` | Cartesian target with orientation. Args: `x_mm`, `y_mm`, `z_mm`, optional `roll_deg`, `pitch_deg`, `yaw_deg`, optional `keep_level: true`. |

When `keep_level` is true (the pickup experiment):

- Solve DLS on 6 joints with an extra cost/constraint on roll/pitch so cube top (or gripper, if unattached) stays world-level.
- If no solution within **2 mm** position and **5°** level, reject; pose unchanged (same contract as current IK).

Suggested pickup sequence the external client will run:

1. `keep_level` pre-grasp: above cube, `z = cube_top + 40 mm`
2. Descend to grasp height
3. Close gripper (PR 2 attaches)
4. Lift to `z ≥ 150 mm` with `keep_level: true`
5. Hold (servo or stopped playback)

### UI

- TCP readout: show RPY as well as XYZ (15 Hz cap unchanged).
- IK panel: checkbox **Keep tool level**.

### Tests

- From Ready, `keep_level` move to `(280, 0, 200)` → `tcp_level == true`, error < 2 mm.
- After grasp + lift, cube `top_normal` dot world Z > 0.996 (≈5°).
- Unreachable level pose returns an error; joints unchanged.

### Files

- `Sources/RobotCore/Kinematics.swift` — `solve(target:orientation:)` or `solveLevel`
- `Tests/RobotCoreTests/RobotCoreTests.swift`
- MCP catalog + `MCPControl`

---

## PR 4 — Camera frames for MaleCNS (and Grok)

**Why:** Front/Top already exist. The fly adapter needs **pixels**, not a camera enum. MCP today has no screenshot tool.

### Behavior

- `rebot_capture_view` snapshots the **RealityKit scene camera** (`cameraEntity`), not the window chrome / SwiftUI sliders.
- Optional `camera`: `Orbit` | `Front` | `Top` (default: current). Capturing Front/Top must not permanently steal the user’s Orbit view unless `apply: true`.
- Size: `width`/`height` default **320×240**, max **640×480**. JPEG quality 0.7.
- Return JSON: `{camera, width, height, jpeg_base64, cube_in_view, tcp_in_view}` plus the usual `state` (keep payload under the 1 MiB IPC cap; 320×240 JPEG is fine).
- Implementation: offscreen `ARView` snapshot / `ImageRenderer` of the viewport. Reuse existing `captureImage` smoke path if it already freezes a RealityKit frame; do not rebuild the STL hierarchy.

### Views

Keep current presets; optionally offset **Top** look-at to include both base and default cube (`target.x ≈ 0.18` m) so “overhead of arm and cube” is true without the user panning. If you change Top/Front framing, do it behind `cameraRevision` and document it — Grok’s “establish Front/Top” then just calls `rebot_set_view`.

### Tests

- Capture Front and Top with cube present: JPEG decodes, dimensions match, `cube_in_view` true at default spawn.
- Capture does not change joints, gripper, or cube pose.
- Capture with `apply: false` leaves `view.camera` unchanged.

### Files

- `RobotViewport.captureJPEG(camera:size:)`
- `ControlCatalog` + `MCPControl`
- Integration: `scripts/test-mcp.py` writes a JPEG under `Verification/`

---

## PR 5 — Servo mode (closed-loop client)

**Why:** 1.5 MCP pose tools start `MotionPlayer`, return `accepted`, and **reject** the next command until stopped. A 20–50 Hz fly loop cannot use that.

### Behavior

- `rebot_set_control_mode` : `scripted` | `servo`
- Entering `servo` stops playback, ends slider tracking, status `Servo mode`.
- Leaving `servo` freezes the current pose (`scripted` + `playback: stopped`).
- In `servo`, **disable** `rebot_move_joints` / `move_to_position` / presets / sequence play (clear error: “stop servo first”). Allow `rebot_get_state`, `rebot_set_view`, `rebot_capture_view`, `rebot_set_cube`.

New tool:

| Tool | Purpose |
|---|---|
| `rebot_servo_joints` | Immediate pose, same as a slider tick. Args: `joints_deg` (6) and/or `gripper_mm`. Floor-limited; returns **actual** pose (may be clamped). |

Optional (nice for the fly decoder):

| Tool | Purpose |
|---|---|
| `rebot_servo_tcp` | Immediate position-only or `keep_level` IK from current joints; apply result immediately (no quintic). |

Rate: no MotionPlayer. Clients poll/capture as fast as they want; the app applies the latest command on the main actor. If a command arrives during `applyPose`, it replaces the target (last-wins). Still floor-limited.

### Safety

- Same joint/gripper limits as 1.5.
- Servo does not bypass fingertip floor stops.
- MCP off / page leave: exit servo, stop (same as turning MCP off today).

### Tests

- Scripted move then servo command without mode change → error.
- Servo 100 joint updates: `current` matches last allowed command; smoke-style 0 lagging requirement; cube attach still works if gripper closes in servo.
- `test-mcp.py` servo round-trip < 50 ms local (best-effort assert, record timing).

### Files

- `AppModel.controlMode`
- `MCPControl.handle`
- `AGENTS.md` — servo uses the **interactive** apply path, not `ManualMotion`

---

## PR 6 — Docs, version, integration tests

- Bump `ControlCatalog.version` to **1.6**.
- `README.md`: kinematic + **one scene cube**, kinematic grasp, no payload physics.
- `MCP.md`: new tools, servo workflow, capture size limits, “fly client lives outside this app”.
- `CHANGELOG.md`, `AGENTS.md` (cube entity, servo = interactive apply).
- `scripts/test-mcp.py`: all new tools; keep the existing 12-tool checks.
- `Verification/README.md`: cube/grasp/capture/servo.
- Example: `examples/pick-cube.json` sequence (scripted fallback): Ready → pre-grasp → close → lift, for humans without the fly.

Web README: cube is visual-only in the browser.

---

## Suggested implementation order

```
PR1 cube → PR2 grasp → PR3 level IK → PR4 capture → PR5 servo → PR6 docs
```

PR1+PR2+PR3 are enough for **Codex/Grok to pick up the cube with scripted MCP** (no fly). PR4+PR5 are what the MaleCNS adapter needs.

## External fly-brain client (not this repo)

Once 1.6 ships, `fly-brain` should:

1. `rebot_set_cube` default spawn; `rebot_set_view` Top; `rebot_set_control_mode servo`.
2. `rebot_capture_view` → sample JPEG onto MaleCNS R1–R6 (same idea as doomfly).
3. LIF step; decode a **small** output (e.g. Δx, Δy, Δz, dGripper, keep_level) — not 6 raw motors unless a trained readout exists.
4. `rebot_servo_tcp` / `rebot_servo_joints`; attach happens inside the lab when the gripper closes in the AABB.

The lab is the body. The connectome is the external controller. Do not merge them.

## Acceptance for the stated experiment

A scripted (no-fly) smoke that 1.6 must pass:

1. Spawn default cube; Top and Front captures show arm + cube.
2. Open gripper; `move_to_pose` keep_level above cube; descend; close to 20 mm → `attached`.
3. Lift to z ≥ 150 mm keep_level → cube rises with TCP; `tcp_level` true; cube top parallel to ground within 5°.
4. Hold 2 s (playback stopped / servo idle); cube stays attached and level.
5. Open gripper → cube rests on the plane at the drop XY.

MaleCNS is optional after that: same steps driven by PR5 servo + PR4 frames instead of scripted IK.
