# Plan: MaleCNS fly brain controlling the ReBot arm

Date: 2026-09-11 evening. Architecture plan. The grok lab is the body. MaleCNS, LIF, downloads, and training stay outside `rebot-motion-lab-grok`.

**Scope:** `/Users/monomyth/code/grok/fly-brain` only. Public MaleCNS / Janelia / doomfly sources are fine. Atlas: [fly-connectome.grok.me](https://fly-connectome.grok.me) ([/robotics](https://fly-connectome.grok.me/robotics)). Arm-transfer notes: `grok_report.pdf` (dopamine), `grok_report-2.pdf` (embodiment).

**One-line goal:** Front + Gripper pictures into MaleCNS optical neurons; a **named descending-neuron bus** commands the arm; lab IK is the **VNC unpacker**; **dopamine stamps which camera snapshots predicted a good grasp**, not how the B601 bends.

**Motor rule:** scored fly command = thin DN vocabulary (tool-frame Δx/Δy/Δz, approach, grip, lift, abort), unpacked by `rebot_servo_tcp` / gripper. That unpacker **is** the VNC analog. Do **not** train 166k cells as a 6-DoF policy. Do **not** put T1 `vnc_motor` rates on the live joystick (wrong plant; this LIF does not spike them). Overlay / quintic / optical k-NN are not the fly. A handwritten mode skeleton (search → align → close → lift → abort) is **lab routing**, labeled as such. T1 MN rates are a **reachability log**.

**Learning rule:** teacher/imitation may learn **how to move** on the DN bus. Dopamine is **not** RPE through the arm. At **trial end**, success → PAM-like pulse, fail → PPL1-like pulse, only on existing KC→MBON synapses. MBON may **gate** DN channels (grip gain, approach, abort). It must not emit joint targets. Dense “closer to cube” reward is forbidden.

**Not claimed:** the fly already knows a B601; 166k LIF will pick; Doomfly’s DA gates will pass.

**Earlier drafts**

- 2026-09-09: skipped the VNC; optical k-NN scored as fly.
- 2026-09-11 morning: **leg MN rates → TCP**. MaleCNS includes the cord so you can *trace* eye → DN → MN. The B601’s cord is already lab IK. Audit: `|W|^k` hits T1 MNs at hop 4; LIF Hz stays 0. Keep MN groups indexed; do not steer with them.

---

## 1. What already exists

### Body (done)

`rebot-motion-lab-grok`: cube, Front/Gripper/Top/Orbit JPEG `apply:false` offscreen, servo TCP/joints, kinematic attach, `keep_level`, teacher script. IPC `rebot-motionlab-grok-<uid>`. No MaleCNS in Swift.

### Brain (partial)

Cache, LIF, Front+Gripper encoder, optical k-NN pick on one spawn (`phase2-ungated.npz`) — **non-fly baseline**. `vnc_motor` / `leg_mn_*` groups now indexed (708 / 173 T1). `decode_leg_mn` exists but is **not** the live scored path. Audit: R1 fires by ~100 steps; DN and T1 MN Hz = 0 through 500 steps at gain 0.02–0.275.

v1.0: 708 `vnc_motor`; T1/T2/T3 ≈ 173/175/152; `fl/ml/hl` = 135/116/130.

---

## 2. Architecture (after fly-connectome.grok.me + reports)

```
  Front cam  → R1–R6 / R8 / (later T4-T5, LC10, LPLC2 if they fire)
  Gripper    → separate wrist channel (midline / contact), not concat into one ViT
  Proprio    → TCP, grip (ascending analog)
           │
           ▼
  MaleCNS LIF (full graph, many steps, reset once per episode)
           │
           ├─ named DN rates  →  DN mixer (~8–12 channels)   ← scored fly bus
           ├─ KC / MBON11     →  gain / abort gate after DA
           └─ vnc_motor T1    →  log only (reachability)
           │
           ▼
  lab IK unpacker  =  VNC analog   rebot_servo_tcp keep_level + gripper
  mode skeleton    =  lab routing  search/align/close/lift/abort (labeled)
  giant-fibre      =  abort: open + retreat, not a planner
```

**Embodiment map** (from the reports; keep)

| Fly | B601 lab |
|---|---|
| Compound eyes | Front camera |
| Tarsi / foreleg tap | Gripper camera |
| Chordotonal | TCP + gripper width |
| Descending neurons | 8–12 dim bus, **not** 7 joint cmds |
| VNC motor circuits | Lab IK / attach / floor |
| Neck connective | Python ↔ MCP |
| Giant fibre | Abort |

CX / heading ring: skip (mounted arm). Compass = tool-frame error.

### DN bus (the thing you train or read)

| Channel | Fly analog | Arm |
|---|---|---|
| dx, dy, dz | DNg-like steering | Tool-frame offset |
| approach | loom-gated reach | Speed along grasp axis |
| grip | gnathal ingest | Close / hold / open |
| lift | takeoff / load DNs | +z after contact |
| abort | giant fibre | Open + retreat |

Policy never emits joint torques. Unpacker is existing IK.

**Forbidden as scored fly**

- Optical k-NN / ridge on concatenated JPEGs
- Overlay-servo / quintic acting
- T1 MN pool → TCP
- `hop_drive` as substitute for LIF DN spikes
- DA module emitting joints
- Dense per-step distance reward

### Dopamine (from grok_report.pdf)

Flies do not DA-tune femur MNs for “correct extension.” They tag **this snapshot is worth approaching**.

| Signal | Use DA? |
|---|---|
| Joint unpacker / IK | No |
| DN mixer | Gate only, not PPO on Δq |
| “This wrist view means close” | Yes |
| “This front view is the object” | Yes (appetitive after contact) |
| Abort / drop / collision | Yes, aversive PPL1 |
| Mode skeleton | No — hand-specified |

**Pulse at trial end, not every tick**

- success (HANDOFF + lift + hold ≥ 2 s; report used ~300 ms — we keep lab 2 s) → PAM-like
- fail (drop, timeout, collision, pinch air) → PPL1-like

Write only into KC→MBON. Readout = extra bias on **existing DN channels** and mode-transition **gains**. Timing > magnitude (CS–US delay). Optional later: octopamine = search gain; NPF/SIFa = stop after N picks. Those are scalars, not an advantage estimator.

### Observation / timing

Front and Gripper stay **split**. Vision eval: JPEGs → photoreceptors, no cube XYZ. Debug state labeled. Orbit not stolen.

Neural: many 0.1 ms steps per 10–20 Hz servo tick; **no per-frame reset**. Pass A is now **named DN rates** (and log T1 MNs). If DNs stay silent, report that; do not paper with k-NN.

---

## 3. Downloads

Same three feathers in `$MALECNS_HOME=/Users/monomyth/code/data/malecns`. Meta must keep photoreceptors, DN groups, KC/MBON11/PPL101, **and** `vnc_motor` / `leg_mn_*` for the log. CC-BY.

---

## 4. Control loop

Folded → Ready → servo.

Each tick: Front+Gripper `apply:false` → R1/R8 → LIF → **read DN bus** → optional MBON gate → clip → `rebot_servo_tcp` + gripper. Log DN rates, T1 MN rates, KC/MBON/PPL, mode, `acting_map=dn_bus`.

`fly_picked`: kinematic gates (attach, live cube z≥100, tcp_level, hold≥2s); `acting_map=dn_bus` every tick; DN rates varied with the image; overlay/quintic/k-NN/MN-joystick did not act.

`da_learned`: terminal pulse changed KC→MBON **and** that change gates the DN bus (DA-on vs off / shuffled / freeze / erase / silence-MBON). A pick with DA running unused is not DA.

---

## 5. Curriculum

**Phase 0 — Teacher.** Log Front+Gripper + commands. Environment baseline, not fly.

**Phase 1 — DN bus smoke.** Frozen graph. No k-NN. Cube vs black changes **named DN rates**. Those rates move TCP/gripper. Silence those DNs → command collapses. T1 MN Hz logged; silence is a LIF fact, not a reason to swap joystick.

**Phase 2 — Optional small mixer.** If engineered DN→TCP does not pick: linear/MLP on **DN rates** (+ optional proprio), teacher actions. Not JPEG features. Mode skeleton may be **labeled lab**, not `fly_picked`.

**Phase 3 — Dopamine.** After DN bus moves the arm. Terminal PAM/PPL only. MBON gates DN channels. Same honesty gates as before (on/off, yoked, freeze, erase). No dense shaping.

**Phase 4 — Eval table:** teacher; DN bus DA off/on; shuffled DA; black frames; DN silenced; MBON silenced after train; T1 MN log; optical k-NN and overlay as **non-fly** rows.

---

## 6–7. Layout and order

`controller/malecns_cache`, `runtime` (LIF, encoder, **DN decoder**, KCToMBON), `rebot_adapter`, tests.

1. Keep VNC groups (already indexed) as log.
2. DN hop audit: R1/R8 → named DNs (same honesty as the MN audit).
3. `acting_map=dn_bus` live path, `--no-overlay`, Front/Gripper offscreen, Folded→Ready.
4. Pick attempts only after DN Pass A.
5. Optional DN mixer.
6. Terminal DA gating DN gains; ablations.

---

## 8. Risks

- DN→TCP is still engineered (atlas: name the bus, don’t claim identity).
- Retina is a JPEG proxy.
- LIF may never light DNs either — then the honest result is “graph has a path, this neuron model does not carry it.”
- Kinematic grasp ≠ hardware.
- Report mode skeleton as lab.

---

## 9. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Brain | `fly-brain/controller` | Lab is body |
| Scored motor | Named DN bus | Atlas + grok_report-2: DNs are the neck vocabulary |
| Unpacker | Lab IK | That **is** VNC; not T1 MN→TCP |
| T1 MNs | Log only | LIF silent; wrong plant for a 6-DoF arm |
| Cameras | Front and Gripper **split** | Eyes vs tarsi; do not concat |
| Modes | Hand skeleton, labeled lab | DA must not discover close vs lift |
| Learning | Imitation on DN bus; DA stamps snapshots | grok_report.pdf: not RPE through the arm |
| Reward | Terminal PAM / PPL1 | Not per-step distance |
| First “it works” | DN rates move with vision **and** move the arm | k-NN pick is not success |

---

## 10. Open questions

1. Which MaleCNS DN types map to which bus channels (DNg13 vs DNp20/DNa02 already in decoder) — measure who actually fires before freezing the list.
2. Crop optic lobe + DN set for speed vs full graph for published eval.
3. Hold 2 s (lab contract) vs ~300 ms in the report — keep 2 s.
4. Octopamine / NPF scalars later, not in Phase 1.

---

## References

- Atlas: https://fly-connectome.grok.me/ https://fly-connectome.grok.me/robotics
- MaleCNS: https://male-cns.janelia.org/
- Cell 189 (18) Berg et al. 2026
- doomfly: https://github.com/nftechie/doomfly
- Huang, Luo et al. 2024: https://doi.org/10.1038/s41586-024-07819-w
- Lab MCP: `rebot-motion-lab-grok/MCP.md`
- User notes: `grok_report.pdf`, `grok_report-2.pdf`
