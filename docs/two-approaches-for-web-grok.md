# Two ways to let MaleCNS drive a ReBot B601-DM arm

**Purpose of this note.** A local Grok session built a kinematic arm lab and a MaleCNS (adult male *Drosophila* CNS) controller. The user asked that **fly neurons that operate a leg** operate the robot arm, and that **dopamine reinforce correct tasks**. A public atlas ([fly-connectome.grok.me](https://fly-connectome.grok.me), especially `/robotics`) plus two Grok-Build reports (`grok_report.pdf` on dopamine, `grok_report-2.pdf` on embodiment) argue for a different split: **descending neurons as a named action bus**, **VNC as a fast unpacker**, **dopamine as snapshot valence**.

This document states both approaches so an independent model can judge them. It does **not** claim Approach B is proven better at picking a cube. It explains why a local session treated B as a **stronger architectural argument**, and where that may be wrong.

**Please analyze:** which approach better matches (1) the user’s sentence, (2) MaleCNS biology, (3) a kinematic 6-DoF arm that already has IK, (4) the measured LIF facts below. Recommend keep A, keep B, or a hybrid. Flag any cheat that would score a scripted/overlay policy as “the fly.”

---

## Shared facts (do not dispute without evidence)

### Body

- Simulator: ReBot B601-DM, kinematic (not MuJoCo friction). TCP = closed fingertip. Fingers tool −X, open ±Y. Cube 10–90 mm.
- Grasp: pads close on cube volume → attach (HANDOFF). Open +12 mm → drop.
- IK already exists: `keep_level` servo of TCP + gripper. Floor clamp.
- Cameras: Front (scene, Gemini 336L analog) and Gripper/wrist (Gemini 305 analog, ~15° down). Capture can be offscreen so the user’s Orbit view does not jerk.
- Success gates used so far: cube attached, live cube z ≥ 100 mm, TCP level, hold ≥ 2 s.
- Overlay-servo / quintic teacher **can pick**. That is **not** a fly pick.

### Brain

- MaleCNS v1.0: ~166,700 neurons, brain + optic lobes + **ventral nerve cord**. FlyWire female brain has **no VNC**.
- Frozen signed CSR LIF in Python. Swift is body-only.
- Photoreceptors: JPEG luma → R1–R6, chroma → R8 (proxy grid, not a measured eyemap).
- Plasticity already coded: ~4,184 **existing** KC→MBON11 synapses, PPL101 pulse, eligibility KC×MBON. Reward **sign from a task event**, not from PPL Hz (PPL saturates).
- v1.0 annotations: **708 `vnc_motor` cells**; T1/T2/T3 ≈ 173/175/152; subclass front/mid/hind leg 135/116/130. Named pools include Ti flexor/extensor, Tr flexor/extensor, Fe reductor, Ta depressor/levator.

### Measured on this LIF (2026-09-11)

| Condition | R1 Hz | named DN Hz | T1 MN Hz |
|---|---|---|---|
| 3 LIF steps (old eval), reset every frame | ~0 | ~0 | ~0 |
| 100 steps, i_ext on R1 | ~18 | 0 | 0 |
| 500 steps | ~103 | 0 | 0 |
| synaptic_gain 0.02 … 0.275, 50–200 steps | R1 same | 0 | 0 |

- Structural `|W|^k` (unsigned weights): R1 mass reaches **named DNs by hop 2–3**, **T1 motor neurons by hop 4**.
- If **all** R1 spike at once, first-hop current on DN and T1 MN is **0** (optic lobe gets current). So MN/DN need **multi-hop spike propagation**, not hop-1.
- An optical **k-NN** on Front+Gripper summaries (209 logged demo rows, no attached-bit switch) **did pick** once (cube z 20→120 mm, hold 3.5 s). Reviewers rejected several earlier “picks” that were overlay lock-and-lift, attached k-NN, or a grip-threshold map swap. The surviving k-NN is still **imitation of demos**, not DN/MN rates.

**Implication of the audit:** a joystick on **current** T1 MN (or named DN) rates would command **zeros**. That does not decide the *right* joystick after the neuron model is fixed. It does decide that steering with silent MNs **today** is theater.

---

## User’s original ask (the sentence to satisfy)

Pictures into optical neurons. **Motor / DN / MBON activity** controlling the arm. **Dopamine-reinforced.** Explicit follow-up: use **motor neurons that control a fly leg** (MaleCNS includes VNC; FlyWire cannot). Overlay/teacher must not be scored as the fly.

Tension: “leg motor neurons operate the arm” vs “steal the fly’s *split* (named DN bus + spinal unpacker)” from the atlas/PDFs.

---

## Approach A — Leg motor neurons are the joystick

**Claim.** The live command is a function of live **`vnc_motor` rates**, preferably front-leg / T1 (Ti/Tr/Fe/Ta flexor vs extensor pools). Brain DNs may carry current to the cord; they are not a substitute. Lab IK still unpacks TCP; the **numbers** come from MN rates.

**Loop.** Front+Gripper JPEG → R1/R8 → many LIF steps, no per-frame reset → read T1 MN pool contrasts → Δx, Δy, Δz, dGrip → `servo_tcp`.

**Dopamine.** Task events (attach / lift / hold vs pinch-air / drop / timeout) pulse PPL101. Eligibility on KC×MBON11. MBON11 must **bias the MN command**. Otherwise DA is unused.

**Pass/fail.** Cube vs black changes **T1 MN rates**. Silence `vnc_motor` → arm stops. `fly_picked` only if `acting_map=leg_mn` and MN rates varied with the image.

**Why it matches the user.** MaleCNS is unique because of the cord. “Neurons that flex a tibia” are the cells that *operate a leg*. Mapping those rates onto an arm is engineered, but the **source cells** are honest.

**Why it is hard / maybe wrong.**

1. This LIF never spikes those cells (table above). Fixing gain/hops/KC drive might still fail.
2. A fly tibia flexor is not a B601 joint. Six legs, no 6-DoF chain. You are still inventing a map.
3. Flies do not dopamine-tune femur MNs for “correct extension” (see Approach B / PDF). Forcing DA through MN rates may be the wrong computation even if MNs fire.
4. The lab **already** has a VNC analog: IK + attach. Using MNs as a slow outer joystick duplicates the cord on the wrong body.

---

## Approach B — Named DN bus; lab IK is the VNC; DA stamps snapshots

**Claim (atlas + PDFs).** Do not train 166,700 cells as an arm controller. Steal the **split**:

| Fly | B601 lab |
|---|---|
| Eyes / optic lobe | Front camera (object offset, loom, not a class label) |
| Tarsi / foreleg tap | Wrist / gripper camera (midline, contact) |
| Chordotonal | TCP + gripper width |
| Descending neurons (~thin sparse bus) | 8–12 channels: dx,dy,dz, approach, grip, lift, abort |
| VNC motor circuits | **Existing IK / floor / attach** (fast unpacker) |
| Giant fibre | Abort: open + retreat |
| CX heading ring | Skip (mounted arm). Compass = tool-frame error |

**Loop.** Same JPEGs → R1/R8 → LIF → **read named DN rates** → DN mixer → lab IK. Front and Gripper stay **split** (do not concat into one embedding). Optional handwritten mode skeleton (search → align → close → lift → abort) is **lab routing**, labeled, not `fly_picked`.

**Dopamine (PDF).** Not a scalar “good job” / RPE through the arm. PAM/PPL1 write **valence into a mushroom-body compartment**: Kenyon cells = sparse snapshot of sensors; DAN + KC → depress/potentiate KC→MBON. Imitation (teacher) learns **how to move** on the DN bus. DA learns **which camera snapshots are worth routing** onto that bus. One pulse at **trial end** (success PAM, fail PPL1), not dense “closer to cube” (that teaches the front camera to chase the arm). MBON may **gate** DN channel gains (grip, approach, abort). It must not emit joints. VNC/IK is not trained by DA.

**Pass/fail.** Cube vs black changes **named DN rates**. Silence those DNs → command collapses. T1 MN Hz is logged (reachability), not the joystick. `fly_picked` if `acting_map=dn_bus` and DNs varied with the image. `da_learned` only if terminal DA changes KC→MBON **and** that change gates the bus (on vs off, shuffled, freeze, erase).

**Why a local session called this a stronger *argument*.**

1. **Same connectome, different lesson.** MaleCNS’s headline is an intact **neck**: DNs out, proprio/efference back, VNC as spinal cord. Robotics transfer on fly-connectome.grok.me: “If you cannot list the channels that cross from perception to actuation, you do not have a fly-like architecture — you have a blob.” Approach A lists **muscle pools**. Approach B lists **the neck bus**. For a body that is not a fly, the neck is the portable interface; the muscles are the plant you don’t have.

2. **The plant already exists.** The lab IK + kinematic grasp **is** what a VNC does: unpack a low-dimensional command at high rate, close a local loop. Putting T1 MN rates through another map onto TCP asks the connectome to also be a B601 plant. The PDF: “You would not train the 166,700-cell fly as an arm controller. The body is wrong.”

3. **Our LIF agrees with “don’t wait for MNs.”** Structural path to T1 MNs exists at hop 4; spikes do not. Named DNs are also silent at 3 hops but are **closer** on `|W|^k` (hop 2–3). If anything in this LIF will light up, it is more likely an early DN than a T1 MN. That is a **practical** reason to try B first, not a proof B is the user’s goal.

4. **Dopamine’s real job in the fly.** MB compartments tag **odors/visual snapshots** with valence. They do not run a critic on joint error. Approach A’s “DA must change the MN command to count” imports mammalian RPE into the cord. Approach B’s “DA gates which snapshot is worth approaching” matches PAM/PPL1. Dense shaping (“+ for closer TCP”) is explicitly the wrong teacher (Schultz-style, front cam chases the arm).

5. **Two cameras, two ranges.** Flies put cheap fast sensors on the effector (tarsi) and a different stack on the eyes. Concatenating Front+Gripper into one k-NN (what actually picked) is the opposite of a neck. B keeps the split even if the mixer is still engineered.

6. **Honesty vs theater.** Steering with cells whose rate is identically zero is a cheat of the same family as scoring overlay as fly. B still requires **live DN rates to move**. If they stay zero, B must also fail closed, not fall back to k-NN.

**Why B may be weaker for the user’s sentence.**

- It **demotes** the cells they named (leg MNs) to a log. That can look like the 2026-09-09 “skip the VNC” draft they rejected.
- Named DN → TCP is still an **engineered gamepad** (doomfly’s DN→keys). The atlas says name the bus; it does not say DNp20 *is* Δy.
- If DNs stay silent, B is as empty as A.
- A handwritten mode skeleton is easy to smuggle in as the real picker (the same overlay bug).

---

## Hybrid (if you reject a pure A or B)

1. **Scored command = named DN bus** unpacked by lab IK (B’s split).
2. **Must log T1 MN rates** every tick. Publish whether they ever move. If a later LIF crop actually spikes T1 MNs, an ablation can *add* them as extra bus channels — not silently replace DNs.
3. **DA = terminal snapshot valence** gating DN gains (B), not MN RPE (not A).
4. **Never** score k-NN / overlay / mode expert as `fly_picked`.
5. **Pass A** is whichever scored cells you chose (DNs for B, T1 MNs for A) vs black frame. Silence those cells → arm stops.

---

## Questions for the analyzing model

1. Given the user’s sentence (“leg motor neurons operate the arm”) vs MaleCNS neck biology, is A, B, or the hybrid the honest architecture?
2. Is lab IK a legitimate VNC analog, or is skipping live MN rates repeating “VNC unused”?
3. Should DA ever write into motor synapses / MN readout, or only KC→MBON snapshot valence?
4. If this LIF never spikes DNs *or* MNs, is the scientific result “this neuron model cannot motor a B601,” and should imitation k-NN stay a labeled **non-fly** baseline only?
5. Which MaleCNS DN types are the least-wrong bus for tool-frame error, grip, lift, abort (DNg13, DNp20, DNa02, giant fibre, etc.)?
6. What cheats would you reject in a review (`fly_picked` true while rates are constant, mode machine acting, concat-JPEG k-NN, dense distance reward)?

---

## Local session’s current stance (for you to attack)

Approach B is a **stronger argument about what to steal from a fly for a robot that already has a cord (IK)**. Approach A is a **stronger reading of the user’s words**. The LIF audit forbids pretending MNs (or DNs) already drive the arm. The k-NN pick proves the **lab** is solvable, not that MaleCNS motors it.

We should not have said B is “better” as if a pick experiment had decided it. We should say: **B is the fly-shaped interface; A is the user’s cell list; the data say neither cell list is live yet.**
