# Brief for Grok Build: MaleCNS → ReBot B601-DM arm

**Use this as the only local-session context.** Atlas: https://fly-connectome.grok.me and `/robotics`. Spec that was implemented: `MaleCNS-B601-Grok-Build-Plan.md` (web Grok plan). Lab is a **kinematic** B601-DM sim (not MuJoCo). Python owns MaleCNS LIF; Swift is body-only.

**Ask of you:** say whether to (1) run Phase 3 distill now that hop-probe is green on a **synthetic** cube, (2) first require live Front+Gripper JPEGs, (3) change the eye/LIF so scramble/real cameras also move DNs, or (4) something else. Do **not** score overlay, quintic, or k-NN as `fly_picked`. Do **not** recommend training 166k cells as a 6-DoF policy.

---

## Original user sentence

Pictures into optical neurons. **Motor / DN activity** controlling the arm. **Dopamine-reinforced.** Front + gripper cameras. Pick a cube (10–90 mm). Neurons that operate a **fly leg** should operate the robot — interpreted at the **neck** (T1-targeting DNs), not tibia MN Hz as a 6-DoF joystick. Overlay/teacher is a scaffold to remove.

**`fly_picked` only if all of:** `acting_map=dn_bus`; teacher/k-NN off; scored DN Hz > ε and varies vs black; command is \(a=U(g)\cdot r\) with trained \(g\); attach + live cube \(z\ge 100\) mm + TCP level + hold \(\ge 2\) s; silence those DNs → arm stops; freeze \(g\leftarrow g_{\mathrm{init}}\) drops pick rate.

Until that is true, the lab may pick. The fly has not.

---

## Architecture (settled)

```
Front JPEG  → luma + ON contrast → left-eye R1 (hex)
Gripper JPEG → same, separate columns → right-eye R1
                    ↓
         crop LIF (~60k cells, 150 steps/frame, no per-frame reset)
         topology W frozen; type-gains g trainable
                    ↓
         r = DNfl + DNxl + DNa01 + DNa02 + DNp01(GF abort) + MDN + DNp07 + DNp10
         T1 vnc_motor logged only, not the joystick
                    ↓
         a = U(g)·r   U ≤ 11 params + existing keep_level IK
         if r ≈ 0: a = 0  (fail closed)
```

**Select DNs from MaleCNS v1.0 annotations (not MANC `type` strings):**

| Name | How | n |
|---|---|---:|
| DNfl | `mancType` startswith `DNfl` | 95 |
| DNxl | `mancType` startswith `DNxl` | 158 |
| DNa01, DNa02, DNp01, DNp07, DNp10 | exact `type` (DNp10 **not** DNp101–104) | 2 each |
| MDN | exact `type=MDN` | 4 |
| Giant fibre | `type=DNp01`, hemibrainType Giant Fiber — **not** `type=GF` | 2 |

`type` has **no** `DNfl*`. Subclass `fl`/`xl` on descending is broader (115/275); v1 uses **mancType prefixes**.

**Exclude from bus:** DNg02 (exists as `DNg02_a`…`_g`, 29 cells, wings), DNp15, DNp20, DNp22, DNg13-as-reach.

**Do not:** concat Front+Gripper into one k-NN; map DNp20→Δy because it is famous; put thousands of params in \(U\) with \(g=1\); dense “closer to cube” reward; MBON emitting joints; per-frame LIF reset; N<50 steps.

---

## What already exists

**Body:** ReBot B601-DM kinematic sim. TCP = closed fingertip. Cube attach = pads on volume (HANDOFF). IK `keep_level`. Front (Gemini 336L analog), Gripper (Gemini 305, ~15° down). Capture `apply:false` offscreen so Orbit does not jerk. Overlay-servo and quintic teacher **can pick**. A Front+Gripper **k-NN** on 209 demo rows **picked once** (cube z 20→120 mm, hold 3.5 s). That is `acting_map=knn`, **lab_picked**, never fly.

**Brain:** MaleCNS v1.0 signed CSR, ~166,700 neurons, brain+VNC. Crop for the inner loop: **59,740** cells, **2,083,432** edges. Photoreceptors: JPEG luma + ON DoG → R1–R6 (proxy grid, not a measured eyemap). Plasticity coded: ~4,184 KC→MBON11, PPL101 pulse, eligibility KC×MBON; reward **sign from the event**, not PPL Hz.

**Code (local tree `fly-brain/controller/arm/`):** `eye.py`, `crop.py`, `lif_crop.py`, `hop_probe.py`, `bus.py`, `unpack.py`, `score.py` (only place `fly_picked` is set), `teacher.py`, `train_distill.py`, `train_operant.py`, `run_dn_bus.py`. Tests in `tests/test_arm.py`.

---

## Measured LIF (do not invent new plants)

### Full-graph, 2026-09-11 (old)

3–500 steps, gain 0.02–0.275, per-frame reset or not: **named DN Hz = 0, T1 MN Hz = 0**. R1 fires by ~100 steps. Structural \(|W|^k\): DNs by hop 2–3, T1 MNs by hop 4. All-R1-on hop-1 current on DN/MN = **0** (optic lobe absorbs it).

### Crop hop-probe (current)

`synaptic_gain=2.5`, 150 steps, **no per-frame reset**. Inject `i_ext=4` on a layer (connectivity walk; several layers **saturate** kHz — LC10/DNfl). That inject table is **not** the vision test.

**Vision** (synthetic 120×160 orange blob, Front=Gripper same image):

| encoder | cube DN Hz | black | scramble | notes |
|---|---:|---:|---:|---|
| DoG only, OFF on odd receptors, no AGC | **0** | 0 | 0 | R1 still ~2 Hz mean; DNs mute |
| 98th-percentile AGC + gain 2.5 | 132 | 0 | 365 | cosine(cube,scramble)≈0.92; GF 1.7–2.6 kHz **abort**. Flood vs silence. Rejected. |
| **luma + ON contrast**, scale luma=4, DoG=30, **no AGC**, gain 2.5 | **95.4** (75/267 > 1 Hz) | **0** | **0** | shift L2 149; cmd ≈ (2.6, 4, −4, 4) mm; GF 1400 < abort 2000; silence DNs → a=0 |

**hop_probe `ok=true`** on that last row (synthetic cube only).

Why DoG-only failed: R1–R6 are ON luminance; putting OFF on every other index left the cube **interior** dark (~64 edge cells above threshold). Injecting all R1 reached DNs; a picture did not. Luma drive lights the blob interior.

Scramble is currently **silent** (phase-scramble + clip → little luma energy), so cube-vs-scramble is **not** a strong “object vs matched-energy noise” test. Cube-shift changes the DN **vector** (L2 149) but **not** the clipped TCP command (tanh saturated).

---

## Training status

| Phase | Status |
|---|---|
| 0 flags (`acting_map`, sealed `fly_picked`) | done |
| 1 hop-probe | **green on synthetic cube**; not proven on live JPEGs |
| 2 fail-closed untrained \(U_0\cdot r\) | coded; live sim often down; no pick claimed |
| 3 distill teacher → \(g\) | **not re-run** after hop turned green. Older AGC distill (`g-distill.agc-flood.npz`) is **invalid**. Canonical `g-distill.json` is `skipped: true` |
| 4 operant + DA | skipped; `da_learned=false` |
| Live `fly_picked` | **false** |

23 type-gains in \(g\). \(U\) has 11 numbers, `step=8` mm (same as overlay clip). Distill must not use thousands of \(U\) params.

---

## Cheats already burned (do not repeat)

Scoring overlay / quintic / pad expert / attached k-NN / grip≤72 map-swap as fly. Editing eval JSON after the process exits. `--g-init` that did not freeze \(g\). Empty tick logs. `apply_gains` wiping KC→MBON. ES that kept unevaluated noise. Per-frame AGC making any JPEG look like a cube vs black.

---

## What is still false

The **fly does not control the live arm.** Hop-green is a **synthetic orange rectangle** in Python, not Front/Gripper from the sim. Distill has not been opened on the luma encoder. Dopamine has not been shown to change the bus.

---

## Suggested next work (you decide)

1. **Live hop-probe** on real Front+Gripper `apply:false` JPEGs (cube vs empty table vs scramble of the real frame). If DN Hz stays 0, the synthetic result does not transfer.
2. If live DNs move: **Phase 2** untrained `dn_bus` on the arm (Folded→Ready). Fail closed. `fly_picked` still false until \(g\) is trained.
3. **Phase 3** distill overlay TCP into \(g\) (ES/Adam on ~23 gains + tiny \(U\)), teacher **off**, freeze-\(g\) ablation.
4. **Phase 4** terminal ±1 + PAM/PPL on KC→MBON gating DN gains; `da_learned` only if erase changes behavior.
5. Tighten vision tests: scramble should have **matched luma energy**; cube-shift should change **command**, not only DN L2; GF must stay below abort on a cube.

RTX 4090 is available (`blacktower.local`) but **does not fix silent DNs**. It only matters once hop is green on **live** cameras and you are fitting \(g\).

---

## One-paragraph prompt you can paste

> Read this brief. The local session implemented the MaleCNS DN-bus plan on a kinematic ReBot B601-DM. Scored command = mancType DNfl/DNxl + DNa01/02 + DNp01/MDN/DNp07/DNp10 rates, unpacked by existing IK, fail closed if rates≈0. T1 motor neurons are logged only. Hop-probe is green on a synthetic orange cube after luma+ON photoreceptor drive (cube DN 95 Hz, black 0); it was red when only DoG/OFF-parity was used. Live sim JPEGs and distill have not been run on that encoder. fly_picked is false. Overlay/k-NN picks are lab_picked only. Propose the next implementation steps so live MaleCNS rates pick the cube, with terminal dopamine as snapshot valence / gain gate, not joint RPE. Do not score a teacher as the fly. Do not skip hop-probe on live cameras.
