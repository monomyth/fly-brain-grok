# What you need to do

Run these **on isengard**, in `~/robotics/grok`. This Mac must not open `/dev/ttyACM0`.

## No motion (you can do this now)

1. Cube on the pad, lights on if the overview looks black.
2. No teleop / `run-dual-policy` / gripper scripts running.
3. Then:

```sh
cd ~/robotics/grok
./status
./phase0
./grab
./hop
./shadow
```

`./hop` `ok=true` means scored DNs split the table vs black pixels. It is **not** a pick. `fly_picked` stays false.

## Motion (you must be at the cell)

Geometry is spec (`b601-v1.json`: URDF + follower limits + linear 0–90 mm vs 0…−270°). Not caliper.

```sh
cd ~/robotics/grok
./hover           # print plan, no motors
./hover --go      # ENABLES TORQUE, lifts TCP 15 mm, returns. E-stop in reach.
./ready           # print unfold plan, no motors
./ready --go      # ENABLES TORQUE, unfolds to spec Ready (~95°) and STAYS. Clear the pad.
./fold --go       # ENABLES TORQUE, returns to calibrated zeros and STAYS.
./nudge           # hop last grab, print capped fly Δmm. No motors.
./nudge --go      # ENABLES TORQUE, apply that Δmm from current pose, return.
./cycle --go      # Ready, grab, fly nudge, always fold at the end. E-stop in reach.
```

Herdr and SSH are not an e-stop. `connect()` on the follower enables torque.

## Do not run from here

`open-gripper`, `close-gripper`, `lerobot-teleoperate`, `lerobot-rollout`, `run-dual-policy` unless you intend to move the arm.
