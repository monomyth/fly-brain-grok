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

Not wired in these scripts yet. When it is, you still have to:

1. Be there. Hardware e-stop in reach. Support the arm.
2. Confirm `./phase0` shows CAN free and no motor process.
3. Hand-fold zeros, gripper shut, then the usual `lerobot-calibrate` (that **does** enable motors).
4. Caliper: open/close mm vs degrees → we write `b601-v1.json`.
5. Tape TCP / table height.
6. Only then: millimetre hovers, still on isengard, never from the Mac.

Herdr and SSH are not an e-stop. `connect()` on the follower enables torque.

## Do not run from here

`open-gripper`, `close-gripper`, `lerobot-teleoperate`, `lerobot-rollout`, `run-dual-policy` unless you intend to move the arm.
