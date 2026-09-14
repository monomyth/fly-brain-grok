#!/usr/bin/env python3
"""Grab wrist/overview JPEGs on the robot host. Opens Orbbec SDK, not the B601 follower."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SERIAL = {"wrist": "CV2L761000FA", "overview": "CPC6463000PZ"}
WH = (848, 480)


def _grab_one(name: str, n: int) -> dict:
    import cv2
    import numpy as np
    from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline

    serial = SERIAL[name]
    ctx = Context()
    device = ctx.query_devices().get_device_by_serial_number(serial)
    pipeline = Pipeline(device)
    config = Config()
    color = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
        WH[0], WH[1], OBFormat.YUYV, 30
    )
    config.enable_stream(color)
    pipeline.start(config)
    frames_out = []
    t0 = time.monotonic()
    try:
        got = 0
        last = time.monotonic()
        while got < n:
            bundle = pipeline.wait_for_frames(1000)
            frame = bundle.get_color_frame() if bundle else None
            if frame is None:
                if time.monotonic() - last > 5:
                    raise TimeoutError(f"{name}: no color frames")
                continue
            last = time.monotonic()
            recv = time.time()
            raw = np.asarray(frame.get_data(), dtype=np.uint8).reshape(WH[1], WH[0], 2)
            bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_out.append({"receive_time": recv, "rgb": rgb})
            got += 1
    finally:
        pipeline.stop()
    elapsed = max(time.monotonic() - t0, 1e-6)
    return {
        "name": name,
        "serial": serial,
        "model": device.get_device_info().get_name(),
        "n": len(frames_out),
        "fps": len(frames_out) / elapsed,
        "elapsed_s": elapsed,
        "width": WH[0],
        "height": WH[1],
        "frames": frames_out,
    }


def main(argv: list[str] | None = None) -> int:
    import cv2

    p = argparse.ArgumentParser(description="Phase 1 Orbbec grab (no motors)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--cameras", nargs="+", default=["wrist", "overview"])
    args = p.parse_args(argv)
    if args.frames < 1:
        print("frames must be ≥ 1")
        return 2
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "phase": 1,
        "connect_called": False,
        "serials": {},
        "cameras": {},
        "timestamp_kind": "host_receive_after_wait_for_frames",
        "hardware_sync": False,
    }
    for name in args.cameras:
        if name not in SERIAL:
            print(f"unknown camera {name}")
            return 2
        burst = _grab_one(name, args.frames)
        rgb = burst["frames"][-1]["rgb"]
        path = out / f"{name}.jpg"
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"{name}: jpeg encode failed")
        path.write_bytes(jpeg.tobytes())
        t_first = burst["frames"][0]["receive_time"]
        dts = [fr["receive_time"] - burst["frames"][i - 1]["receive_time"] for i, fr in enumerate(burst["frames"]) if i]
        meta["serials"][name] = burst["serial"]
        meta["cameras"][name] = {
            "serial": burst["serial"],
            "model": burst["model"],
            "jpeg_bytes": path.stat().st_size,
            "width": burst["width"],
            "height": burst["height"],
            "n": burst["n"],
            "fps": burst["fps"],
            "mean_dt_s": float(sum(dts) / len(dts)) if dts else None,
            "path": str(path),
            "age_vs_first_s": burst["frames"][-1]["receive_time"] - t_first,
        }
        del burst["frames"]
    (out / "capture.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "cameras"} | {"ok": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
