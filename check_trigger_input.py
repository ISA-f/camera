"""Probe the external trigger input of a MindVision camera.

Three modes, meant to be run in this order when a trigger pulse produces no
frame. Each one isolates a different link in the chain:

  soft   - fire a software trigger. Proves the capture path (SDK, ISP, buffers)
           works with no wiring involved. If this fails, the problem is not
           electrical and there is no point measuring voltages.

  level  - switch the input line to general-purpose input and poll its logic
           level. Nothing is driven by the camera, so this is safe to leave
           running while you slowly raise the applied voltage on a bench supply:
           the voltage at which the printed level flips IS the input threshold.

  count  - put the camera back in hardware-trigger mode and watch the frame
           counters. Distinguishes "the trigger never arrives" (counters frozen)
           from "the trigger arrives but the frame is lost" (counters moving).

Usage:
    python3 check_trigger_input.py soft
    python3 check_trigger_input.py level
    python3 check_trigger_input.py count
"""

import argparse
import time

import mvsdk

from trigger_camera import check_status, find_camera, resolve_hardware_trigger_mode

POLL_INTERVAL_S = 0.005  # 200 Hz is fast enough to follow a hand-turned knob


def describe_io(capability):
    print(
        f"Programmable IO: {capability.iInputIoCounts} input(s), "
        f"{capability.iOutputIoCounts} output(s)"
    )
    if capability.iInputIoCounts == 0:
        print(
            "Warning: this camera reports no programmable input. 'level' mode\n"
            "         cannot read the line; use 'count' mode instead."
        )


def describe_trigger_capability(camera_handle):
    mask = mvsdk.CameraGetExtTrigCapability(camera_handle)
    names = {
        1 << 0: "rising edge",
        1 << 1: "falling edge",
        1 << 2: "high level",
        1 << 3: "low level",
        1 << 4: "both edges",
    }
    supported = [name for bit, name in names.items() if mask & bit]
    print(f"External trigger capability mask: 0x{mask:x} ({', '.join(supported) or 'unknown'})")


def mode_soft(camera_handle, _capability, args):
    """Fire a software trigger: does the capture path work at all?"""
    # Index 1 is the software trigger in the MindVision mode table.
    check_status(mvsdk.CameraSetTriggerMode(camera_handle, 1), "CameraSetTriggerMode(1)")
    check_status(mvsdk.CameraSetTriggerCount(camera_handle, 1), "CameraSetTriggerCount")
    check_status(mvsdk.CameraPlay(camera_handle), "CameraPlay")

    print("\nFiring a software trigger...")
    check_status(mvsdk.CameraSoftTrigger(camera_handle), "CameraSoftTrigger")
    try:
        raw_buffer, head = mvsdk.CameraGetImageBuffer(camera_handle, args.timeout_ms)
    except mvsdk.CameraException as error:
        print(
            f"FAILED: {error}\n"
            "The capture path itself is broken (exposure, bandwidth, SDK setup).\n"
            "Fix this before looking at the wiring."
        )
        return
    try:
        print(
            f"OK: got {head.iWidth}x{head.iHeight}, "
            f"bIsTrigger={head.bIsTrigger}, exposure={head.uiExpTime} us\n"
            "The capture path is fine. Any remaining problem is in the trigger\n"
            "signal or its wiring."
        )
    finally:
        mvsdk.CameraReleaseImageBuffer(camera_handle, raw_buffer)


def mode_level(camera_handle, capability, args):
    """Poll the input line level. Read-only: the camera drives nothing."""
    if capability.iInputIoCounts == 0:
        print("This camera has no programmable input line; cannot poll a level.")
        return

    index = args.io_index
    status = mvsdk.CameraSetInPutIOMode(camera_handle, index, mvsdk.IOMODE_GP_INPUT)
    if status != mvsdk.CAMERA_STATUS_SUCCESS:
        print(
            f"CameraSetInPutIOMode(GP_INPUT) failed: {status} "
            f"({mvsdk.CameraGetErrorString(status)}).\n"
            "This firmware may not allow repurposing the trigger line. "
            "Use 'count' mode instead."
        )
        return

    print(
        f"\nInput IO {index} switched to general-purpose input.\n"
        "Reading its logic level 200 times per second. Raise the applied voltage\n"
        "slowly from 0 V; the value at which 'level' flips is the threshold.\n"
        "Press Ctrl+C to stop and restore trigger mode.\n"
    )

    try:
        previous = None
        transitions = 0
        last_report = 0.0
        started = time.monotonic()
        while True:
            level = mvsdk.CameraGetIOState(camera_handle, index)
            now = time.monotonic()
            if level != previous:
                if previous is not None:
                    transitions += 1
                    print(
                        f"[{now - started:8.3f} s] level {previous} -> {level}   "
                        f"(transition #{transitions})"
                    )
                else:
                    print(f"[{now - started:8.3f} s] initial level = {level}")
                previous = level
            elif now - last_report >= 1.0:
                print(f"[{now - started:8.3f} s] level = {level} (steady)")
                last_report = now
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print(f"\nStopped after {transitions} transition(s).")
        if transitions == 0:
            print(
                "The level never changed. Either the signal never reaches the pin\n"
                "(wrong pin, reversed polarity, floating ground) or it stays below\n"
                "the threshold at every voltage you tried."
            )
    finally:
        mvsdk.CameraSetInPutIOMode(camera_handle, index, mvsdk.IOMODE_TRIG_INPUT)
        print(f"Input IO {index} restored to trigger mode.")


def mode_count(camera_handle, capability, args):
    """Hardware trigger mode: are pulses reaching the camera at all?"""
    mode, description = resolve_hardware_trigger_mode(capability)
    check_status(
        mvsdk.CameraSetTriggerMode(camera_handle, mode), f"CameraSetTriggerMode({mode})"
    )
    print(f"Trigger mode {mode}: {description}")
    check_status(mvsdk.CameraSetTriggerCount(camera_handle, 1), "CameraSetTriggerCount")
    check_status(
        mvsdk.CameraSetExtTrigSignalType(camera_handle, args.edge),
        "CameraSetExtTrigSignalType",
    )
    check_status(mvsdk.CameraPlay(camera_handle), "CameraPlay")

    print(
        "\nWatching frame counters. Apply your trigger pulses now.\n"
        "  total rising   -> pulses ARE reaching the camera\n"
        "  total frozen   -> nothing reaches the camera (electrical problem)\n"
        "Press Ctrl+C to stop.\n"
    )
    try:
        previous = (-1, -1, -1)
        while True:
            stat = mvsdk.CameraGetFrameStatistic(camera_handle)
            current = (stat.iTotal, stat.iCapture, stat.iLost)
            if current != previous:
                print(
                    f"total={stat.iTotal:6d}  captured={stat.iCapture:6d}  "
                    f"lost={stat.iLost:6d}"
                )
                previous = current
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    # Optional so that a bare "Run" from an IDE still does something useful:
    # 'soft' is the safe first step and touches no wiring.
    parser.add_argument(
        "mode",
        nargs="?",
        default="soft",
        choices=["soft", "level", "count"],
        help="diagnostic to run (default: soft)",
    )
    parser.add_argument("--index", type=int, default=0, help="camera index")
    parser.add_argument("--io-index", type=int, default=0, help="input IO line index")
    parser.add_argument(
        "--edge",
        type=int,
        default=mvsdk.EXT_TRIG_LEADING_EDGE,
        choices=[0, 1, 2, 3, 4],
        help="0=rising, 1=falling, 2=high level, 3=low level, 4=both edges",
    )
    parser.add_argument("--timeout-ms", type=int, default=3000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Mode: {args.mode}  (other modes: soft, level, count)")

    camera_handle = None
    try:
        device = find_camera(args.index)
        print(f"Opening {device.GetFriendlyName()} ({device.GetPortType()})")
        camera_handle = mvsdk.CameraInit(device, -1, -1)
        capability = mvsdk.CameraGetCapability(camera_handle)

        describe_io(capability)
        describe_trigger_capability(camera_handle)

        {"soft": mode_soft, "level": mode_level, "count": mode_count}[args.mode](
            camera_handle, capability, args
        )
    finally:
        if camera_handle is not None:
            mvsdk.CameraStop(camera_handle)
            mvsdk.CameraUnInit(camera_handle)


if __name__ == "__main__":
    main()
