"""Acquire one frame per external (hardware) trigger from a MindVision camera.

Target hardware: LCOptics AIC-231GM-USB == MindVision MV-SUA231GM (mono, USB).
An external ~4 V pulse on the camera GPIO IN line makes the sensor expose once;
this script blocks on the SDK until such a frame arrives, then stores it in
``latest_measurement`` as a NumPy array.

The MindVision Linux SDK must be installed (libMVSDK.so) and mvsdk.py must be
importable. Set MVSDK_LIBRARY if libMVSDK.so is not next to mvsdk.py.
"""

import argparse
import ctypes
import os
import platform
import signal
import time

import cv2
import numpy as np

try:
    import mvsdk
except ImportError as error:
    raise SystemExit(
        "The official MindVision Python module 'mvsdk' was not found. "
        "Copy mvsdk.py from the MindVision SDK Python examples or add its "
        "folder to PYTHONPATH."
    ) from error
except OSError as error:
    raise SystemExit(str(error)) from error


CAMERA_INDEX = 0

# Trigger mode indices reported by the SDK in tSdkCameraCapbility.pTriggerDesc.
# 0 = continuous grab, 1 = software trigger, 2 = hardware (external) trigger.
# The index is resolved from the camera description; this is only the fallback.
HARDWARE_TRIGGER_MODE_FALLBACK = 2

# Edge/level of the external signal. EXT_TRIG_LEADING_EDGE (rising) is the right
# choice for a 0 V -> 4 V pulse.
TRIGGER_SIGNAL = mvsdk.EXT_TRIG_LEADING_EDGE

# Debounce for the input line, in microseconds. Raise it if one physical pulse
# produces several frames (contact bounce / noisy cable).
TRIGGER_JITTER_US = 100

# How long CameraGetImageBuffer waits before reporting a timeout. The script
# retries, so this only controls how often the "still waiting" line is printed.
FRAME_TIMEOUT_MS = 2000

# Manual exposure defaults. Auto exposure is useless in trigger mode because the
# AE loop needs a continuous stream to converge on.
EXPOSURE_TIME_US = 10_000.0
ANALOG_GAIN = 1.0

# This variable always contains the most recent triggered image.
latest_measurement = None

_stop_requested = False


def check_status(status, operation):
    """Raise for the SDK calls that return an error code instead of raising."""
    if status != mvsdk.CAMERA_STATUS_SUCCESS:
        raise RuntimeError(
            f"{operation} failed with SDK status {status} "
            f"({mvsdk.CameraGetErrorString(status)})"
        )


def try_status(status, operation):
    """Same as check_status, but only warn: some firmwares lack these knobs."""
    if status != mvsdk.CAMERA_STATUS_SUCCESS:
        print(
            f"Warning: {operation} is unsupported on this camera "
            f"(status {status}, {mvsdk.CameraGetErrorString(status)})"
        )


def _request_stop(_signum, _frame):
    global _stop_requested
    _stop_requested = True


def find_camera(index):
    devices = mvsdk.CameraEnumerateDevice()
    if not devices:
        raise RuntimeError(
            "No MindVision camera was found. Check the USB cable, and on Linux "
            "check that the udev rules from the SDK installer are in place "
            "(/etc/udev/rules.d/88-mvusb.rules)."
        )
    if index >= len(devices):
        raise RuntimeError(
            f"Camera index {index} is unavailable; {len(devices)} camera(s) found"
        )
    return devices[index]


def resolve_hardware_trigger_mode(capability):
    """Look up the hardware/external trigger entry in the camera description."""
    for i in range(capability.iTriggerDesc):
        entry = capability.pTriggerDesc[i]
        description = entry.GetDescription()
        lowered = description.lower()
        if "hard" in lowered or "ext" in lowered or "硬" in description:
            return entry.iIndex, description
    return HARDWARE_TRIGGER_MODE_FALLBACK, "hardware trigger (assumed)"


def configure_camera(camera_handle, capability, args):
    """Put the camera into external-trigger mode with a deterministic exposure."""
    mono_camera = capability.sIspCapacity.bMonoSensor != 0
    media_type = (
        mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono_camera else mvsdk.CAMERA_MEDIA_TYPE_BGR8
    )
    # The ISP output format must be fixed before streaming starts, not per frame.
    check_status(
        mvsdk.CameraSetIspOutFormat(camera_handle, media_type),
        "CameraSetIspOutFormat",
    )

    mode, description = resolve_hardware_trigger_mode(capability)
    check_status(
        mvsdk.CameraSetTriggerMode(camera_handle, mode),
        f"CameraSetTriggerMode({mode})",
    )
    print(f"Trigger mode {mode}: {description}")

    # Exactly one frame per external pulse.
    check_status(
        mvsdk.CameraSetTriggerCount(camera_handle, 1), "CameraSetTriggerCount"
    )
    check_status(
        mvsdk.CameraSetExtTrigSignalType(camera_handle, args.edge),
        "CameraSetExtTrigSignalType",
    )
    try_status(
        mvsdk.CameraSetExtTrigJitterTime(camera_handle, args.jitter_us),
        "CameraSetExtTrigJitterTime",
    )
    try_status(
        mvsdk.CameraSetExtTrigDelayTime(camera_handle, args.delay_us),
        "CameraSetExtTrigDelayTime",
    )

    # Manual exposure: an auto-exposure loop cannot converge on a trigger stream.
    check_status(mvsdk.CameraSetAeState(camera_handle, 0), "CameraSetAeState")
    check_status(
        mvsdk.CameraSetExposureTime(camera_handle, args.exposure_us),
        "CameraSetExposureTime",
    )
    try_status(
        mvsdk.CameraSetAnalogGainX(camera_handle, args.gain), "CameraSetAnalogGainX"
    )

    return mono_camera


def allocate_frame_buffer(capability, mono_camera):
    """One buffer, sized for the largest frame the sensor can produce."""
    buffer_size = (
        capability.sResolutionRange.iWidthMax
        * capability.sResolutionRange.iHeightMax
        * (1 if mono_camera else 3)
    )
    frame_buffer = mvsdk.CameraAlignMalloc(buffer_size, 16)
    if not frame_buffer:
        raise RuntimeError(f"CameraAlignMalloc({buffer_size}) returned NULL")
    return frame_buffer


def process_raw_frame(camera_handle, raw_buffer, frame_head, frame_buffer):
    """Run the ISP on a raw buffer and return the result as a NumPy array.

    Always releases raw_buffer, including on failure: the driver's internal
    queue starves after a few frames if buffers are not handed back.
    """
    try:
        check_status(
            mvsdk.CameraImageProcess(
                camera_handle, raw_buffer, frame_buffer, frame_head
            ),
            "CameraImageProcess",
        )
    finally:
        mvsdk.CameraReleaseImageBuffer(camera_handle, raw_buffer)

    # Windows hands out bottom-up (BMP-style) images; Linux is already top-down.
    if platform.system() == "Windows":
        mvsdk.CameraFlipFrameBuffer(frame_buffer, frame_head, 1)

    mono_frame = frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8
    channels = 1 if mono_frame else 3
    image_size = frame_head.iWidth * frame_head.iHeight * channels
    image_bytes = ctypes.string_at(frame_buffer, image_size)

    shape = (
        (frame_head.iHeight, frame_head.iWidth)
        if mono_frame
        else (frame_head.iHeight, frame_head.iWidth, 3)
    )
    return np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()


def grab_triggered_frame(camera_handle, frame_buffer, timeout_ms, verbose=True):
    """Block until the next external trigger produces a frame.

    Returns the image as a NumPy array, or None if a stop was requested while
    waiting.
    """
    global latest_measurement

    while True:
        if _stop_requested:
            return None
        try:
            raw_buffer, frame_head = mvsdk.CameraGetImageBuffer(
                camera_handle, timeout_ms
            )
            break
        except mvsdk.CameraException as error:
            if error.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                raise
            if verbose:
                print("Still waiting for a GPIO trigger...", flush=True)

    latest_measurement = process_raw_frame(
        camera_handle, raw_buffer, frame_head, frame_buffer
    )
    return latest_measurement


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Grab one frame per external trigger from a MindVision camera."
    )
    parser.add_argument(
        "--index", type=int, default=CAMERA_INDEX, help="camera index (default: 0)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="number of triggered frames to grab; 0 means run until Ctrl+C",
    )
    parser.add_argument(
        "--edge",
        type=int,
        default=TRIGGER_SIGNAL,
        choices=[0, 1, 2, 3, 4],
        help="0=rising, 1=falling, 2=high level, 3=low level, 4=both edges",
    )
    parser.add_argument(
        "--exposure-us", type=float, default=EXPOSURE_TIME_US, help="exposure in us"
    )
    parser.add_argument(
        "--gain", type=float, default=ANALOG_GAIN, help="analog gain multiplier"
    )
    parser.add_argument(
        "--jitter-us", type=int, default=TRIGGER_JITTER_US, help="input debounce in us"
    )
    parser.add_argument(
        "--delay-us", type=int, default=0, help="delay between trigger and exposure"
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=FRAME_TIMEOUT_MS, help="per-wait timeout"
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="if set, every triggered frame is written there as a PNG",
    )
    parser.add_argument(
        "--show", action="store_true", help="display each frame in an OpenCV window"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    camera_handle = None
    frame_buffer = None
    try:
        device = find_camera(args.index)
        print(f"Opening {device.GetFriendlyName()} ({device.GetPortType()})")
        camera_handle = mvsdk.CameraInit(device, -1, -1)
        capability = mvsdk.CameraGetCapability(camera_handle)

        mono_camera = configure_camera(camera_handle, capability, args)
        frame_buffer = allocate_frame_buffer(capability, mono_camera)

        check_status(mvsdk.CameraPlay(camera_handle), "CameraPlay")
        print("Waiting for GPIO triggers (Ctrl+C to stop)...", flush=True)

        grabbed = 0
        while not _stop_requested and (args.count == 0 or grabbed < args.count):
            frame = grab_triggered_frame(camera_handle, frame_buffer, args.timeout_ms)
            if frame is None:
                break
            grabbed += 1
            print(
                f"Frame {grabbed}: {frame.shape[1]}x{frame.shape[0]}, "
                f"mean={frame.mean():.1f}",
                flush=True,
            )

            if args.save_dir:
                path = os.path.join(
                    args.save_dir, f"frame_{grabbed:05d}_{time.time_ns()}.png"
                )
                if not cv2.imwrite(path, frame):
                    print(f"Warning: could not write {path}")

            if args.show:
                cv2.imshow("triggered", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        return latest_measurement
    finally:
        if camera_handle is not None:
            # Stop streaming before freeing the buffer the ISP writes into.
            mvsdk.CameraStop(camera_handle)
            mvsdk.CameraUnInit(camera_handle)
        if frame_buffer is not None:
            mvsdk.CameraAlignFree(frame_buffer)
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
