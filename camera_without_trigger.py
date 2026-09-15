"""Take a photo with a MindVision camera in continuous mode - no trigger needed.

This is the free-running counterpart of trigger_camera.py: the sensor streams on
its own and the script simply grabs a frame and saves it. Use it to check focus,
aperture and exposure before wiring anything to the GPIO input.

Target hardware: LCOptics AIC-231GM-USB == MindVision MV-SUA231GM (mono, USB).
Note that this camera is not a UVC device, so cv2.VideoCapture cannot see it;
everything goes through libMVSDK.

Usage:
    python3 camera_without_trigger.py                      # one auto-exposed shot
    python3 camera_without_trigger.py -o shot.png
    python3 camera_without_trigger.py --exposure-us 8000   # manual exposure
    python3 camera_without_trigger.py --count 10 --interval 0.5
    python3 camera_without_trigger.py --preview            # live window, SPACE saves
"""

import argparse
import os
import time

import cv2
import mvsdk

from trigger_camera import (
    allocate_frame_buffer,
    check_status,
    find_camera,
    process_raw_frame,
    try_status,
)

CONTINUOUS_TRIGGER_MODE = 0  # 0 = free run, 1 = software trigger, 2 = hardware
DEFAULT_OUTPUT = "photo.png"

# Frames discarded before the first saved shot. In continuous mode the very
# first frames are exposed with whatever settings the sensor powered up with,
# and auto exposure needs a few frames to converge.
WARMUP_FRAMES = 8

FRAME_TIMEOUT_MS = 5000


def configure_camera(camera_handle, capability, args):
    """Free-running mode, with either auto or manual exposure."""
    mono_camera = capability.sIspCapacity.bMonoSensor != 0
    media_type = (
        mvsdk.CAMERA_MEDIA_TYPE_MONO8 if mono_camera else mvsdk.CAMERA_MEDIA_TYPE_BGR8
    )
    # Must be fixed before streaming starts.
    check_status(
        mvsdk.CameraSetIspOutFormat(camera_handle, media_type), "CameraSetIspOutFormat"
    )
    check_status(
        mvsdk.CameraSetTriggerMode(camera_handle, CONTINUOUS_TRIGGER_MODE),
        "CameraSetTriggerMode(continuous)",
    )

    if args.exposure_us is None:
        # Auto exposure works here precisely because the stream is continuous.
        check_status(mvsdk.CameraSetAeState(camera_handle, 1), "CameraSetAeState(auto)")
        print("Exposure: auto")
    else:
        check_status(
            mvsdk.CameraSetAeState(camera_handle, 0), "CameraSetAeState(manual)"
        )
        check_status(
            mvsdk.CameraSetExposureTime(camera_handle, args.exposure_us),
            "CameraSetExposureTime",
        )
        try_status(
            mvsdk.CameraSetAnalogGainX(camera_handle, args.gain),
            "CameraSetAnalogGainX",
        )
        print(f"Exposure: manual, {args.exposure_us:.0f} us, gain x{args.gain}")

    if not mono_camera and capability.sIspCapacity.bAutoWb:
        try_status(mvsdk.CameraSetWbMode(camera_handle, 1), "CameraSetWbMode(auto)")

    print(f"Sensor: {'mono' if mono_camera else 'colour'}")
    return mono_camera


def grab_frame(camera_handle, frame_buffer, timeout_ms=FRAME_TIMEOUT_MS):
    """Grab the next streamed frame as a NumPy array."""
    raw_buffer, frame_head = mvsdk.CameraGetImageBuffer(camera_handle, timeout_ms)
    return process_raw_frame(camera_handle, raw_buffer, frame_head, frame_buffer)


def warm_up(camera_handle, frame_buffer, frames):
    """Discard the first frames so auto exposure has time to settle."""
    for _ in range(frames):
        try:
            grab_frame(camera_handle, frame_buffer)
        except mvsdk.CameraException as error:
            if error.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                raise
            # A timeout during warm-up is not fatal; the real grab will report it.
            return


def output_path(base, index, total):
    """photo.png for a single shot, photo_001.png... for a series."""
    if total == 1:
        return base
    root, extension = os.path.splitext(base)
    return f"{root}_{index:03d}{extension or '.png'}"


def save(frame, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"Could not write {path!r}")
    print(
        f"Saved {path}  ({frame.shape[1]}x{frame.shape[0]}, "
        f"mean brightness {frame.mean():.1f})"
    )


def run_preview(camera_handle, frame_buffer, args):
    """Live window: SPACE saves the current frame, q or ESC quits."""
    print("Preview: SPACE = save, q or ESC = quit")
    saved = 0
    while True:
        try:
            frame = grab_frame(camera_handle, frame_buffer)
        except mvsdk.CameraException as error:
            if error.error_code == mvsdk.CAMERA_STATUS_TIME_OUT:
                continue
            raise

        cv2.imshow("camera (no trigger)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            saved += 1
            save(frame, output_path(args.output, saved, 0))
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Photograph with a MindVision camera without any trigger."
    )
    parser.add_argument("--index", type=int, default=0, help="camera index")
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT, help=f"output file ({DEFAULT_OUTPUT})"
    )
    parser.add_argument("--count", type=int, default=1, help="how many photos to take")
    parser.add_argument(
        "--interval", type=float, default=0.0, help="seconds between photos"
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        help="manual exposure in microseconds; omit for auto exposure",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="analog gain multiplier")
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_FRAMES,
        help=f"frames discarded before the first shot (default {WARMUP_FRAMES})",
    )
    parser.add_argument(
        "--preview", action="store_true", help="live window instead of a single shot"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

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

        if args.warmup > 0:
            warm_up(camera_handle, frame_buffer, args.warmup)

        if args.preview:
            return run_preview(camera_handle, frame_buffer, args)

        frame = None
        for i in range(1, args.count + 1):
            if i > 1 and args.interval > 0:
                time.sleep(args.interval)
            frame = grab_frame(camera_handle, frame_buffer)
            save(frame, output_path(args.output, i, args.count))
        return frame
    except mvsdk.CameraException as error:
        raise SystemExit(f"SDK error: {error}") from error
    finally:
        if camera_handle is not None:
            mvsdk.CameraStop(camera_handle)
            mvsdk.CameraUnInit(camera_handle)
        if frame_buffer is not None:
            mvsdk.CameraAlignFree(frame_buffer)
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
