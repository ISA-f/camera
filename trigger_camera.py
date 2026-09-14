"""Acquire one frame per external GPIO trigger from a MindVision camera.

The MindVision SDK Python module (mvsdk.py) must be available on PYTHONPATH.
The camera GPIO input must be wired according to the MV-SUA231GM manual.
"""

import ctypes
import sys

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


CAMERA_INDEX = 0
TRIGGER_SIGNAL = 0  # 0: rising edge, 1: falling edge
FRAME_TIMEOUT_MS = 10_000

# This variable always contains the most recent triggered BGR image.
latest_measurement = None


def check_status(status, operation):
    if status != mvsdk.CAMERA_STATUS_SUCCESS:
        raise RuntimeError(f"{operation} failed with SDK status {status}")


def find_camera(index):
    devices = mvsdk.CameraEnumerateDevice()
    if not devices:
        raise RuntimeError("No MindVision camera was found")
    if index >= len(devices):
        raise RuntimeError(
            f"Camera index {index} is unavailable; {len(devices)} camera(s) found"
        )
    return devices[index]


def configure_external_trigger(camera_handle):
    check_status(
        mvsdk.CameraSetTriggerMode(camera_handle, 1),
        "CameraSetTriggerMode(EXTERNAL_TRIGGER)",
    )
    check_status(
        mvsdk.CameraSetExtTrigSignalType(camera_handle, TRIGGER_SIGNAL),
        "CameraSetExtTrigSignalType",
    )


def grab_triggered_frame(camera_handle, capability):
    global latest_measurement

    mono_camera = capability.sIspCapacity.bMonoSensor != 0
    media_type = (
        mvsdk.CAMERA_MEDIA_TYPE_MONO8
        if mono_camera
        else mvsdk.CAMERA_MEDIA_TYPE_BGR8
    )
    check_status(
        mvsdk.CameraSetIspOutFormat(camera_handle, media_type),
        "CameraSetIspOutFormat",
    )

    frame_buffer = None
    raw_buffer = None
    frame_head = None

    try:
        while True:
            try:
                raw_buffer, frame_head = mvsdk.CameraGetImageBuffer(
                    camera_handle, FRAME_TIMEOUT_MS
                )
                break
            except mvsdk.CameraException as error:
                if error.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    raise
                print("Still waiting for a GPIO trigger...")

        buffer_size = (
            capability.sResolutionRange.iWidthMax
            * capability.sResolutionRange.iHeightMax
            * (1 if mono_camera else 3)
        )
        frame_buffer = mvsdk.CameraAlignMalloc(buffer_size, 16)
        check_status(
            mvsdk.CameraImageProcess(
                camera_handle, raw_buffer, frame_buffer, frame_head
            ),
            "CameraImageProcess",
        )

        channels = 1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
        image_size = frame_head.iWidth * frame_head.iHeight * channels
        image_bytes = ctypes.string_at(frame_buffer, image_size)
        latest_measurement = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            (frame_head.iHeight, frame_head.iWidth, channels)
        ).copy()
        return latest_measurement
    finally:
        if raw_buffer is not None:
            mvsdk.CameraReleaseImageBuffer(camera_handle, raw_buffer)
        if frame_buffer is not None:
            mvsdk.CameraAlignFree(frame_buffer)


def main():
    global latest_measurement

    camera_handle = None
    try:
        device = find_camera(CAMERA_INDEX)
        camera_handle = mvsdk.CameraInit(device, -1, -1)
        capability = mvsdk.CameraGetCapability(camera_handle)

        configure_external_trigger(camera_handle)
        check_status(mvsdk.CameraPlay(camera_handle), "CameraPlay")

        print("Waiting for a GPIO trigger...")
        latest_measurement = grab_triggered_frame(camera_handle, capability)
        print(
            "Received one frame: "
            f"{latest_measurement.shape[1]}x{latest_measurement.shape[0]}"
        )
        return latest_measurement
    finally:
        if camera_handle is not None:
            mvsdk.CameraUnInit(camera_handle)


if __name__ == "__main__":
    main()
