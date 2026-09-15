"""Find out why CameraEnumerateDevice() returns an empty list.

Run this before trigger_camera.py whenever the camera is "not found". It walks
the chain from the USB bus up to the SDK and reports the first broken link:

    USB bus -> kernel driver -> udev permissions -> libMVSDK -> SDK enumeration

Usage:  python3 diagnose_camera.py
"""

import glob
import os
import platform
import re
import subprocess
import sys

UDEV_RULES = "/etc/udev/rules.d/88-mvusb.rules"

# MindVision USB cameras are NOT UVC devices: they speak a vendor protocol and
# are only reachable through libMVSDK. Anything that shows up as /dev/video* is
# almost certainly a different camera (a laptop webcam, for example).
KNOWN_MV_VENDOR_IDS = {"f622", "080b"}


def euid():
    """Effective UID, or None on platforms without one (Windows)."""
    return getattr(os, "geteuid", lambda: None)()


def section(title):
    print(f"\n=== {title} ===")


def run(cmd):
    """Run a command, return (ok, output)."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
        return out.returncode == 0, (out.stdout + out.stderr).strip()
    except FileNotFoundError:
        return False, f"{cmd[0]}: not installed"
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]}: timed out"


def vendor_ids_from_udev_rules():
    """Read the vendor IDs the SDK installer whitelisted, if the rules exist."""
    if not os.path.exists(UDEV_RULES):
        return set(), False
    with open(UDEV_RULES, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    ids = set(re.findall(r'idVendor\}?=="?([0-9a-fA-F]{4})', text))
    return {i.lower() for i in ids}, True


def check_usb_bus(vendor_ids):
    """Is the camera physically enumerated on the USB bus?

    Returns (on_bus, device_node) where device_node is /dev/bus/usb/BBB/DDD.
    """
    section("1. USB bus")
    ok, output = run(["lsusb"])
    if not ok and "not installed" in output:
        print("lsusb missing -> sudo apt install usbutils")
        return None, None
    print(output or "(no output)")

    candidates = [
        line
        for line in output.splitlines()
        if any(f" {vid}:" in line.lower() for vid in vendor_ids)
        or re.search(r"mindvision|mv-sua|aic-", line, re.IGNORECASE)
    ]
    if candidates:
        print("\n--> Likely MindVision device:")
        for line in candidates:
            print("   ", line)
        match = re.match(r"Bus (\d+) Device (\d+):", candidates[0])
        node = f"/dev/bus/usb/{match.group(1)}/{match.group(2)}" if match else None
        return True, node

    print(
        "\n--> No MindVision device on the bus.\n"
        "    The camera is not powered / not connected / the cable is charge-only.\n"
        "    Try another cable and another port, then re-run. Nothing below can\n"
        "    succeed until this line changes."
    )
    return False, None


def check_kernel_messages():
    section("2. Recent kernel USB messages")
    ok, output = run(["dmesg", "--level=err,warn,info", "--since", "-5min"])
    if not ok:
        ok, output = run(["journalctl", "-k", "--since", "-5 minutes", "--no-pager"])
    if not ok:
        print("Could not read kernel log (try: sudo dmesg | tail -40)")
        return
    lines = [ln for ln in output.splitlines() if re.search(r"usb|xhci", ln, re.I)]
    print("\n".join(lines[-20:]) or "(no recent USB events - replug the camera)")


def check_udev(vendor_ids, rules_exist, camera_node=None):
    section("3. udev rules and permissions")
    if rules_exist:
        print(
            f"{UDEV_RULES}: present, vendor IDs {sorted(vendor_ids) or '(none parsed)'}"
        )
    else:
        print(
            f"{UDEV_RULES}: MISSING\n"
            "--> Without this file the USB node stays root-only, so libMVSDK\n"
            "    cannot open the camera and enumeration returns an empty list.\n"
            "    Fix:  sudo cp 88-mvusb.rules /etc/udev/rules.d/\n"
            "          sudo udevadm control --reload-rules && sudo udevadm trigger\n"
            "          then physically replug the camera."
        )

    print(f"\nEffective UID: {euid()} ({'root' if euid() == 0 else 'normal user'})")
    ok, groups = run(["id", "-nG"])
    if ok:
        print(f"Groups: {groups}")
        if "plugdev" not in groups.split():
            print(
                "Note: not in 'plugdev'. The shipped rule grants access to that "
                "group: sudo usermod -aG plugdev $USER, then log out and back in."
            )

    print("\nPermissions on USB device nodes:")
    for node in sorted(glob.glob("/dev/bus/usb/*/*"))[:40]:
        try:
            st = os.stat(node)
        except OSError:
            continue
        writable = os.access(node, os.R_OK | os.W_OK)
        mark = "  <-- CAMERA" if node == camera_node else ""
        print(f"  {node}  mode={oct(st.st_mode & 0o777)}  rw_for_me={writable}{mark}")

    if camera_node and not os.access(camera_node, os.R_OK | os.W_OK):
        print(
            f"\n--> {camera_node} is the camera and is NOT writable by you.\n"
            "    This is the failure. libMVSDK reports it as\n"
            "    'user control fd open failed' and then enumerates nothing.\n"
            "    Install the udev rule above and replug the camera."
        )


def check_conflicting_drivers():
    section("4. Conflicting drivers / stale processes")
    nodes = sorted(glob.glob("/dev/video*"))
    print(f"/dev/video* nodes: {nodes or '(none)'}")
    if nodes:
        print(
            "  These are V4L2/UVC devices. A MindVision SUA camera does NOT create\n"
            "  one, so any node here belongs to a different camera (e.g. a webcam)."
        )

    ok, output = run(["lsmod"])
    if ok and re.search(r"^uvcvideo", output, re.M):
        print("uvcvideo is loaded (normal; only a problem if it claimed the camera)")

    # Only report processes that really hold a USB node open. Matching on
    # "python" alone flags every unrelated editor and system daemon.
    holders = []
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = fd_dir.split("/")[2]
        try:
            for fd in os.listdir(fd_dir):
                target = os.readlink(os.path.join(fd_dir, fd))
                if target.startswith("/dev/bus/usb/"):
                    holders.append((pid, target))
        except OSError:
            continue  # process exited, or not ours to inspect

    if holders:
        print("\nProcesses holding a USB node open (only one may own the camera):")
        for pid, target in holders:
            try:
                with open(f"/proc/{pid}/comm", encoding="utf-8") as handle:
                    name = handle.read().strip()
            except OSError:
                name = "?"
            print(f"  pid {pid} ({name}) -> {target}")
    else:
        print("No process currently holds a USB node open.")


def check_library():
    section("5. libMVSDK")
    override = os.environ.get("MVSDK_LIBRARY")
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libMVSDK.so")
    print(f"MVSDK_LIBRARY = {override or '(unset)'}")
    for path in ["/lib/libMVSDK.so", "/usr/lib/libMVSDK.so", default, override]:
        if path:
            print(f"  {path}: {'found' if os.path.exists(path) else 'missing'}")


def check_sdk():
    section("6. SDK enumeration")
    try:
        import mvsdk
    except Exception as error:  # noqa: BLE001 - we want to report any failure
        print(f"import mvsdk FAILED: {type(error).__name__}: {error}")
        return
    print(f"Loaded library object: {mvsdk._sdk}")

    devices = mvsdk.CameraEnumerateDevice()
    print(f"CameraEnumerateDevice() -> {len(devices)} device(s)")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev.GetFriendlyName()} | {dev.GetPortType()} | SN {dev.GetSn()}")

    if devices:
        print("\nThe SDK sees the camera. trigger_camera.py should work.")
        return

    if platform.system() != "Windows" and euid() != 0:
        print(
            "\nEmpty list as a normal user. Re-run as root to separate a permission\n"
            "problem from a connection problem:\n"
            "    sudo -E env PATH=$PATH $(which python3) diagnose_camera.py\n"
            "If root sees the camera, the udev rules are the problem (section 3)."
        )


def main():
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.version.split()[0]} ({platform.architecture()[0]})")

    if platform.system() == "Windows":
        print(
            "\nRunning on Windows. The USB/udev checks below are Linux-only; "
            "skipping to the SDK check."
        )
        check_sdk()
        return

    vendor_ids, rules_exist = vendor_ids_from_udev_rules()
    vendor_ids = vendor_ids or KNOWN_MV_VENDOR_IDS

    on_bus, camera_node = check_usb_bus(vendor_ids)
    if on_bus is False:
        check_kernel_messages()
    check_udev(vendor_ids, rules_exist, camera_node)
    check_conflicting_drivers()
    check_library()
    check_sdk()


if __name__ == "__main__":
    main()
