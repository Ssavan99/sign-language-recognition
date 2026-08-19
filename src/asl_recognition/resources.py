"""Host and process memory reporting used to keep local runs within budget.

Training this project's full split is a multi-hour job on an ordinary desktop.
The numbers here exist so a run can be judged against the machine it is on
rather than started hopefully and killed halfway through. Nothing in this module
requires a third-party dependency, so it stays usable in CI and in a minimal
install.
"""

from __future__ import annotations

import sys
from typing import Any

# Refuse to start when the host has less head-room than this. The figure is
# measured, not guessed: a CPU run of this pipeline peaked at 0.70 GiB resident
# and 1.24 GiB commit, so 1.5 GiB leaves roughly a 2x margin over observed peak
# resident use while staying reachable on a 16 GB desktop with a browser open.
#
# This guards against *starting* a run the host cannot support. It cannot
# guarantee one finishes: other applications may grow after the check passes.
DEFAULT_MINIMUM_AVAILABLE_BYTES = 1536 * 1024**2


class InsufficientMemoryError(RuntimeError):
    """Raised when a run would start with too little host memory to be safe."""


def _windows_process_memory() -> dict[str, int] | None:
    import ctypes
    import ctypes.wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
    except (AttributeError, OSError):  # pragma: no cover - depends on host API
        return None
    return {
        "resident_bytes": int(counters.WorkingSetSize),
        "peak_resident_bytes": int(counters.PeakWorkingSetSize),
        "commit_bytes": int(counters.PrivateUsage),
    }


def _windows_available_memory() -> int | None:
    import ctypes
    import ctypes.wintypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.wintypes.DWORD),
            ("dwMemoryLoad", ctypes.wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32")
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except (AttributeError, OSError):  # pragma: no cover - depends on host API
        return None
    return int(status.ullAvailPhys)


def _proc_status_kilobytes(field: str) -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1])
    except (OSError, IndexError, ValueError):  # pragma: no cover - depends on host
        return None
    return None


def _linux_process_memory() -> dict[str, int] | None:
    resident = _proc_status_kilobytes("VmRSS")
    if resident is None:
        return None
    peak = _proc_status_kilobytes("VmHWM")
    result = {"resident_bytes": resident * 1024}
    if peak is not None:
        result["peak_resident_bytes"] = peak * 1024
    return result


def _linux_available_memory() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):  # pragma: no cover - depends on host
        return None
    return None


def _rusage_peak_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS and the BSDs report bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def process_memory() -> dict[str, int]:
    """Report this process's memory use, omitting fields the host cannot supply.

    Callers must treat every key as optional. A missing key means the platform
    did not expose the number, never that the number was zero.
    """

    if sys.platform == "win32":
        windows = _windows_process_memory()
        if windows is not None:
            return windows
        return {}
    linux = _linux_process_memory() if sys.platform.startswith("linux") else None
    if linux is not None:
        return linux
    peak = _rusage_peak_bytes()
    return {"peak_resident_bytes": peak} if peak is not None else {}


def available_memory() -> int | None:
    """Return host memory available for a new allocation, or None if unknown."""

    if sys.platform == "win32":
        return _windows_available_memory()
    if sys.platform.startswith("linux"):
        return _linux_available_memory()
    return None


def memory_report() -> dict[str, Any]:
    """Combine process and host memory into one JSON-serialisable record."""

    report: dict[str, Any] = dict(process_memory())
    available = available_memory()
    if available is not None:
        report["host_available_bytes"] = available
    return report


def check_available_memory(
    minimum_available_bytes: int = DEFAULT_MINIMUM_AVAILABLE_BYTES,
    *,
    allow_low_memory: bool = False,
) -> dict[str, Any]:
    """Refuse to start when the host is too loaded for the run to finish safely.

    A previous full-split attempt on the reference machine destabilised the
    desktop rather than failing cleanly, so this check is a guard against
    starting work that cannot finish, not a performance tuning knob.
    """

    if minimum_available_bytes < 0:
        raise ValueError("minimum_available_bytes must be non-negative")
    report = memory_report()
    report["minimum_available_bytes"] = int(minimum_available_bytes)
    report["allow_low_memory"] = bool(allow_low_memory)
    available = report.get("host_available_bytes")
    if available is None:
        report["preflight"] = "unknown"
        return report
    if available >= minimum_available_bytes:
        report["preflight"] = "ok"
        return report
    report["preflight"] = "overridden" if allow_low_memory else "insufficient"
    if allow_low_memory:
        return report
    raise InsufficientMemoryError(
        f"host has {available / 1024**3:.2f} GiB available memory but this run needs at least "
        f"{minimum_available_bytes / 1024**3:.2f} GiB. Close other applications, or pass "
        "--allow-low-memory to start anyway."
    )


__all__ = [
    "DEFAULT_MINIMUM_AVAILABLE_BYTES",
    "InsufficientMemoryError",
    "available_memory",
    "check_available_memory",
    "memory_report",
    "process_memory",
]
