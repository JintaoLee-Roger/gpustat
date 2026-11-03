"""Imports pynvml with sanity checks and custom patches."""

# pylint: disable=protected-access

import atexit
import ctypes
import functools
import os
import sys
import textwrap
import warnings

# If this environment variable is set, we will bypass pynvml version validation
# so that legacy pynvml (nvidia-ml-py3) can be used. This would be useful
# in a case where there are conflicts on pynvml dependencies.
# However, beware that pynvml might produce wrong results (see #107).
ALLOW_LEGACY_PYNVML = os.getenv("ALLOW_LEGACY_PYNVML", "")
ALLOW_LEGACY_PYNVML = ALLOW_LEGACY_PYNVML.lower() not in ('false', '0', '')


try:
    # Check pynvml version: we require 11.450.129 or newer.
    # https://github.com/wookayin/gpustat/pull/107
    import pynvml
    if not (
        # Requires nvidia-ml-py >= 11.460.79
        hasattr(pynvml, 'NVML_BRAND_NVIDIA_RTX') or
        # Requires nvidia-ml-py >= 11.450.129, < 11.510.69
        hasattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses_v2')
    ) and not ALLOW_LEGACY_PYNVML:
        raise ImportError("pynvml library is outdated.")

    if not hasattr(pynvml, '_nvmlGetFunctionPointer'):
        raise ImportError(
            """pynvml appears to be a non-official package.

            The PyPI package `pynvml` should not be used.
            Please use the official NVIDIA bindings `nvidia-ml-py`.
            """)

except (ImportError, SyntaxError, RuntimeError) as e:
    _pynvml = sys.modules.get('pynvml', None)

    raise ImportError(textwrap.dedent(
        """\
        pynvml is missing or an outdated version is installed.

        gpustat requires nvidia-ml-py>=11.450.129, and the *official* NVIDIA python bindings
        should be used; neither nvidia-ml-py3 nor gpuopenanalytics/pynvml is compatible.

        For more details, please refer to:
            https://github.com/wookayin/gpustat/issues/107
            https://github.com/wookayin/gpustat/issues/153

        The root cause: """ + str(e) +
        """

        Your pynvml installation: """ + repr(_pynvml) +
        """

        -----------------------------------------------------------
        (Suggested Fix) Please reinstall `gpustat`:

        $ pip install --force-reinstall gpustat

        If it still does not fix the problem, please uninstall `pynvml` packages and reinstall `nvidia-ml-py` manually:

        $ pip uninstall nvidia-ml-py3 pynvml
        $ pip install --force-reinstall --ignore-installed 'nvidia-ml-py'
        """)) from e


class NvidiaCompatibilityWarning(UserWarning):
    pass


def check_driver_nvml_version(driver_version_str: str):
    """Show warnings when an incompatible driver is used."""

    def safeint(v) -> int:
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    driver_version = tuple(safeint(v) for v in
                           driver_version_str.strip().split("."))

    # #161: invalid process information on 535.xx
    is_pynvml_535_77 = (
        hasattr(pynvml.c_nvmlProcessInfo_t, 'usedGpuCcProtectedMemory') and
        # Note: __name__ changed to pynvml.c_nvmlProcessInfo_v2_t since 12.535.108+
        pynvml.c_nvmlProcessInfo_t.__name__ == 'c_nvmlProcessInfo_t'
    )

    if (535, 43) <= driver_version < (535, 86):
        # See #161: these are buggy, gives wrong process information
        # except for nvidia-ml-py == 12.535.77 (which is a buggy version too).
        # Note: NVIDIA 535.86+ and nvidia-ml-py 12.535.108+ fixes the bug
        if not is_pynvml_535_77:
            warnings.warn(
                f"This version of NVIDIA Driver {driver_version_str} is incompatible, "
                "process information will be inaccurate. "
                "Upgrade the NVIDIA driver to 535.104.05 or higher, "
                "or use nvidia-ml-py==12.535.77. For more details, see "
                "https://github.com/wookayin/gpustat/issues/161.",
                category=NvidiaCompatibilityWarning, stacklevel=2)
    else:
        if is_pynvml_535_77:   # pynvml 12.535.77 should not be used
            warnings.warn(
                "This version of nvidia-ml-py (possibly 12.535.77) is incompatible. "
                "Please upgrade nvidia-ml-py to the latest version. "
                "(pip install --upgrade --force-reinstall nvidia-ml-py)",
                category=NvidiaCompatibilityWarning, stacklevel=2)


# Monkey-patch nvml due to breaking changes in pynvml.
# See #107, #141, and test_gpustat.py for more details.

_original_nvmlGetFunctionPointer = pynvml._nvmlGetFunctionPointer
_original_nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo


class pynvml_monkeypatch:

    @staticmethod  # Note: must be defined as a staticmethod to allow mocking.
    def original_nvmlGetFunctionPointer(name):
        return _original_nvmlGetFunctionPointer(name)

    @staticmethod
    def _nvmlGetFunctionPointer(name):
        # Never substitute a different API version's C function pointer: the
        # Python binding must allocate the matching process-info structure.
        return pynvml_monkeypatch.original_nvmlGetFunctionPointer(name)

    @staticmethod  # Note: must be defined as a staticmethod to allow mocking.
    def original_nvmlDeviceGetMemoryInfo(*args, **kwargs):
        return _original_nvmlDeviceGetMemoryInfo(*args, **kwargs)

    has_memoryinfo_v2 = None

    @staticmethod
    @functools.wraps(pynvml.nvmlDeviceGetMemoryInfo)
    def nvmlDeviceGetMemoryInfo(handle):
        """A patched version of nvmlDeviceGetMemoryInfo.

        This tries `version=N.nvmlMemory_v2` if the nvmlDeviceGetMemoryInfo_v2
        function is available (for driver >= 515), or fallback to the legacy
        v1 API for (driver < 515) to yield a correct result. See #141.
        """
        M = pynvml_monkeypatch

        if M.has_memoryinfo_v2 is None:
            try:
                pynvml._nvmlGetFunctionPointer("nvmlDeviceGetMemoryInfo_v2")
                M.has_memoryinfo_v2 = True
            except pynvml.NVMLError_FunctionNotFound:  # type: ignore
                M.has_memoryinfo_v2 = False

        if hasattr(pynvml, 'nvmlMemory_v2'):  # pynvml >= 11.510.69
            try:
                memory = M.original_nvmlDeviceGetMemoryInfo(
                    handle, version=pynvml.nvmlMemory_v2)
            except (pynvml.NVMLError_FunctionNotFound, pynvml.NVMLError_NotSupported):
                # Older drivers/devices may expose v2 but not support it.
                memory = M.original_nvmlDeviceGetMemoryInfo(handle)
        else:
            if M.has_memoryinfo_v2:
                warnings.warn(
                    "Your NVIDIA driver requires a compatible version of "
                    "pynvml (>= 11.510.69) installed to display the correct "
                    "memory usage information (See #141 for more details). "
                    "Please try `pip install --upgrade nvidia-ml-py`.",
                    category=UserWarning)
            memory = M.original_nvmlDeviceGetMemoryInfo(handle)

        return memory


setattr(pynvml, '_nvmlGetFunctionPointer', pynvml_monkeypatch._nvmlGetFunctionPointer)
setattr(pynvml, 'nvmlDeviceGetMemoryInfo', pynvml_monkeypatch.nvmlDeviceGetMemoryInfo)


def _running_processes(handle, name):
    # Processes can start between NVML's size and data calls. A transient
    # insufficient buffer must not turn the whole process list into "N/A".
    for attempt in range(3):
        try:
            return _running_processes_once(handle, name)
        except pynvml.NVMLError_InsufficientSize:
            if attempt == 2:
                raise


def _running_processes_once(handle, name):
    try:
        return getattr(pynvml, name)(handle)
    except (pynvml.NVMLError_FunctionNotFound,
            getattr(pynvml, 'NVMLLibraryMismatchError', pynvml.NVMLError_FunctionNotFound)):
        # Recent bindings translate missing unversioned APIs to a mismatch
        # exception. The versioned v2 API still works on these older drivers.
        # Use the public binding so its array layout matches the v2 ABI.
        fallback = getattr(pynvml, name + '_v2', None)
        if fallback is not None:
            return fallback(handle)
        return _running_processes_v2(handle, name)


def _running_processes_v2(handle, name):
    """Older bindings omit the public v2 function; use its explicit ABI.

    Never pass a v3 array to the v2 function. NVML can grow the process list
    between the size query and the data query, so allow bounded retries.
    """
    fn = pynvml._nvmlGetFunctionPointer(name + '_v2')
    count = ctypes.c_uint(0)
    result = fn(handle, ctypes.byref(count), None)
    if result == pynvml.NVML_SUCCESS:
        return []
    if result != pynvml.NVML_ERROR_INSUFFICIENT_SIZE:
        pynvml._nvmlCheckReturn(result)
    for _ in range(3):
        count.value = count.value * 2 + 5
        array = (pynvml.c_nvmlProcessInfo_v2_t * count.value)()
        result = fn(handle, ctypes.byref(count), array)
        if result == pynvml.NVML_ERROR_INSUFFICIENT_SIZE:
            continue
        pynvml._nvmlCheckReturn(result)
        processes = [pynvml.nvmlStructToFriendlyObject(array[i])
                     for i in range(count.value)]
        for process in processes:
            if process.usedGpuMemory == (1 << 64) - 1:
                process.usedGpuMemory = None
        return processes
    pynvml._nvmlCheckReturn(result)


def nvmlDeviceGetComputeRunningProcesses(handle):
    return _running_processes(handle, 'nvmlDeviceGetComputeRunningProcesses')


def nvmlDeviceGetGraphicsRunningProcesses(handle):
    return _running_processes(handle, 'nvmlDeviceGetGraphicsRunningProcesses')


# Upon importing this module, let pynvml be initialized and remain active
# throughout the lifespan of the python process (until gpustat exists).
_initialized: bool
_init_error = None
try:
    pynvml.nvmlInit()
    _initialized = True

    def _shutdown():
        pynvml.nvmlShutdown()
    atexit.register(_shutdown)

except pynvml.NVMLError as exc:
    _initialized = False
    _init_error = exc


def ensure_initialized():
    if not _initialized:
        raise _init_error  # type: ignore


__all__ = [
    'pynvml',
    'check_driver_nvml_version',
    'ensure_initialized',
]
