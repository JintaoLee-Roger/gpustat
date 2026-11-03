"""Regression coverage for Windows, process churn and dependency compatibility."""
from io import StringIO
from types import SimpleNamespace

import psutil
import pytest
from mockito import unstub, when

import gpustat
from gpustat import core, nvml
from gpustat.test_gpustat import _configure_mock, mock_gpu_handles


@pytest.fixture(autouse=True)
def mocked_devices():
    core.GPUStatCollection.global_processes.clear()
    _configure_mock()
    yield
    unstub()
    core.GPUStatCollection.global_processes.clear()


def fake_process(name='python.exe', username='PC\\alice', cmdline=None):
    return SimpleNamespace(
        username=lambda: username,
        name=lambda: name,
        cmdline=lambda: cmdline or [],
        cpu_percent=lambda: 12.5,
        memory_percent=lambda: 1.0,
    )


@pytest.mark.parametrize('name', ['dwm.exe', 'DWM.EXE', 'explorer.exe',
                                 'ShellExperienceHost.exe', 'svchost.exe',
                                 'SearchHost.exe', 'TextInputHost.exe'])
def test_windows_filter_and_override(monkeypatch, name):
    monkeypatch.setattr(core, 'IS_WINDOWS', True)
    gpu = gpustat.new_query()[0]
    normal = dict(gpu.processes[0], command='python.exe')
    gpu.processes[:] = [normal, dict(normal, command=name, pid=123)]
    output = StringIO()
    stats = core.GPUStatCollection([gpu])
    stats.print_formatted(output, no_color=True, show_cmd=True, show_user=True,
                          show_pid=True, show_header=False)
    assert '/48448(' in output.getvalue()
    assert '/123(' not in output.getvalue()
    output = StringIO()
    stats.print_formatted(output, no_color=True, show_cmd=True, show_pid=True,
                          show_system_processes=True, show_header=False)
    assert '/123(' in output.getvalue()
    # JSON remains raw so scripting and diagnosis do not lose records.
    assert len(stats.jsonify()['gpus'][0]['processes']) == 2


@pytest.mark.parametrize('name', ['chrome.exe', 'Code.exe', 'WindowsTerminal.exe',
                                 'python.exe', 'my-dwm.exe'])
def test_windows_keeps_user_applications(monkeypatch, name):
    monkeypatch.setattr(core, 'IS_WINDOWS', True)
    assert not core.is_system_process({'command': name, 'username': 'PC\\alice'})
    assert not core.is_system_process({'command': '?', 'username': '?'})


def test_windows_command_line_access_denied(monkeypatch):
    monkeypatch.setattr(core, 'IS_WINDOWS', True)
    process = fake_process(name='DWM.EXE')

    def denied():
        raise psutil.AccessDenied(48448)

    process.cmdline = denied
    when(psutil).Process(pid=48448).thenReturn(process)
    row = gpustat.new_query()[0].processes[0]
    assert row['command'] == 'DWM.EXE'
    assert row['gpu_memory_usage'] == 4000
    assert core.is_system_process(row)


def test_windows_path_and_wddm_memory(monkeypatch):
    monkeypatch.setattr(core, 'IS_WINDOWS', True)
    when(psutil).Process(pid=48448).thenReturn(
        fake_process(cmdline=[r'C:\Program Files\Python\python.exe', 'train.py']))
    when(nvml.pynvml).nvmlDeviceGetComputeRunningProcesses(mock_gpu_handles[0]).thenReturn([
        SimpleNamespace(pid=48448, usedGpuMemory=(1 << 64) - 1)])
    when(nvml.pynvml).nvmlDeviceGetGraphicsRunningProcesses(mock_gpu_handles[0]).thenReturn([])
    gpu = gpustat.new_query()[0]
    row = gpu.processes[0]
    assert row['command'] == 'python.exe'
    assert row['username'] == 'PC\\alice'
    assert row['gpu_memory_usage'] is None
    output = StringIO()
    gpu.print_to(output, with_colors=False, show_cmd=True, show_user=True, show_pid=True)
    assert 'PC\\alice:python.exe/48448(?M)' in output.getvalue()


def test_recycled_pid_refreshes_metadata():
    when(psutil).Process(pid=48448).thenReturn(fake_process(username='old'))
    assert gpustat.new_query()[0].processes[0]['username'] == 'old'
    when(psutil).Process(pid=48448).thenReturn(fake_process(username='new'))
    assert gpustat.new_query()[0].processes[0]['username'] == 'new'


def test_multi_gpu_cpu_sampling_waits_once(monkeypatch):
    sleeps = []
    monkeypatch.setattr(core.time, 'sleep', sleeps.append)
    shared = SimpleNamespace(pid=48448, usedGpuMemory=100 * core.MB)
    when(nvml.pynvml).nvmlDeviceGetComputeRunningProcesses(mock_gpu_handles[1]).thenReturn([shared])
    stats = gpustat.new_query()
    assert sleeps == [0.1]
    first = stats[0].processes[0]
    second = next(p for p in stats[1].processes if p['pid'] == first['pid'])
    assert first['cpu_percent'] == second['cpu_percent']


def test_unsupported_memory_keeps_other_metrics():
    when(nvml.pynvml).nvmlDeviceGetMemoryInfo(mock_gpu_handles[0]).thenRaise(
        nvml.pynvml.NVMLError_NotSupported())
    stats = gpustat.new_query()
    assert stats[0].memory_used is None
    assert stats[0].memory_total is None
    assert stats[0].memory_free is None
    assert stats[0].utilization == 76
    assert len(stats[0].processes) == 2
    assert stats[1].memory_used == 9000


def test_memory_v2_not_supported_falls_back(monkeypatch):
    patch = nvml.pynvml_monkeypatch
    monkeypatch.setattr(patch, 'has_memoryinfo_v2', True)
    attempts = []

    def memory(handle, **kwargs):
        attempts.append(kwargs)
        if kwargs:
            raise nvml.pynvml.NVMLError_NotSupported()
        return 'v1-memory'

    monkeypatch.setattr(patch, 'original_nvmlDeviceGetMemoryInfo', memory)
    assert patch.nvmlDeviceGetMemoryInfo('handle') == 'v1-memory'
    assert attempts == [{'version': nvml.pynvml.nvmlMemory_v2}, {}]


def test_missing_process_api_uses_versioned_binding():
    when(nvml.pynvml).nvmlDeviceGetComputeRunningProcesses('handle').thenRaise(
        nvml.pynvml.NVMLError_FunctionNotFound())
    when(nvml.pynvml, strict=False).nvmlDeviceGetComputeRunningProcesses_v2('handle').thenReturn(['v2'])
    assert nvml.nvmlDeviceGetComputeRunningProcesses('handle') == ['v2']


def test_compact_output_keeps_lost_gpu_error():
    gpu = core.InvalidGPU(0, '((GPU is lost))', nvml.pynvml.NVMLError_GpuIsLost())
    output = StringIO()
    gpu.print_to(output, with_colors=False)
    assert '((GPU is lost))' in output.getvalue()


def test_process_list_growth_retries():
    when(nvml.pynvml).nvmlDeviceGetComputeRunningProcesses('handle').thenRaise(
        nvml.pynvml.NVMLError_InsufficientSize()).thenReturn(['process'])
    assert nvml.nvmlDeviceGetComputeRunningProcesses('handle') == ['process']


def test_process_list_growth_retry_is_bounded():
    when(nvml.pynvml).nvmlDeviceGetComputeRunningProcesses('handle').thenRaise(
        nvml.pynvml.NVMLError_InsufficientSize())
    with pytest.raises(nvml.pynvml.NVMLError_InsufficientSize):
        nvml.nvmlDeviceGetComputeRunningProcesses('handle')
