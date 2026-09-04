# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.xpu import XpuDeviceSpec


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="requires an Intel XPU runtime",
)
def test_xpu_event_backend_is_supported() -> None:
    spec = XpuDeviceSpec()
    backend = spec.event_ipc_backend
    assert backend.device_type == "xpu"
    device = torch.device("xpu", 0)

    backend.check_event_support(device)
    event = backend.create_event(device)
    backend.record_event(event, torch.xpu.current_stream())
    assert backend.query_event(event) in (True, False)
    assert backend.export_event(event, device) == b""
    assert isinstance(backend.import_event(b"", device), type(event))
    assert get_event_ipc_backend(device).device_type == "xpu"
