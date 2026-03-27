"""eichi_utils.host_memory の単体テスト"""

import os
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
spec = importlib.util.spec_from_file_location(
    "host_memory", os.path.join(ROOT, "webui", "eichi_utils", "host_memory.py")
)
host_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host_memory)


class TestHostMemAvailableGb:
    def test_returns_float_or_none(self):
        result = host_memory.host_mem_available_gb()
        assert result is None or isinstance(result, float)

    def test_positive_if_available(self):
        result = host_memory.host_mem_available_gb()
        if result is not None:
            assert result > 0

    def test_reasonable_range(self):
        """Available RAM should be between 0 and 4096 GB if returned."""
        result = host_memory.host_mem_available_gb()
        if result is not None:
            assert 0 < result < 4096


class TestHostMemTotalGb:
    def test_returns_float_or_none(self):
        result = host_memory.host_mem_total_gb()
        assert result is None or isinstance(result, float)

    def test_positive_if_available(self):
        result = host_memory.host_mem_total_gb()
        if result is not None:
            assert result > 0

    def test_total_gte_available(self):
        total = host_memory.host_mem_total_gb()
        avail = host_memory.host_mem_available_gb()
        if total is not None and avail is not None:
            assert total >= avail


class TestHostMemSnapshot:
    def test_returns_dict(self):
        snap = host_memory.host_mem_snapshot()
        assert isinstance(snap, dict)
        assert "avail_gb" in snap
        assert "total_gb" in snap

    def test_values_consistent(self):
        snap = host_memory.host_mem_snapshot()
        if snap["avail_gb"] is not None and snap["total_gb"] is not None:
            assert snap["total_gb"] >= snap["avail_gb"]
