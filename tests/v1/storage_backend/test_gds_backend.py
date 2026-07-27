# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest import mock
import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.gds_backend import (
    _DATA_FILE_SUFFIX,
    _METADATA_FILE_SUFFIX,
    GdsBackend,
    get_extra_config_bool,
    pack_metadata,
    unpack_metadata,
)
from tests.v1.utils import create_test_memory_obj, has_cufile, has_hipfile

# Optional override for tempfile root. In CI we point this at a GDS-capable
# host-backed mount (see .buildkite/k3_tests/unit/run.sh); locally, it's
# unset and tempfile falls back to its default (usually /tmp). cuFile needs
# direct-I/O-capable storage (ext4/xfs on real disk); overlayfs and tmpfs
# fail with CU_FILE_IO_NOT_SUPPORTED (err=5027).
_TEST_TMPDIR = os.environ.get("LMCACHE_TEST_TMPDIR") or None


def create_test_config(gds_path: str, gds_path_sharding: str = "by_gpu"):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        gds_path=gds_path,
        gds_path_sharding=gds_path_sharding,
        lmcache_instance_id="test_instance",
        gds_buffer_size=256,
        extra_config={"use_direct_io": True},
    )
    return config


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    # NO UNDERSCORE HERE for model_name
    return CacheEngineKey(
        model_name="testmodel",
        world_size=3,
        worker_id=1,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


def create_test_layer_key(key_id: int = 0, layer_id: int = 0) -> LayerCacheEngineKey:
    """Create a LayerCacheEngineKey for testing layer-wise GDS operations."""
    return LayerCacheEngineKey(
        model_name="testmodel",
        world_size=3,
        worker_id=1,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
        layer_id=layer_id,
    )


def create_test_metadata():
    """Create a test metadata for LMCacheMetadata."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
    )


@pytest.fixture
def temp_gds_path():
    temp_dir = tempfile.mkdtemp(dir=_TEST_TMPDIR)
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def async_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


@pytest.fixture
def gds_backend(temp_gds_path, async_loop):
    config = create_test_config(temp_gds_path)
    metadata = create_test_metadata()
    return GdsBackend(
        config=config,
        loop=async_loop,
        metadata=metadata,
        dst_device=f"{torch_device_type}:0",
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not (has_cufile() or has_hipfile()),
    reason="Requires NVIDIA cuFile (libcufile.so) or AMD hipFile (libhipfile.so). "
    "Skipping on systems without GDS support.",
)
@pytest.mark.skipif(sys.platform != "linux", reason="TestGdsBackend runs only on Linux")
class TestGdsBackend:
    def test_init(self, temp_gds_path, async_loop):
        config = create_test_config(temp_gds_path)
        metadata = create_test_metadata()
        backend = GdsBackend(
            config=config,
            loop=async_loop,
            metadata=metadata,
            dst_device=f"{torch_device_type}:0",
        )
        assert backend.gds_path == temp_gds_path
        assert backend.dst_device == f"{torch_device_type}:0"
        assert os.path.exists(temp_gds_path)

    def test_str(self, gds_backend):
        assert str(gds_backend) == "GdsBackend"

    def test_key_to_path_and_insert_key(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj(device=torch_device_type)
        gds_backend.insert_key(key, memory_obj)
        # Check that the key is in hot_cache
        assert key in gds_backend.hot_cache
        meta = gds_backend.hot_cache[key]
        assert meta.shape == memory_obj.metadata.shape
        assert meta.dtype == memory_obj.metadata.dtype

    def test_contains_key_not_exists(self, gds_backend):
        key = create_test_key(1)
        assert not gds_backend.contains(key)
        assert not gds_backend.contains(key, pin=True)

    def test_contains_key_exists(self, gds_backend):
        key = create_test_key(0)
        memory_obj = create_test_memory_obj(device=torch_device_type)
        gds_backend.insert_key(key, memory_obj)
        assert gds_backend.contains(key)
        assert gds_backend.contains(key, pin=True)

    def test_exists_in_put_tasks(self, gds_backend):
        key = create_test_key(0)
        assert not gds_backend.exists_in_put_tasks(key)
        # Simulate adding to put_tasks
        with gds_backend.put_lock:
            gds_backend.put_tasks.add(key)
        assert gds_backend.exists_in_put_tasks(key)

@pytest.mark.asyncio
@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "linux", reason="Runs only on Linux")
class TestGdsMultiPath:
    """Tests for multi-path GDS backend support.

    Verifies comma-separated gds_path parsing, GPU-to-NVMe affinity
    (by_gpu sharding), directory creation for all paths, and backward
    compatibility with single-path configs.
    """

    @staticmethod
    def _make_backend(
        gds_path: str,
        dst_device: str,
        async_loop,
        gds_path_sharding: str = "by_gpu",
    ):
        """Create a GdsBackend with mocked allocator and fstype.

        Mocks are used so the tests run without cuFile / real NVMe.
        """

        class DummyAllocator:
            def __init__(self):
                self.base_pointer = 0

            def close(self):
                pass

        config = create_test_config(gds_path, gds_path_sharding=gds_path_sharding)
        metadata = create_test_metadata()
        with (
            mock.patch(
                "lmcache.v1.storage_backend.gds_backend.get_fstype",
                return_value="tmpfs",
            ),
            mock.patch.object(
                GdsBackend,
                "initialize_allocator",
                return_value=DummyAllocator(),
            ),
            mock.patch(
                "lmcache.v1.storage_backend.gds_backend.ctypes.CDLL",
                return_value=mock.MagicMock(),
            ),
        ):
            backend = GdsBackend(
                config=config,
                loop=async_loop,
                metadata=metadata,
                dst_device=dst_device,
            )
        return backend

    def test_single_path_backward_compat(self, temp_gds_path, async_loop):
        """A single gds_path (no commas) behaves like before."""
        backend = self._make_backend(temp_gds_path, f"{torch_device_type}:0", async_loop)
        try:
            assert backend.gds_paths == [temp_gds_path]
            assert backend.gds_path == temp_gds_path
        finally:
            backend.close()

    def test_multi_path_parsing(self, async_loop):
        """Comma-separated paths are split into gds_paths list."""
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(3)]
        try:
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, f"{torch_device_type}:0", async_loop)
            try:
                assert backend.gds_paths == paths
                assert len(backend.gds_paths) == 3
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_multi_path_parsing_with_spaces(self, async_loop):
        """Spaces around commas are stripped."""
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(2)]
        try:
            gds_path = f"  {paths[0]}  ,  {paths[1]}  "
            backend = self._make_backend(gds_path, f"{torch_device_type}:0", async_loop)
            try:
                assert backend.gds_paths == paths
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_gpu_affinity_selects_path(self, async_loop):
        """Different accelerator device indices select different paths via modulo."""
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(4)]
        try:
            gds_path = ",".join(paths)
            for device_id in range(8):
                dst = f"{torch_device_type}:{device_id}"
                backend = self._make_backend(gds_path, dst, async_loop)
                try:
                    expected = paths[device_id % 4]
                    assert backend.gds_path == expected, (
                        f"{torch_device_type}:{device_id} should map to {expected}, "
                        f"got {backend.gds_path}"
                    )
                finally:
                    backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_all_directories_created(self, async_loop):
        """All paths in gds_paths get their directories created."""
        base = tempfile.mkdtemp(dir=_TEST_TMPDIR)
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, f"{torch_device_type}:0", async_loop)
            try:
                for p in paths:
                    assert os.path.isdir(p), f"{p} should exist"
            finally:
                backend.close()
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_key_to_path_uses_selected_path(self, async_loop):
        """_key_to_path builds file paths under the selected gds_path."""
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(2)]
        try:
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, f"{torch_device_type}:0", async_loop)
            try:
                key = create_test_key(0)
                file_path, _, _, _ = backend._key_to_path(key)
                assert file_path.startswith(backend.gds_path)
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_deterministic_path_selection(self, async_loop):
        """Same device always selects the same path."""
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(3)]
        try:
            gds_path = ",".join(paths)
            selected = set()
            for _ in range(5):
                backend = self._make_backend(
                    gds_path,
                    f"{torch_device_type}:1",
                    async_loop,
                )
                try:
                    selected.add(backend.gds_path)
                finally:
                    backend.close()
            assert len(selected) == 1, "Path selection should be deterministic"
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_scan_discovers_entries_across_all_paths(self, async_loop):
        """Startup scan finds metadata written under a non-affinity path.

        {torch_device_type}:0 has affinity to paths[0], but a metadata file is placed
        under paths[1].  The scan should still discover it because
        ``_scan_metadata`` iterates all ``gds_paths``.
        """
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(2)]
        try:
            # chunk_hash must have >=4 decimal digits so l1/l2 dirs
            # are each 2 chars (the scanner filters on len == 2).
            key = CacheEngineKey(
                model_name="testmodel",
                world_size=3,
                worker_id=1,
                chunk_hash=1234,
                dtype=torch.bfloat16,
            )
            hash_str = str(key.chunk_hash)
            l1_dir = hash_str[:2]
            l2_dir = hash_str[2:4]
            key_str = urllib.parse.quote(key.to_string(), safe="")

            # Build directory structure under paths[1] (non-affinity path)
            subdir = os.path.join(paths[1], l1_dir, l2_dir)
            os.makedirs(subdir, exist_ok=True)

            # Write a valid metadata file
            dummy_tensor = torch.zeros(2, 256, 8, 128, dtype=torch.bfloat16)
            metadata_bytes = pack_metadata(
                dummy_tensor,
                fmt=MemoryFormat.KV_2LTD,
                lmcache_version="1",
            )
            meta_path = os.path.join(
                subdir,
                key_str + _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX,
            )
            with open(meta_path, "wb") as f:
                f.write(metadata_bytes)

            # Create backend — {torch_device_type}:0 affinity selects paths[0]
            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, f"{torch_device_type}:0", async_loop)
            try:
                assert backend.gds_path == paths[0]

                # Wait for the async _scan_metadata to finish
                time.sleep(1)

                # contains() would NOT find this via _try_to_read_metadata
                # (which only checks the affinity path).  If it returns
                # True, the cross-path scan populated hot_cache.
                assert backend.contains(key), (
                    "Scan should discover metadata across all gds_paths"
                )
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_try_to_read_metadata_finds_across_all_paths(self, async_loop):
        """contains() fallback finds metadata on a non-affinity path.

        After clearing hot_cache (so the startup scan results are gone),
        ``contains()`` falls back to ``_try_to_read_metadata`` which
        should search all ``gds_paths``, not just the affinity path.
        """
        paths = [tempfile.mkdtemp(dir=_TEST_TMPDIR) for _ in range(2)]
        try:
            key = CacheEngineKey(
                model_name="testmodel",
                world_size=3,
                worker_id=1,
                chunk_hash=1234,
                dtype=torch.bfloat16,
            )
            hash_str = str(key.chunk_hash)
            l1_dir = hash_str[:2]
            l2_dir = hash_str[2:4]
            key_str = urllib.parse.quote(key.to_string(), safe="")

            # Place metadata under paths[1] (non-affinity path)
            subdir = os.path.join(paths[1], l1_dir, l2_dir)
            os.makedirs(subdir, exist_ok=True)

            dummy_tensor = torch.zeros(2, 256, 8, 128, dtype=torch.bfloat16)
            metadata_bytes = pack_metadata(
                dummy_tensor,
                fmt=MemoryFormat.KV_2LTD,
                lmcache_version="1",
            )
            meta_path = os.path.join(
                subdir,
                key_str + _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX,
            )
            with open(meta_path, "wb") as f:
                f.write(metadata_bytes)

            gds_path = ",".join(paths)
            backend = self._make_backend(gds_path, f"{torch_device_type}:0", async_loop)
            try:
                assert backend.gds_path == paths[0]

                # Wait for async scan, then clear its results
                time.sleep(1)
                with backend.hot_lock:
                    backend.hot_cache.clear()

                # contains() now relies solely on _try_to_read_metadata
                assert backend.contains(key), (
                    "_try_to_read_metadata should search all gds_paths"
                )
            finally:
                backend.close()
        finally:
            for p in paths:
                shutil.rmtree(p, ignore_errors=True)

    def test_gds_path_sharding_default(self, temp_gds_path, async_loop):
        """Default gds_path_sharding is 'by_gpu' (backend inits OK)."""
        backend = self._make_backend(temp_gds_path, f"{torch_device_type}:0", async_loop)
        try:
            assert backend.gds_path == temp_gds_path
        finally:
            backend.close()

    def test_gds_path_sharding_explicit_by_gpu(self, temp_gds_path, async_loop):
        """Explicitly setting gds_path_sharding='by_gpu' works."""
        backend = self._make_backend(
            temp_gds_path,
            f"{torch_device_type}:0",
            async_loop,
            gds_path_sharding="by_gpu",
        )
        try:
            assert backend.gds_path == temp_gds_path
        finally:
            backend.close()

    def test_gds_path_sharding_unsupported_raises(self, temp_gds_path, async_loop):
        """Unsupported gds_path_sharding value raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported path sharding strategy"):
            self._make_backend(
                temp_gds_path,
                f"{torch_device_type}:0",
                async_loop,
                gds_path_sharding="round_robin",
            )


def test_get_extra_config_bool_valid_inputs():
    """String, uppercase string, literal bool, and missing key all parse correctly."""
    config = create_test_config("/tmp")
    config.extra_config = {
        "str_true": "true",
        "str_false": "FALSE",
        "lit_true": True,
        "lit_false": False,
    }
    assert get_extra_config_bool("str_true", config) is True
    assert get_extra_config_bool("str_false", config) is False
    assert get_extra_config_bool("lit_true", config) is True
    assert get_extra_config_bool("lit_false", config) is False
    assert get_extra_config_bool("missing", config) is None


def test_get_extra_config_bool_invalid_value_raises():
    """Non-bool non-string value raises RuntimeError."""
    config = create_test_config("/tmp")
    config.extra_config = {"flag": 42}
    with pytest.raises(RuntimeError, match="Invalid value"):
        get_extra_config_bool("flag", config)


def test_pack_unpack_metadata_multiple_dtypes():
    """pack_metadata / unpack_metadata roundtrip is correct for several dtypes."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int8]:
        tensor = torch.zeros(4, 8, dtype=dtype)
        packed = pack_metadata(tensor, fmt=MemoryFormat.KV_2LTD, tag="test")
        shape, out_dtype, nbytes, fmt, extra = unpack_metadata(packed)
        assert shape == tensor.shape
        assert out_dtype == dtype
        assert nbytes == tensor.numel() * tensor.element_size()
        assert fmt == MemoryFormat.KV_2LTD
        assert extra["tag"] == "test"
