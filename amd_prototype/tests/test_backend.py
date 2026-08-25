import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flash_attention_amd_prototype import backend


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def current_device(self) -> int:
        return 0

    def get_device_name(self, device: int) -> str:
        del device
        return "AMD Instinct Test GPU"

    def get_device_properties(self, device: int) -> SimpleNamespace:
        del device
        return SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")

    def is_bf16_supported(self) -> bool:
        return True


class BackendProbeTests(unittest.TestCase):
    def test_missing_torch_has_actionable_reason(self) -> None:
        with patch.object(
            backend,
            "_module_available",
            side_effect=lambda name: name == "triton",
        ):
            capability = backend.probe_rocm()

        self.assertFalse(capability.available)
        self.assertEqual(capability.reason, "PyTorch is not installed")
        self.assertTrue(capability.triton_available)

    def test_rocm_runtime_reports_architecture(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(hip="test-rocm"),
            cuda=_FakeCuda(available=True),
        )
        with (
            patch.object(backend, "_module_available", return_value=True),
            patch.object(backend, "import_module", return_value=fake_torch),
        ):
            capability = backend.probe_rocm()

        self.assertTrue(capability.available)
        self.assertEqual(capability.architecture, "gfx942:sramecc+:xnack-")
        self.assertEqual(capability.hip_version, "test-rocm")
        self.assertTrue(capability.bf16_supported)

    def test_non_rocm_pytorch_is_rejected(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(hip=None),
            cuda=_FakeCuda(available=True),
        )
        with (
            patch.object(backend, "_module_available", return_value=True),
            patch.object(backend, "import_module", return_value=fake_torch),
        ):
            capability = backend.probe_rocm()

        self.assertFalse(capability.available)
        self.assertIn("does not include ROCm", capability.reason or "")


if __name__ == "__main__":
    unittest.main()