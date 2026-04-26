"""Tests for audit engine (Python mirror of Rust blockchain)."""

import hashlib
import json
import pytest
from datetime import datetime


class TestAuditChainPython:
    """Python implementation of the audit chain for testing."""

    def test_sha256_chain(self):
        """Verify SHA-256 hash chain integrity."""
        blocks = []
        prev_hash = "0"

        for i in range(5):
            data = {
                "index": i,
                "timestamp": datetime.now().isoformat(),
                "operation": f"insert_{i}",
                "prev_hash": prev_hash,
            }
            raw = json.dumps(data, sort_keys=True).encode()
            block_hash = hashlib.sha256(raw).hexdigest()
            data["hash"] = block_hash
            blocks.append(data)
            prev_hash = block_hash

        # Verify chain
        for i, block in enumerate(blocks):
            assert block["prev_hash"] == ("0" if i == 0 else blocks[i-1]["hash"])
            # Recompute hash
            check_data = {k: v for k, v in block.items() if k != "hash"}
            check_hash = hashlib.sha256(
                json.dumps(check_data, sort_keys=True).encode()
            ).hexdigest()
            assert check_hash == block["hash"]

    def test_tamper_detection(self):
        """Detect tampered block in chain."""
        blocks = []
        prev_hash = "0"

        for i in range(3):
            data = {"index": i, "prev_hash": prev_hash, "op": f"op_{i}"}
            raw = json.dumps(data, sort_keys=True).encode()
            block_hash = hashlib.sha256(raw).hexdigest()
            data["hash"] = block_hash
            blocks.append(data)
            prev_hash = block_hash

        # Tamper with middle block
        original_hash = blocks[1]["hash"]
        blocks[1]["op"] = "TAMPERED"
        tampered_hash = hashlib.sha256(
            json.dumps({k: v for k, v in blocks[1].items() if k != "hash"},
                       sort_keys=True).encode()
        ).hexdigest()

        assert tampered_hash != original_hash
        # Next block's prev_hash no longer matches
        assert blocks[2]["prev_hash"] == original_hash
        assert blocks[2]["prev_hash"] != tampered_hash


class TestRestorationAudit:
    """Test restoration-specific audit entries."""

    def test_restoration_log_entry(self):
        entry = {
            "mural_id": "cave_45_north_wall",
            "defect_type": "flaking",
            "method": "inpainting",
            "reference_id": "cave_45_south_wall",
            "expert_id": "dr_zhang",
            "confidence": 0.92,
        }
        raw = json.dumps(entry, sort_keys=True).encode()
        h = hashlib.sha256(raw).hexdigest()
        assert len(h) == 64

    def test_defect_types(self):
        defect_types = ["flaking", "saline", "hollowing", "cracking", "fading", "mold"]
        for dt in defect_types:
            entry = {"defect_type": dt, "mural_id": "test"}
            h = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            assert len(h) == 64
