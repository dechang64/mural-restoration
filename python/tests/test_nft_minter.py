"""Tests for CollectibleMinter (数字藏品铸造引擎)."""

import numpy as np
import pytest
import json
from analysis.nft_minter import (
    CollectibleMinter, DigitalCollectible, MuralProvenance,
    RestorationRecord, FeatureFingerprint, RarityTier,
    CollectibleMetadata, DEFECT_SEVERITY, DYNASTY_WEIGHT,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tang_provenance():
    return MuralProvenance(
        cave_id="cave_45",
        wall="north",
        dynasty="tang",
        location="莫高窟",
        period="盛唐 (705-781)",
        description="观无量寿经变",
    )


@pytest.fixture
def common_restoration():
    return RestorationRecord(
        defect_type="fading",
        defect_severity="minor",
        method="color_transfer",
        confidence=0.88,
        processing_time_ms=1200.0,
    )


@pytest.fixture
def critical_restoration():
    return RestorationRecord(
        defect_type="saline",
        defect_severity="critical",
        method="inpainting",
        reference_id="cave_45_south",
        expert_id="dr_zhang",
        confidence=0.92,
        processing_time_ms=3500.0,
    )


@pytest.fixture
def feature_vector():
    np.random.seed(42)
    vec = np.random.randn(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def minter():
    return CollectibleMinter()


# ============================================================
# RarityTier Tests
# ============================================================

class TestRarityTier:
    def test_all_tiers_exist(self):
        assert len(RarityTier) == 5

    def test_labels_cn(self):
        assert RarityTier.COMMON.label_cn == "常见"
        assert RarityTier.LEGENDARY.label_cn == "传说"

    def test_labels_en(self):
        assert RarityTier.RARE.label_en == "Rare"
        assert RarityTier.EPIC.label_en == "Epic"

    def test_colors(self):
        assert len(RarityTier.COMMON.color) == 7  # hex color
        assert RarityTier.COMMON.color.startswith("#")

    def test_max_supply_decreasing(self):
        supplies = [tier.max_supply for tier in RarityTier]
        for i in range(len(supplies) - 1):
            assert supplies[i] > supplies[i + 1]

    def test_ordering(self):
        assert RarityTier.COMMON < RarityTier.UNCOMMON < RarityTier.RARE
        assert RarityTier.RARE < RarityTier.EPIC < RarityTier.LEGENDARY


# ============================================================
# Rarity Calculation Tests
# ============================================================

class TestRarityCalculation:
    def test_common_fading_modern(self, minter):
        prov = MuralProvenance(cave_id="cave_1", wall="north", dynasty="modern")
        rest = RestorationRecord(defect_type="fading", defect_severity="minor", method="inpainting")
        assert minter.compute_rarity(prov, rest) == RarityTier.COMMON

    def test_uncommon_flaking_ming(self, minter):
        prov = MuralProvenance(cave_id="cave_100", wall="south", dynasty="ming")
        rest = RestorationRecord(defect_type="flaking", defect_severity="major", method="inpainting")
        assert minter.compute_rarity(prov, rest) == RarityTier.UNCOMMON

    def test_rare_cracking_tang(self, minter):
        prov = MuralProvenance(cave_id="cave_45", wall="east", dynasty="tang")
        rest = RestorationRecord(defect_type="cracking", defect_severity="major", method="texture_synthesis")
        assert minter.compute_rarity(prov, rest) == RarityTier.RARE

    def test_epic_mold_song(self, minter):
        prov = MuralProvenance(cave_id="cave_55", wall="ceiling", dynasty="song")
        rest = RestorationRecord(defect_type="mold", defect_severity="critical", method="inpainting")
        assert minter.compute_rarity(prov, rest) == RarityTier.EPIC

    def test_legendary_saline_tang_critical(self, minter):
        prov = MuralProvenance(cave_id="cave_45", wall="north", dynasty="tang")
        rest = RestorationRecord(defect_type="saline", defect_severity="critical", method="inpainting")
        assert minter.compute_rarity(prov, rest) == RarityTier.LEGENDARY

    def test_legendary_hollowing_northern_wei(self, minter):
        prov = MuralProvenance(cave_id="cave_259", wall="west", dynasty="northern_wei")
        rest = RestorationRecord(defect_type="hollowing", defect_severity="critical", method="inpainting")
        assert minter.compute_rarity(prov, rest) == RarityTier.LEGENDARY


# ============================================================
# Feature Hash Tests
# ============================================================

class TestFeatureHash:
    def test_deterministic(self, minter, feature_vector):
        h1 = minter.compute_feature_hash(feature_vector)
        h2 = minter.compute_feature_hash(feature_vector)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_vectors(self, minter):
        v1 = np.zeros(768, dtype=np.float32)
        v2 = np.ones(768, dtype=np.float32)
        assert minter.compute_feature_hash(v1) != minter.compute_feature_hash(v2)

    def test_list_input(self, minter):
        h = minter.compute_feature_hash([1.0, 2.0, 3.0])
        assert len(h) == 64


# ============================================================
# Token ID Tests
# ============================================================

class TestTokenId:
    def test_format(self, minter, tang_provenance):
        tid = minter.generate_token_id(tang_provenance, RarityTier.RARE)
        assert tid.startswith("MG-45-R-")
        assert len(tid.split("-")) == 4

    def test_unique(self, minter, tang_provenance):
        ids = set()
        for _ in range(100):
            tid = minter.generate_token_id(tang_provenance, RarityTier.COMMON)
            ids.add(tid)
        assert len(ids) == 100  # all unique

    def test_rarity_codes(self, minter):
        prov = MuralProvenance(cave_id="cave_1", wall="north", dynasty="tang")
        codes = []
        for tier in RarityTier:
            tid = minter.generate_token_id(prov, tier)
            code = tid.split("-")[2]
            codes.append(code)
        assert codes == ["C", "U", "R", "E", "L"]


# ============================================================
# Minting Tests
# ============================================================

class TestMinting:
    def test_mint_basic(self, minter, tang_provenance, common_restoration):
        c = minter.mint(tang_provenance, common_restoration)
        assert isinstance(c, DigitalCollectible)
        assert c.token_id.startswith("MG-")
        assert c.mint_timestamp != ""
        assert c.mint_tx_hash.startswith("0x")
        assert c.rarity == RarityTier.RARE  # tang + fading = 4*1*1 = 4 → RARE
        assert c.edition == 1

    def test_mint_with_feature(self, minter, tang_provenance, critical_restoration, feature_vector):
        c = minter.mint(
            tang_provenance, critical_restoration,
            feature_vector=feature_vector,
            audit_block_hash="abc123",
            audit_block_index=42,
        )
        assert c.fingerprint.feature_hash != ""
        assert c.fingerprint.feature_dim == 768
        assert c.audit_block_hash == "abc123"
        assert c.audit_block_index == 42
        assert c.rarity == RarityTier.LEGENDARY

    def test_mint_increments_edition(self, minter, tang_provenance, common_restoration):
        c1 = minter.mint(tang_provenance, common_restoration)
        c2 = minter.mint(tang_provenance, common_restoration)
        assert c2.edition == c1.edition + 1

    def test_mint_tracks_total(self, minter, tang_provenance, common_restoration):
        assert minter.total_minted == 0
        minter.mint(tang_provenance, common_restoration)
        assert minter.total_minted == 1
        minter.mint(tang_provenance, common_restoration)
        assert minter.total_minted == 2

    def test_mint_generates_metadata(self, minter, tang_provenance, common_restoration):
        c = minter.mint(tang_provenance, common_restoration)
        assert c.metadata.name != ""
        assert c.metadata.description != ""
        assert len(c.metadata.attributes) >= 8
        # Check trait types
        trait_types = [a["trait_type"] for a in c.metadata.attributes]
        assert "Cave" in trait_types
        assert "Dynasty" in trait_types
        assert "Rarity" in trait_types
        assert "Defect Type" in trait_types

    def test_mint_with_image(self, minter, tang_provenance, common_restoration):
        fake_b64 = "data:image/png;base64,iVBORw0KGgo="
        c = minter.mint(tang_provenance, common_restoration, image_b64=fake_b64)
        assert c.metadata.image == fake_b64


# ============================================================
# Verification Tests
# ============================================================

class TestVerification:
    def test_verify_valid(self, minter, tang_provenance, common_restoration, feature_vector):
        c = minter.mint(tang_provenance, common_restoration,
                        feature_vector=feature_vector,
                        audit_block_hash="hash123", audit_block_index=1)
        valid, reason = c.verify()
        assert valid
        assert reason == "Valid"

    def test_verify_missing_token_id(self):
        c = DigitalCollectible()
        valid, reason = c.verify()
        assert not valid
        assert "token_id" in reason

    def test_verify_missing_feature_hash(self, minter, tang_provenance, common_restoration):
        c = minter.mint(tang_provenance, common_restoration)
        c.fingerprint.feature_hash = ""
        valid, reason = c.verify()
        assert not valid

    def test_verify_by_minter(self, minter, tang_provenance, common_restoration, feature_vector):
        c = minter.mint(tang_provenance, common_restoration,
                        feature_vector=feature_vector,
                        audit_block_hash="0xabc123", audit_block_index=1)
        valid, reason = minter.verify_collectible(c.token_id)
        assert valid

    def test_verify_unknown_token(self, minter):
        valid, reason = minter.verify_collectible("nonexistent")
        assert not valid
        assert "not found" in reason


# ============================================================
# Certificate JSON Tests
# ============================================================

class TestCertificate:
    def test_certificate_json_valid(self, minter, tang_provenance, common_restoration, feature_vector):
        c = minter.mint(tang_provenance, common_restoration,
                        feature_vector=feature_vector,
                        audit_block_hash="abc", audit_block_index=5)
        cert_json = c.to_certificate_json()
        cert = json.loads(cert_json)

        assert cert["token_id"] == c.token_id
        assert cert["token_hash"] == c.compute_token_hash()
        assert cert["rarity"]["label_cn"] == "稀有"
        assert cert["rarity"]["tier"] == RarityTier.RARE.value
        assert cert["provenance"]["cave_id"] == "cave_45"
        assert cert["provenance"]["dynasty"] == "tang"
        assert cert["restoration"]["defect_type"] == "fading"
        assert cert["edition"] == "1/1000"
        assert cert["audit_block_hash"] == "abc"

    def test_certificate_deterministic_hash(self, minter, tang_provenance, common_restoration):
        c = minter.mint(tang_provenance, common_restoration)
        h1 = c.compute_token_hash()
        h2 = c.compute_token_hash()
        assert h1 == h2
        assert len(h1) == 64


# ============================================================
# Query Tests
# ============================================================

class TestQueries:
    def test_list_all(self, minter, tang_provenance, common_restoration):
        minter.mint(tang_provenance, common_restoration)
        minter.mint(tang_provenance, common_restoration)
        assert len(minter.list_minted()) == 2

    def test_filter_by_rarity(self, minter, tang_provenance, common_restoration, critical_restoration):
        minter.mint(tang_provenance, common_restoration)
        minter.mint(tang_provenance, critical_restoration)
        legendary = minter.list_minted(rarity=RarityTier.LEGENDARY)
        assert len(legendary) == 1

    def test_filter_by_cave(self, minter, tang_provenance, common_restoration):
        minter.mint(tang_provenance, common_restoration)
        other_prov = MuralProvenance(cave_id="cave_100", wall="south", dynasty="song")
        minter.mint(other_prov, common_restoration)
        cave45 = minter.list_minted(cave_id="cave_45")
        assert len(cave45) == 1

    def test_rarity_distribution(self, minter, tang_provenance, common_restoration, critical_restoration):
        minter.mint(tang_provenance, common_restoration)
        minter.mint(tang_provenance, critical_restoration)
        dist = minter.rarity_distribution()
        assert dist["稀有"] == 1
        assert dist["传说"] == 1
        assert dist["常见"] == 0

    def test_get_collectible(self, minter, tang_provenance, common_restoration):
        c = minter.mint(tang_provenance, common_restoration)
        retrieved = minter.get_collectible(c.token_id)
        assert retrieved is not None
        assert retrieved.token_id == c.token_id

    def test_get_nonexistent(self, minter):
        assert minter.get_collectible("fake") is None


# ============================================================
# Defect/Dynasty Weight Tests
# ============================================================

class TestWeights:
    def test_defect_severity_complete(self):
        assert set(DEFECT_SEVERITY.keys()) == {
            "fading", "flaking", "cracking", "mold", "saline", "hollowing"
        }

    def test_dynasty_weights_complete(self):
        assert set(DYNASTY_WEIGHT.keys()) == {
            "modern", "qing", "ming", "yuan", "song",
            "five_dynasties", "tang", "sui", "northern_wei", "sixteen_kingdoms"
        }

    def test_tang_is_most_valued(self):
        # Tang is the most valued among common mural eras (not counting pre-Sui)
        common_eras = ["qing", "ming", "yuan", "song", "five_dynasties", "tang"]
        assert DYNASTY_WEIGHT["tang"] >= max(DYNASTY_WEIGHT[e] for e in common_eras)

    def test_saline_and_hollowing_are_most_severe(self):
        max_sev = max(DEFECT_SEVERITY.values())
        assert DEFECT_SEVERITY["saline"] == max_sev
        assert DEFECT_SEVERITY["hollowing"] == max_sev
