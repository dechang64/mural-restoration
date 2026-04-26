"""
Digital Collectible (数字藏品) minting engine for Mural Guardian.

Mints unique digital certificates for mural restoration artifacts.
Each certificate records:
    - Mural provenance (cave, wall, dynasty)
    - Defect type and severity
    - Restoration method and reference
    - DINOv2 feature hash (style fingerprint)
    - Blockchain hash (audit chain link)
    - Rarity tier

The certificate is the collectible — not the image itself.
This avoids copyright issues while preserving cultural provenance.
"""

import hashlib
import json
import uuid
import time
import base64
from enum import IntEnum
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


class RarityTier(IntEnum):
    """Rarity tiers for digital collectibles."""
    COMMON = 0       # 常见 — common defects on well-preserved murals
    UNCOMMON = 1     # 罕见 — moderate defects
    RARE = 2         # 稀有 — severe defects requiring expert intervention
    EPIC = 3         # 史诗 — multi-defect restoration
    LEGENDARY = 4    # 传说 — restoration of lost/vanished murals

    @property
    def label_cn(self) -> str:
        return ["常见", "罕见", "稀有", "史诗", "传说"][self]

    @property
    def label_en(self) -> str:
        return ["Common", "Uncommon", "Rare", "Epic", "Legendary"][self]

    @property
    def color(self) -> str:
        return ["#8B9DAF", "#4CAF50", "#2196F3", "#9C27B0", "#FF9800"][self]

    @property
    def max_supply(self) -> int:
        return [10000, 5000, 1000, 100, 10][self]


# Defect severity weights for rarity calculation
DEFECT_SEVERITY = {
    "fading": 1,      # 褪色 — common
    "flaking": 2,     # 起甲 — moderate
    "cracking": 2,    # 裂隙 — moderate
    "mold": 4,        # 霉变 — severe
    "saline": 5,      # 酥碱 — critical
    "hollowing": 5,   # 空鼓 — critical
}

# Dynasty era weights (older = rarer)
DYNASTY_WEIGHT = {
    "modern": 1,
    "qing": 1,
    "ming": 1,
    "yuan": 2,
    "song": 2,
    "five_dynasties": 3,
    "tang": 4,       # 唐代壁画最珍贵
    "sui": 5,
    "northern_wei": 6,
    "sixteen_kingdoms": 7,
}


@dataclass
class MuralProvenance:
    """Cultural provenance of the source mural."""
    cave_id: str = ""         # e.g., "cave_45"
    wall: str = ""            # e.g., "north", "south", "east", "west", "ceiling"
    dynasty: str = ""         # e.g., "tang", "song"
    location: str = ""        # e.g., "莫高窟"
    period: str = ""          # e.g., "盛唐 (705-781)"
    description: str = ""     # e.g., "观无量寿经变"


@dataclass
class RestorationRecord:
    """Record of the restoration process."""
    defect_type: str = ""     # from DefectType enum
    defect_severity: str = "" # "minor", "major", "critical"
    method: str = ""          # "inpainting", "color_transfer", etc.
    reference_id: str = ""    # reference mural used for style matching
    expert_id: str = ""       # expert who approved
    confidence: float = 0.0   # model confidence
    processing_time_ms: float = 0.0


@dataclass
class FeatureFingerprint:
    """DINOv2 feature fingerprint for style verification."""
    feature_hash: str = ""    # SHA-256 of the feature vector
    feature_dim: int = 768
    model_name: str = "dinov2_vitb14"
    similarity_threshold: float = 0.85


@dataclass
class CollectibleMetadata:
    """ERC-721 compatible metadata for the digital collectible."""
    name: str = ""
    description: str = ""
    image: str = ""           # base64 encoded or IPFS URI
    external_url: str = ""
    attributes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DigitalCollectible:
    """A complete digital collectible (数字藏品).

    The collectible IS the certificate, not the image.
    It proves: "This restoration artifact was generated from
    Cave X, Wall Y, using method Z, verified by feature hash H."
    """
    # Identity
    token_id: str = ""
    mint_tx_hash: str = ""    # on-chain transaction hash
    mint_timestamp: str = ""

    # Provenance
    provenance: MuralProvenance = field(default_factory=MuralProvenance)

    # Restoration
    restoration: RestorationRecord = field(default_factory=RestorationRecord)

    # Feature fingerprint
    fingerprint: FeatureFingerprint = field(default_factory=FeatureFingerprint)

    # Rarity
    rarity: RarityTier = RarityTier.COMMON
    edition: int = 1          # edition number (1/1000)
    max_edition: int = 10000  # max supply for this rarity

    # Audit chain link
    audit_block_hash: str = ""
    audit_block_index: int = 0

    # Metadata (ERC-721 compatible)
    metadata: CollectibleMetadata = field(default_factory=CollectibleMetadata)

    def compute_token_hash(self) -> str:
        """Compute the unique token hash from all fields."""
        data = {
            "provenance": asdict(self.provenance),
            "restoration": asdict(self.restoration),
            "fingerprint": asdict(self.fingerprint),
            "rarity": self.rarity.value,
            "edition": self.edition,
            "audit_block_hash": self.audit_block_hash,
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_certificate_json(self) -> str:
        """Export as JSON certificate."""
        return json.dumps({
            "token_id": self.token_id,
            "token_hash": self.compute_token_hash(),
            "mint_tx_hash": self.mint_tx_hash,
            "mint_timestamp": self.mint_timestamp,
            "provenance": asdict(self.provenance),
            "restoration": asdict(self.restoration),
            "fingerprint": asdict(self.fingerprint),
            "rarity": {
                "tier": self.rarity.value,
                "label_cn": self.rarity.label_cn,
                "label_en": self.rarity.label_en,
                "color": self.rarity.color,
            },
            "edition": f"{self.edition}/{self.max_edition}",
            "audit_block_hash": self.audit_block_hash,
            "audit_block_index": self.audit_block_index,
            "metadata": asdict(self.metadata),
        }, indent=2, ensure_ascii=False)

    def verify(self) -> Tuple[bool, str]:
        """Verify the integrity of this collectible.

        Returns:
            (is_valid, reason)
        """
        if not self.token_id:
            return False, "Missing token_id"
        if not self.mint_timestamp:
            return False, "Missing mint_timestamp"
        if not self.fingerprint.feature_hash:
            return False, "Missing feature_hash"
        if not self.audit_block_hash:
            return False, "Missing audit_block_hash"
        return True, "Valid"


class CollectibleMinter:
    """Mints digital collectibles from mural restoration artifacts.

    Pipeline:
        1. Receive restoration result + provenance
        2. Compute feature fingerprint
        3. Determine rarity tier
        4. Generate token ID and metadata
        5. Link to audit chain
        6. Output digital collectible certificate
    """

    def __init__(self, chain_gateway: Optional[Any] = None):
        """
        Args:
            chain_gateway: Optional blockchain gateway for on-chain minting.
                          If None, uses mock mode (off-chain certificates).
        """
        self._chain = chain_gateway
        self._minted: Dict[str, DigitalCollectible] = {}
        self._edition_counters: Dict[str, int] = {}

    @property
    def total_minted(self) -> int:
        return len(self._minted)

    def compute_rarity(self, provenance: MuralProvenance,
                       restoration: RestorationRecord) -> RarityTier:
        """Determine rarity tier from provenance and restoration.

        Rarity is based on:
        - Defect severity (higher = rarer)
        - Dynasty age (older = rarer)
        - Number of defects (more = rarer)
        """
        defect_score = DEFECT_SEVERITY.get(restoration.defect_type, 1)
        dynasty_score = DYNASTY_WEIGHT.get(provenance.dynasty, 1)
        severity_mult = {"minor": 1, "major": 1.5, "critical": 2.0}
        sev_mult = severity_mult.get(restoration.defect_severity, 1.0)

        score = defect_score * dynasty_score * sev_mult

        if score >= 40:
            return RarityTier.LEGENDARY
        elif score >= 15:
            return RarityTier.EPIC
        elif score >= 4:
            return RarityTier.RARE
        elif score >= 2:
            return RarityTier.UNCOMMON
        else:
            return RarityTier.COMMON

    def compute_feature_hash(self, feature_vector) -> str:
        """Compute SHA-256 hash of a feature vector.

        Args:
            feature_vector: numpy array or list of floats
        """
        import numpy as np
        if hasattr(feature_vector, 'tobytes'):
            raw = feature_vector.astype(np.float32).tobytes()
        else:
            raw = json.dumps(feature_vector).encode()
        return hashlib.sha256(raw).hexdigest()

    def generate_token_id(self, provenance: MuralProvenance,
                          rarity: RarityTier) -> str:
        """Generate a unique token ID.

        Format: MG-{cave}-{rarity_code}-{uuid_short}
        Example: MG-cave45-R-a1b2c3d4
        """
        rarity_codes = ["C", "U", "R", "E", "L"]
        code = rarity_codes[rarity.value]
        short_uuid = uuid.uuid4().hex[:8]
        cave = provenance.cave_id.replace("cave_", "").replace("_", "")
        return f"MG-{cave}-{code}-{short_uuid}"

    def generate_metadata(self, collectible: DigitalCollectible) -> CollectibleMetadata:
        """Generate ERC-721 compatible metadata."""
        prov = collectible.provenance
        rest = collectible.restoration
        rarity = collectible.rarity

        name = f"壁画守护者 #{collectible.token_id}"
        if prov.cave_id and prov.wall:
            name = f"{prov.cave_id} {prov.wall} · {rarity.label_cn}修复"

        desc_parts = [
            f"来自{prov.location or prov.cave_id}{prov.wall}的{rest.defect_type}修复",
            f"朝代: {prov.dynasty}",
            f"修复方法: {rest.method}",
            f"稀有度: {rarity.label_cn}",
            f"版本: {collectible.edition}/{collectible.max_edition}",
        ]
        if prov.description:
            desc_parts.insert(0, prov.description)

        attributes = [
            {"trait_type": "Cave", "value": prov.cave_id},
            {"trait_type": "Wall", "value": prov.wall},
            {"trait_type": "Dynasty", "value": prov.dynasty},
            {"trait_type": "Defect Type", "value": rest.defect_type},
            {"trait_type": "Severity", "value": rest.defect_severity},
            {"trait_type": "Method", "value": rest.method},
            {"trait_type": "Rarity", "value": rarity.label_en},
            {"trait_type": "Edition", "value": f"{collectible.edition}/{collectible.max_edition}"},
            {"trait_type": "Confidence", "display_type": "number", "value": rest.confidence},
        ]
        if prov.period:
            attributes.insert(2, {"trait_type": "Period", "value": prov.period})

        return CollectibleMetadata(
            name=name,
            description="\n".join(desc_parts),
            image="",  # populated by caller
            attributes=attributes,
        )

    def mint(self,
             provenance: MuralProvenance,
             restoration: RestorationRecord,
             feature_vector=None,
             audit_block_hash: str = "",
             audit_block_index: int = 0,
             image_b64: str = "") -> DigitalCollectible:
        """Mint a digital collectible.

        Args:
            provenance: Cultural provenance of the source mural
            restoration: Record of the restoration process
            feature_vector: DINOv2 feature vector (for fingerprinting)
            audit_block_hash: Link to the audit chain block
            audit_block_index: Audit chain block index
            image_b64: Base64 encoded restoration result image

        Returns:
            DigitalCollectible with certificate
        """
        # 1. Determine rarity
        rarity = self.compute_rarity(provenance, restoration)

        # 2. Compute feature fingerprint
        feature_hash = ""
        feature_dim = 768
        if feature_vector is not None:
            feature_hash = self.compute_feature_hash(feature_vector)
            if hasattr(feature_vector, 'shape'):
                feature_dim = feature_vector.shape[-1]

        # 3. Generate token ID
        token_id = self.generate_token_id(provenance, rarity)

        # 4. Track edition
        rarity_key = f"{provenance.cave_id}_{rarity.label_en}"
        self._edition_counters[rarity_key] = self._edition_counters.get(rarity_key, 0) + 1
        edition = self._edition_counters[rarity_key]

        # 5. Build collectible
        collectible = DigitalCollectible(
            token_id=token_id,
            mint_timestamp=datetime.now(timezone.utc).isoformat(),
            provenance=provenance,
            restoration=restoration,
            fingerprint=FeatureFingerprint(
                feature_hash=feature_hash,
                feature_dim=feature_dim,
            ),
            rarity=rarity,
            edition=edition,
            max_edition=rarity.max_supply,
            audit_block_hash=audit_block_hash,
            audit_block_index=audit_block_index,
        )

        # 6. Generate metadata
        collectible.metadata = self.generate_metadata(collectible)
        if image_b64:
            collectible.metadata.image = image_b64

        # 7. Simulate on-chain minting
        if self._chain:
            collectible.mint_tx_hash = self._chain.mint(collectible)
        else:
            # Mock: deterministic hash as tx hash
            collectible.mint_tx_hash = "0x" + hashlib.sha256(
                f"{token_id}:{collectible.mint_timestamp}".encode()
            ).hexdigest()[:64]

        # 8. Store
        self._minted[token_id] = collectible

        return collectible

    def verify_collectible(self, token_id: str) -> Tuple[bool, str]:
        """Verify a previously minted collectible.

        Returns:
            (is_valid, reason_or_certificate_json)
        """
        if token_id not in self._minted:
            return False, f"Token {token_id} not found"
        collectible = self._minted[token_id]
        return collectible.verify()

    def get_collectible(self, token_id: str) -> Optional[DigitalCollectible]:
        """Retrieve a minted collectible by token ID."""
        return self._minted.get(token_id)

    def list_minted(self, rarity: Optional[RarityTier] = None,
                    cave_id: Optional[str] = None) -> List[DigitalCollectible]:
        """List minted collectibles with optional filters."""
        results = list(self._minted.values())
        if rarity is not None:
            results = [c for c in results if c.rarity == rarity]
        if cave_id is not None:
            results = [c for c in results if c.provenance.cave_id == cave_id]
        return results

    def rarity_distribution(self) -> Dict[str, int]:
        """Get distribution of minted collectibles by rarity."""
        dist = {tier.label_cn: 0 for tier in RarityTier}
        for c in self._minted.values():
            dist[c.rarity.label_cn] += 1
        return dist
