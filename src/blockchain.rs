//! Blockchain-style immutable audit log for Mural Guardian operations.
//!
//! Each operation is recorded as a "block" with:
//! - Timestamp
//! - Operation type and details (restoration-specific)
//! - Previous block's hash (chain integrity)
//! - Current block's hash (SHA-256)
//!
//! This provides tamper-evident logging for restoration decision traceability.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;

/// Mural defect types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DefectType {
    Flaking,       // 起甲
    Saline,        // 酥碱
    Hollowing,     // 空鼓
    Cracking,      // 裂隙
    Fading,        // 褪色
    Mold,          // 霉变
}

/// Restoration methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RestorationMethod {
    Inpainting,          // 修复绘制
    ColorTransfer,       // 色彩迁移
    TextureSynthesis,    // 纹理合成
    VirtualRestoration,  // 虚拟修复
}

/// Types of operations that can be logged
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationType {
    Insert { count: usize },
    Search { k: usize, result_count: usize },
    Delete { count: usize },
    Restoration {
        mural_id: String,
        defect_type: DefectType,
        method: RestorationMethod,
        reference_id: String,
        expert_id: String,
        confidence: f32,
    },
    System { message: String },
}

/// A single block in the audit chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block index (monotonically increasing)
    pub index: u64,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Operation type
    pub operation: OperationType,
    /// Hash of the previous block ("0" for genesis)
    pub prev_hash: String,
    /// Hash of this block
    pub hash: String,
    /// Optional nonce for future proof-of-work extension
    pub nonce: u64,
}

impl Block {
    fn compute_hash(index: u64, timestamp: &str, operation: &OperationType, prev_hash: &str, nonce: u64) -> String {
        let data = format!("{}:{}:{:?}:{}:{}", index, timestamp, operation, prev_hash, nonce);
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Create the genesis block
    pub fn genesis() -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let operation = OperationType::System {
            message: "Mural Guardian audit chain initialized".to_string(),
        };
        let hash = Self::compute_hash(0, &timestamp, &operation, "0", 0);
        Self {
            index: 0,
            timestamp,
            operation,
            prev_hash: "0".to_string(),
            hash,
            nonce: 0,
        }
    }

    /// Create a new block linked to the previous one
    pub fn new(index: u64, operation: OperationType, prev_hash: &str) -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let hash = Self::compute_hash(index, &timestamp, &operation, prev_hash, 0);
        Self {
            index,
            timestamp,
            operation,
            prev_hash: prev_hash.to_string(),
            hash,
            nonce: 0,
        }
    }

    /// Verify this block's hash integrity
    pub fn verify(&self) -> bool {
        let computed = Self::compute_hash(
            self.index,
            &self.timestamp,
            &self.operation,
            &self.prev_hash,
            self.nonce,
        );
        computed == self.hash
    }
}

/// The blockchain audit log
pub struct AuditChain {
    blocks: VecDeque<Block>,
    max_blocks: usize,
    log_path: Option<PathBuf>,
}

impl AuditChain {
    pub fn new(max_blocks: usize) -> Self {
        let mut chain = Self {
            blocks: VecDeque::with_capacity(max_blocks),
            max_blocks,
            log_path: None,
        };
        chain.blocks.push_back(Block::genesis());
        chain
    }

    pub fn with_log_file(mut self, path: PathBuf) -> Self {
        self.log_path = Some(path);
        self
    }

    pub fn append(&mut self, operation: OperationType) -> &Block {
        let prev = self.blocks.back().expect("chain always has genesis");
        let index = prev.index + 1;
        let block = Block::new(index, operation, &prev.hash);

        if let Some(ref path) = self.log_path {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{}", serde_json::to_string(&block).unwrap_or_default());
            }
        }

        if self.blocks.len() >= self.max_blocks {
            self.blocks.pop_front();
        }

        self.blocks.push_back(block);
        self.blocks.back().expect("just pushed")
    }

    pub fn verify_chain(&self) -> bool {
        let blocks: Vec<&Block> = self.blocks.iter().collect();
        if blocks.is_empty() {
            return true;
        }
        if !blocks[0].verify() {
            return false;
        }
        for window in blocks.windows(2) {
            if !window[1].verify() {
                return false;
            }
            if window[1].prev_hash != window[0].hash {
                return false;
            }
        }
        true
    }

    pub fn recent(&self, n: usize) -> Vec<&Block> {
        self.blocks.iter().rev().take(n).collect()
    }

    pub fn all(&self) -> &VecDeque<Block> {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.len() <= 1
    }

    pub fn latest_hash(&self) -> &str {
        &self.blocks.back().expect("chain always has genesis").hash
    }

    pub fn load_from_file(&mut self, path: &PathBuf) -> Result<usize, String> {
        if !path.exists() {
            return Ok(0);
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read log file: {}", e))?;
        let mut count = 0;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            if let Ok(block) = serde_json::from_str::<Block>(line) {
                if block.index == 0 && !self.blocks.is_empty() { continue; }
                self.blocks.push_back(block);
                count += 1;
            }
        }
        while self.blocks.len() > self.max_blocks {
            self.blocks.pop_front();
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_block() {
        let block = Block::genesis();
        assert_eq!(block.index, 0);
        assert_eq!(block.prev_hash, "0");
        assert!(block.verify());
    }

    #[test]
    fn test_chain_integrity() {
        let mut chain = AuditChain::new(100);
        chain.append(OperationType::Insert { count: 10 });
        chain.append(OperationType::Search { k: 5, result_count: 5 });
        chain.append(OperationType::Delete { count: 2 });
        assert!(chain.verify_chain());
        assert_eq!(chain.len(), 4);
    }

    #[test]
    fn test_restoration_audit() {
        let mut chain = AuditChain::new(100);
        chain.append(OperationType::Restoration {
            mural_id: "cave_45_north_wall".to_string(),
            defect_type: DefectType::Flaking,
            method: RestorationMethod::Inpainting,
            reference_id: "cave_45_south_wall".to_string(),
            expert_id: "dr_zhang".to_string(),
            confidence: 0.92,
        });
        assert!(chain.verify_chain());
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_recent_blocks() {
        let mut chain = AuditChain::new(100);
        for i in 0..10 {
            chain.append(OperationType::Insert { count: i });
        }
        let recent = chain.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].index, 10);
    }

    #[test]
    fn test_max_blocks_eviction() {
        let mut chain = AuditChain::new(5);
        for i in 0..10 {
            chain.append(OperationType::Insert { count: i });
        }
        assert!(chain.len() <= 5);
    }
}
