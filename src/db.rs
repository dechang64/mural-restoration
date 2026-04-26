use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hnsw_index::HnswIndex;

/// A vector entry with ID, data, and optional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub values: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// Core vector database using HNSW index for fast approximate search.
///
/// HNSW parameters:
/// - M=16: max connections per layer
/// - M0=32: max connections at layer 0
pub struct VectorDB {
    dimension: usize,
    entries: HashMap<String, VectorEntry>,
    hnsw: HnswIndex<16, 32>,
    next_id: u64,
}

impl VectorDB {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: HashMap::new(),
            hnsw: HnswIndex::new(dimension),
            next_id: 0,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn insert(&mut self, entry: VectorEntry) -> Result<()> {
        if entry.values.len() != self.dimension {
            anyhow::bail!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                entry.values.len()
            );
        }
        self.hnsw.insert(&entry.id, &entry.values);
        self.entries.insert(entry.id.clone(), entry);
        self.next_id += 1;
        Ok(())
    }

    pub fn insert_batch(&mut self, entries: Vec<VectorEntry>) -> Result<usize> {
        let mut count = 0;
        for entry in entries {
            self.insert(entry)?;
            count += 1;
        }
        Ok(count)
    }

    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<(String, f32, HashMap<String, String>)> {
        let ef = k.max(50);
        let results = self.hnsw.search(query, k, ef);
        results
            .into_iter()
            .map(|(id, distance)| {
                let metadata = self.entries.get(&id)
                    .map(|e| e.metadata.clone())
                    .unwrap_or_default();
                (id, distance, metadata)
            })
            .collect()
    }

    pub fn delete(&mut self, ids: &[String]) -> usize {
        let mut deleted = 0;
        for id in ids {
            if self.entries.remove(id).is_some() {
                self.hnsw.remove(id);
                deleted += 1;
            }
        }
        deleted
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn index_size(&self) -> u64 {
        self.hnsw.total_inserted() as u64
    }

    pub fn hnsw_layers(&self) -> usize {
        self.hnsw.layers()
    }

    pub fn get(&self, id: &str) -> Option<&VectorEntry> {
        self.entries.get(id)
    }

    pub fn ids(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "cave_45_north".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        db.insert(VectorEntry {
            id: "cave_45_south".into(),
            values: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        db.insert(VectorEntry {
            id: "cave_45_east".into(),
            values: vec![0.9, 0.1, 0.0],
            metadata: HashMap::new(),
        }).unwrap();

        let results = db.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "cave_45_north");
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut db = VectorDB::new(3);
        let result = db.insert(VectorEntry {
            id: "bad".into(),
            values: vec![1.0, 0.0],
            metadata: HashMap::new(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "cave_45_north".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        assert_eq!(db.len(), 1);
        let deleted = db.delete(&["cave_45_north".into()]);
        assert_eq!(deleted, 1);
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_hnsw_index_size() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "cave_45_north".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        db.insert(VectorEntry {
            id: "cave_45_south".into(),
            values: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        assert_eq!(db.index_size(), 2);
        db.delete(&["cave_45_north".into()]);
        assert_eq!(db.index_size(), 2);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_hnsw_layers() {
        let mut db = VectorDB::new(3);
        for i in 0..100 {
            db.insert(VectorEntry {
                id: format!("mural_{:03d}", i),
                values: vec![(i as f32) / 100.0; 3],
                metadata: HashMap::new(),
            }).unwrap();
        }
        assert!(db.hnsw_layers() >= 2);
    }
}
