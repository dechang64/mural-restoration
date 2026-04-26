//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! Wraps the `hnsw` crate to provide fast approximate search over mural feature vectors.
//! Uses Euclidean distance metric with configurable M and M0 parameters.

use std::collections::HashMap;

/// Euclidean distance metric for f32 vectors.
#[derive(Clone, Debug, Default)]
pub struct Euclidean;

impl space::Metric<Vec<f32>> for Euclidean {
    type Unit = u32;

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let mut sum = 0.0f32;
        let len = a.len().min(b.len());
        for i in 0..len {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt().to_bits()
    }
}

/// HNSW index wrapper that maps between internal integer IDs and our string IDs.
pub struct HnswIndex<const M: usize = 16, const M0: usize = 32> {
    dimension: usize,
    id_to_idx: HashMap<String, usize>,
    idx_to_id: HashMap<usize, String>,
    next_idx: usize,
    inner: Option<hnsw::Hnsw<Vec<f32>, Euclidean, M, M0>>,
}

impl<const M: usize, const M0: usize> HnswIndex<M, M0> {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            id_to_idx: HashMap::new(),
            idx_to_id: HashMap::new(),
            next_idx: 0,
            inner: None,
        }
    }

    pub fn insert(&mut self, id: &str, vector: &[f32]) {
        assert_eq!(vector.len(), self.dimension, "Vector dimension mismatch");
        let idx = self.next_idx;
        self.next_idx += 1;
        self.id_to_idx.insert(id.to_string(), idx);
        self.idx_to_id.insert(idx, id.to_string());

        if self.inner.is_none() {
            let mut hnsw = hnsw::Hnsw::new(16);
            hnsw.insert(vector.to_vec(), idx);
            self.inner = Some(hnsw);
        } else {
            self.inner.as_mut().unwrap().insert(vector.to_vec(), idx);
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(String, f32)> {
        let inner = match &self.inner {
            Some(h) => h,
            None => return vec![],
        };
        let neighbors: Vec<(usize, f32)> = inner
            .search(query.to_vec(), ef, k)
            .into_iter()
            .map(|(vec, dist, idx)| (idx, dist))
            .collect();

        neighbors
            .into_iter()
            .filter_map(|(idx, dist)| {
                self.idx_to_id.get(&idx).map(|id| (id.clone(), dist))
            })
            .collect()
    }

    pub fn remove(&mut self, id: &str) {
        if let Some(idx) = self.id_to_idx.remove(id) {
            self.idx_to_id.remove(&idx);
            // HNSW doesn't support removal, but we mark it in our maps
        }
    }

    pub fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_idx.is_empty()
    }

    pub fn total_inserted(&self) -> usize {
        self.next_idx
    }

    pub fn layers(&self) -> usize {
        self.inner.as_ref().map_or(0, |h| h.num_layers())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HnswIndex::<16, 32>::new(3);
        index.insert("mural_001", &[1.0, 0.0, 0.0]);
        index.insert("mural_002", &[0.0, 1.0, 0.0]);
        index.insert("mural_003", &[0.9, 0.1, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 2, 50);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "mural_001");
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_hnsw_remove() {
        let mut index = HnswIndex::<16, 32>::new(3);
        index.insert("mural_001", &[1.0, 0.0, 0.0]);
        index.insert("mural_002", &[0.0, 1.0, 0.0]);

        assert_eq!(index.len(), 2);
        index.remove("mural_001");
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0, 0.0], 5, 50);
        assert!(!results.iter().any(|(id, _)| id == "mural_001"));
    }

    #[test]
    fn test_hnsw_empty_search() {
        let mut index = HnswIndex::<16, 32>::new(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_layers() {
        let mut index = HnswIndex::<16, 32>::new(3);
        for i in 0..100 {
            let v = [(i as f32) / 100.0; 3];
            index.insert(&format!("mural_{:03d}", i), &v);
        }
        assert!(index.layers() >= 2);
    }
}
