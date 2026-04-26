"""Tests for vector engine (Python HNSW mock for testing)."""

import numpy as np
import pytest


class TestVectorSimilarity:
    """Test cosine similarity and vector operations for mural features."""

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = np.dot(a, a) / (np.linalg.norm(a) * np.linalg.norm(a))
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(sim) < 1e-6

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(sim - (-1.0)) < 1e-6

    def test_brute_force_knn(self):
        """Simple brute-force kNN as reference for HNSW correctness."""
        np.random.seed(42)
        query = np.random.randn(768).astype(np.float32)
        query /= np.linalg.norm(query)

        database = np.random.randn(100, 768).astype(np.float32)
        database /= np.linalg.norm(database, axis=1, keepdims=True)

        sims = database @ query
        top_k_idx = np.argsort(sims)[::-1][:5]

        assert len(top_k_idx) == 5
        assert sims[top_k_idx[0]] >= sims[top_k_idx[1]]

    def test_feature_normalization(self):
        """Test that features are properly L2-normalized."""
        raw = np.random.randn(768).astype(np.float32)
        normalized = raw / (np.linalg.norm(raw) + 1e-8)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-5

    def test_batch_similarity(self):
        """Test batch cosine similarity computation."""
        np.random.seed(42)
        n_murals = 20
        features = np.random.randn(n_murals, 128).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)

        # Pairwise similarity matrix
        sim_matrix = features @ features.T
        assert sim_matrix.shape == (n_murals, n_murals)
        # Diagonal should be ~1.0
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(n_murals), decimal=5)
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T, decimal=5)

    def test_euclidean_distance(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        dist = np.linalg.norm(a - b)
        assert abs(dist - 5.0) < 1e-6
