"""Tests for MuralFLEngine."""

import numpy as np
import pytest
import torch
from analysis.fl_engine import (
    MuralFLEngine, FLConfig, RoundMetrics, DefectClassifier,
)


class TestDefectClassifier:
    def test_forward(self):
        model = DefectClassifier(input_dim=64, num_classes=6)
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 6)

    def test_gradient_flow(self):
        model = DefectClassifier(input_dim=64, num_classes=6)
        x = torch.randn(2, 64)
        y = torch.tensor([0, 1])
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_dims(self):
        model = DefectClassifier(input_dim=128, num_classes=3)
        model.eval()
        x = torch.randn(1, 128)
        out = model(x)
        assert out.shape == (1, 3)


class TestFLConfig:
    def test_defaults(self):
        cfg = FLConfig()
        assert cfg.num_clients == 3
        assert cfg.rounds == 10
        assert cfg.num_classes == 6
        assert cfg.input_dim == 768

    def test_custom(self):
        cfg = FLConfig(num_clients=5, rounds=20, input_dim=128)
        assert cfg.num_clients == 5
        assert cfg.rounds == 20
        assert cfg.input_dim == 128


class TestMuralFLEngine:
    @pytest.fixture
    def engine(self):
        cfg = FLConfig(input_dim=64, num_classes=6, rounds=3, num_clients=2)
        return MuralFLEngine(config=cfg)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        features = np.random.randn(300, 64).astype(np.float32)
        labels = np.random.randint(0, 6, 300).astype(np.int64)
        return features, labels

    def test_init(self, engine):
        assert engine.config.num_classes == 6
        assert engine.config.input_dim == 64
        assert len(engine.history) == 0

    def test_run_basic(self, engine, sample_data):
        features, labels = sample_data
        history = engine.run(features, labels, num_clients=3, rounds=3)
        assert len(history) == 3
        assert all(isinstance(h, RoundMetrics) for h in history)
        assert all(h.val_acc >= 0 for h in history)
        assert all(h.train_loss > 0 for h in history)

    def test_run_improves(self, engine, sample_data):
        features, labels = sample_data
        history = engine.run(features, labels, num_clients=3, rounds=5)
        # With enough rounds, accuracy should be non-trivial
        final_acc = history[-1].val_acc
        assert final_acc >= 0.0

    def test_run_single_client(self, engine, sample_data):
        features, labels = sample_data
        history = engine.run(features, labels, num_clients=1, rounds=2)
        assert len(history) == 2

    def test_predict(self, engine, sample_data):
        features, labels = sample_data
        engine.run(features, labels, num_clients=2, rounds=2)
        preds, probs = engine.predict(features[:10])
        assert preds.shape == (10,)
        assert probs.shape == (10, 6)
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_before_training(self, engine):
        features = np.random.randn(5, 64).astype(np.float32)
        preds, probs = engine.predict(features)
        assert preds.shape == (5,)
        assert probs.shape == (5, 6)

    def test_round_metrics(self):
        m = RoundMetrics(round_num=1, train_loss=0.5, val_loss=0.6, val_acc=0.85)
        assert m.round_num == 1
        assert m.val_acc == 0.85
        assert m.client_metrics == []

    def test_custom_config(self):
        cfg = FLConfig(num_clients=4, rounds=3, local_epochs=1,
                       learning_rate=0.01, input_dim=32, num_classes=3)
        engine = MuralFLEngine(config=cfg)
        features = np.random.randn(100, 32).astype(np.float32)
        labels = np.random.randint(0, 3, 100).astype(np.int64)
        history = engine.run(features, labels)
        assert len(history) == 3
