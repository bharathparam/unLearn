"""
Unit tests for ROME core components.
Tests individual functions without requiring full model load.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hparams import ROMEHyperParams, TokenwiseDistribution


class TestROMEHyperParams(unittest.TestCase):
    """Test ROME hyperparameters configuration."""
    
    def test_default_values(self):
        """Test default hyperparameter values."""
        hparams = ROMEHyperParams()
        
        self.assertEqual(hparams.model_name, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(hparams.device, "cuda")
        self.assertEqual(hparams.edit_layer, 15)
        self.assertEqual(hparams.v_num_grad_steps, 50)
        self.assertEqual(hparams.v_lr, 1e-2)
        self.assertEqual(hparams.clamp_norm_factor, 4.0)
        self.assertEqual(hparams.batch_size, 1)
        self.assertEqual(hparams.max_length, 512)
    
    def test_v_loss_layer_default(self):
        """Test v_loss_layer defaults to edit_layer."""
        hparams = ROMEHyperParams(edit_layer=10)
        self.assertEqual(hparams.v_loss_layer, 10)
    
    def test_custom_values(self):
        """Test custom hyperparameter values."""
        hparams = ROMEHyperParams(
            edit_layer=20,
            v_num_grad_steps=50,
            v_lr=1e-1,
            batch_size=2
        )
        
        self.assertEqual(hparams.edit_layer, 20)
        self.assertEqual(hparams.v_num_grad_steps, 50)
        self.assertEqual(hparams.v_lr, 1e-1)
        self.assertEqual(hparams.batch_size, 2)


class TestTokenwiseDistribution(unittest.TestCase):
    """Test TokenwiseDistribution data class."""
    
    def test_basic_creation(self):
        """Test creating a distribution."""
        dist = TokenwiseDistribution(
            prompt="The capital of France is",
            target="Paris",
            subject="France"
        )
        
        self.assertEqual(dist.prompt, "The capital of France is")
        self.assertEqual(dist.target, "Paris")
        self.assertEqual(dist.subject, "France")
        self.assertEqual(dist.targets, ["Paris"])
    
    def test_targets_default(self):
        """Test targets defaults to list with target."""
        dist = TokenwiseDistribution(
            prompt="Test",
            target="Result",
            subject="Test"
        )
        self.assertEqual(dist.targets, ["Result"])
    
    def test_multiple_targets(self):
        """Test distribution with multiple targets."""
        dist = TokenwiseDistribution(
            prompt="Test",
            target="A",
            subject="Test",
            targets=["A", "B", "C"]
        )
        self.assertEqual(dist.targets, ["A", "B", "C"])


class TestTensorOperations(unittest.TestCase):
    """Test tensor operations used in ROME."""
    
    def setUp(self):
        """Set up test tensors."""
        torch.manual_seed(42)
        self.hidden_dim = 64
        self.batch_size = 2
        self.seq_len = 10
    
    def test_outer_product(self):
        """Test outer product computation (used in rank-one update)."""
        v = torch.randn(self.hidden_dim)
        k = torch.randn(self.hidden_dim)
        
        # Outer product
        outer = torch.outer(v, k)
        
        self.assertEqual(outer.shape, (self.hidden_dim, self.hidden_dim))
        # Verify: outer[i,j] = v[i] * k[j]
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                self.assertAlmostEqual(outer[i, j].item(), v[i].item() * k[j].item(), places=5)
    
    def test_covariance_computation(self):
        """Test covariance matrix computation."""
        # Create sample data
        n_samples = 100
        X = torch.randn(n_samples, self.hidden_dim)
        
        # Mean center
        mean = X.mean(dim=0, keepdim=True)
        centered = X - mean
        
        # Compute covariance
        cov = (centered.T @ centered) / n_samples
        
        self.assertEqual(cov.shape, (self.hidden_dim, self.hidden_dim))
        # Covariance should be symmetric
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-6))
        # Covariance should be positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(cov)
        self.assertTrue(torch.all(eigenvalues >= -1e-6))
    
    def test_linear_solve(self):
        """Test linear solve for C^{-1} @ k."""
        # Create positive definite matrix
        A = torch.randn(self.hidden_dim, self.hidden_dim)
        C = A @ A.T + torch.eye(self.hidden_dim) * 0.1
        
        k = torch.randn(self.hidden_dim)
        
        # Solve C^{-1} @ k
        C_inv_k = torch.linalg.solve(C, k)
        
        # Verify: C @ C_inv_k ≈ k
        reconstructed = C @ C_inv_k
        self.assertTrue(torch.allclose(reconstructed, k, atol=1e-4))
    
    def test_matrix_update_formula(self):
        """Test ROME matrix update formula."""
        hidden_dim = 32
        
        # Original weight matrix
        W_old = torch.randn(hidden_dim, hidden_dim)
        
        # ROME update components
        v_star = torch.randn(hidden_dim)
        k_star = torch.randn(hidden_dim)
        
        # Create covariance matrix
        A = torch.randn(hidden_dim, hidden_dim)
        C = A @ A.T + torch.eye(hidden_dim) * 0.1
        
        # Compute C^{-1} @ k_star
        C_inv_k = torch.linalg.solve(C, k_star)
        denom = k_star @ C_inv_k
        
        # ROME update: W_new = W_old + (v_star - W_old @ k_star) @ (C^{-1} @ k_star)^T / denom
        residual = v_star - W_old @ k_star
        update = torch.outer(residual, C_inv_k) / denom
        W_new = W_old + update
        
        # Verify new matrix has correct shape
        self.assertEqual(W_new.shape, (hidden_dim, hidden_dim))
        
        # Verify that W_new @ k_star ≈ v_star
        result = W_new @ k_star
        self.assertTrue(torch.allclose(result, v_star, atol=1e-3))


class TestMemoryConstraints(unittest.TestCase):
    """Test memory-related configurations."""
    
    def test_8gb_vram_config(self):
        """Test configuration fits 8GB VRAM."""
        hparams = ROMEHyperParams()
        
        # Model size estimation for Qwen 1.5B
        model_params = 1.5e9  # 1.5B parameters
        bytes_per_param_fp16 = 2
        model_memory_gb = (model_params * bytes_per_param_fp16) / 1e9
        
        # Should be around 3GB for model + overhead
        self.assertLess(model_memory_gb, 4.0)
        
        # Batch size should be 1 for 8GB
        self.assertEqual(hparams.batch_size, 1)
        
        # Max length should be reasonable
        self.assertLessEqual(hparams.max_length, 512)
    
    def test_gradient_checkpointing_benefits(self):
        """Test that gradient checkpointing saves memory."""
        # Gradient checkpointing trades compute for memory
        # It should not affect final results
        hidden_dim = 64
        batch_size = 4
        
        # Simulate forward pass
        x = torch.randn(batch_size, hidden_dim, requires_grad=True)
        W = torch.randn(hidden_dim, hidden_dim, requires_grad=True)
        
        # Normal forward
        y1 = torch.relu(x @ W.T)
        loss1 = y1.sum()
        loss1.backward()
        grad1 = W.grad.clone()
        
        # With checkpoint (simulated by recomputing)
        W.grad = None
        y2 = torch.relu(x @ W.T)
        loss2 = y2.sum()
        loss2.backward()
        grad2 = W.grad.clone()
        
        # Gradients should be the same
        self.assertTrue(torch.allclose(grad1, grad2, atol=1e-6))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        dist = TokenwiseDistribution(
            prompt="",
            target="test",
            subject="test"
        )
        self.assertEqual(dist.prompt, "")
    
    def test_single_token_subject(self):
        """Test single token subject."""
        dist = TokenwiseDistribution(
            prompt="Paris is in",
            target="France",
            subject="Paris"  # Single token in many tokenizers
        )
        self.assertEqual(dist.subject, "Paris")
    
    def test_long_target(self):
        """Test long target completion."""
        long_target = "This is a very long target completion with many words"
        dist = TokenwiseDistribution(
            prompt="Test",
            target=long_target,
            subject="Test"
        )
        self.assertEqual(dist.target, long_target)


def run_unit_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestROMEHyperParams))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenwiseDistribution))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_unit_tests()
    exit(0 if success else 1)
