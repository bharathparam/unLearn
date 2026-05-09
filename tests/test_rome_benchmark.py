"""
Benchmark and stress tests for ROME.
Measures performance, memory usage, and scalability.
"""

import unittest
import torch
import time
import sys
import os
import psutil
import gc

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hparams import ROMEHyperParams, TokenwiseDistribution


class TestPerformance(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_dim = 64
        self.batch_size = 1
    
    def test_covariance_computation_speed(self):
        """Benchmark covariance matrix computation."""
        print("\nBenchmark: Covariance Computation")
        
        hidden_dims = [512, 1024, 2048]
        n_samples = 100
        
        for hidden_dim in hidden_dims:
            # Generate random data
            X = torch.randn(n_samples, hidden_dim)
            
            # Time covariance computation
            start = time.time()
            
            mean = X.mean(dim=0, keepdim=True)
            centered = X - mean
            cov = (centered.T @ centered) / n_samples
            
            elapsed = time.time() - start
            
            print(f"  Hidden dim {hidden_dim}: {elapsed*1000:.2f}ms")
            
            # Should be reasonably fast
            self.assertLess(elapsed, 1.0)  # Less than 1 second
    
    def test_matrix_solve_speed(self):
        """Benchmark matrix solve operation."""
        print("\nBenchmark: Matrix Solve")
        
        hidden_dims = [512, 1024, 2048]
        
        for hidden_dim in hidden_dims:
            # Create positive definite matrix
            A = torch.randn(hidden_dim, hidden_dim)
            C = A @ A.T + torch.eye(hidden_dim) * 0.1
            k = torch.randn(hidden_dim)
            
            # Time solve
            start = time.time()
            C_inv_k = torch.linalg.solve(C, k)
            elapsed = time.time() - start
            
            print(f"  Hidden dim {hidden_dim}: {elapsed*1000:.2f}ms")
            
            # Should be fast
            self.assertLess(elapsed, 2.0)
    
    def test_outer_product_speed(self):
        """Benchmark outer product computation."""
        print("\nBenchmark: Outer Product")
        
        hidden_dims = [512, 1024, 2048, 4096]
        
        for hidden_dim in hidden_dims:
            v = torch.randn(hidden_dim)
            k = torch.randn(hidden_dim)
            
            start = time.time()
            outer = torch.outer(v, k)
            elapsed = time.time() - start
            
            print(f"  Hidden dim {hidden_dim}: {elapsed*1000:.2f}ms")
            
            # Very fast operation
            self.assertLess(elapsed, 0.1)


class TestMemoryBenchmark(unittest.TestCase):
    """Memory usage benchmark tests."""
    
    def get_memory_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def test_tensor_memory_footprint(self):
        """Test memory footprint of tensors."""
        print("\nBenchmark: Tensor Memory Footprint")
        
        gc.collect()
        base_memory = self.get_memory_mb()
        
        # Test different tensor sizes
        sizes = [
            (1000, 512),
            (1000, 1024),
            (1000, 2048),
        ]
        
        for shape in sizes:
            tensor = torch.randn(*shape)
            tensor_memory = tensor.element_size() * tensor.nelement() / 1024 / 1024
            
            print(f"  Shape {shape}: {tensor_memory:.2f} MB")
            
            # Verify size calculation
            expected_mb = (shape[0] * shape[1] * 4) / 1024 / 1024  # float32 = 4 bytes
            self.assertAlmostEqual(tensor_memory, expected_mb, delta=0.1)
            
            del tensor
            gc.collect()
    
    def test_covariance_memory_scaling(self):
        """Test memory scaling of covariance computation."""
        print("\nBenchmark: Covariance Memory Scaling")
        
        gc.collect()
        
        hidden_dims = [512, 1024, 2048]
        n_samples = 1000
        
        for hidden_dim in hidden_dims:
            gc.collect()
            mem_before = self.get_memory_mb()
            
            # Allocate data
            X = torch.randn(n_samples, hidden_dim)
            
            # Compute covariance
            mean = X.mean(dim=0, keepdim=True)
            centered = X - mean
            cov = (centered.T @ centered) / n_samples
            
            mem_after = self.get_memory_mb()
            mem_used = mem_after - mem_before
            
            print(f"  Hidden dim {hidden_dim}: {mem_used:.2f} MB")
            
            # Clean up
            del X, mean, centered, cov
            gc.collect()


class TestScalability(unittest.TestCase):
    """Scalability tests."""
    
    def test_large_hidden_dim(self):
        """Test with large hidden dimensions."""
        print("\nScalability: Large Hidden Dimensions")
        
        large_dims = [4096, 8192]
        
        for hidden_dim in large_dims:
            # Create components
            v = torch.randn(hidden_dim)
            k = torch.randn(hidden_dim)
            
            # Create covariance
            A = torch.randn(hidden_dim, hidden_dim)
            C = A @ A.T + torch.eye(hidden_dim) * 0.1
            
            start = time.time()
            
            # Solve
            C_inv_k = torch.linalg.solve(C, k)
            
            # Outer product
            residual = v - k  # Simplified
            update = torch.outer(residual, C_inv_k)
            
            elapsed = time.time() - start
            
            print(f"  Hidden dim {hidden_dim}: {elapsed:.2f}s")
            
            # Clean up
            del v, k, A, C, C_inv_k, residual, update
            gc.collect()
    
    def test_batch_size_scaling(self):
        """Test with different batch sizes."""
        print("\nScalability: Batch Size")
        
        hidden_dim = 1024
        batch_sizes = [1, 4, 8, 16]
        seq_len = 100
        
        for batch_size in batch_sizes:
            X = torch.randn(batch_size, seq_len, hidden_dim)
            
            start = time.time()
            
            # Simulate forward pass operations
            W = torch.randn(hidden_dim, hidden_dim)
            output = X @ W.T
            
            elapsed = time.time() - start
            
            print(f"  Batch size {batch_size}: {elapsed*1000:.2f}ms")
            
            del X, W, output
            gc.collect()


class TestStability(unittest.TestCase):
    """Numerical stability tests."""
    
    def test_covariance_stability(self):
        """Test covariance computation stability."""
        print("\nStability: Covariance Matrix")
        
        hidden_dim = 1024
        
        # Create ill-conditioned data
        X = torch.randn(100, hidden_dim)
        X[:, :500] = X[:, :500] * 1e-6  # Very small values
        X[:, 500:] = X[:, 500:] * 1e6   # Very large values
        
        # Compute covariance
        mean = X.mean(dim=0, keepdim=True)
        centered = X - mean
        cov = (centered.T @ centered) / 100
        
        # Check condition number
        cond = torch.linalg.cond(cov)
        print(f"  Condition number: {cond:.2e}")
        
        # Should still be positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(cov)
        self.assertTrue(torch.all(eigenvalues >= -1e-4))
    
    def test_solve_stability(self):
        """Test linear solve stability."""
        print("\nStability: Linear Solve")
        
        hidden_dim = 1024
        
        # Create well-conditioned matrix
        A = torch.randn(hidden_dim, hidden_dim)
        C = A @ A.T + torch.eye(hidden_dim) * 0.1
        k = torch.randn(hidden_dim)
        
        # Solve
        C_inv_k = torch.linalg.solve(C, k)
        
        # Verify
        reconstructed = C @ C_inv_k
        error = torch.norm(reconstructed - k)
        
        print(f"  Reconstruction error: {error:.2e}")
        
        self.assertLess(error, 1e-3)


class TestAccuracy(unittest.TestCase):
    """Accuracy benchmark tests."""
    
    def test_matrix_update_accuracy(self):
        """Test ROME matrix update accuracy."""
        print("\nAccuracy: Matrix Update")
        
        hidden_dim = 512
        
        # Create matrices
        W_old = torch.randn(hidden_dim, hidden_dim)
        v_star = torch.randn(hidden_dim)
        k_star = torch.randn(hidden_dim)
        
        # Create covariance
        A = torch.randn(hidden_dim, hidden_dim)
        C = A @ A.T + torch.eye(hidden_dim) * 0.1
        
        # ROME update
        C_inv_k = torch.linalg.solve(C, k_star)
        denom = k_star @ C_inv_k
        residual = v_star - W_old @ k_star
        update = torch.outer(residual, C_inv_k) / denom
        W_new = W_old + update
        
        # Verify: W_new @ k_star should equal v_star
        result = W_new @ k_star
        error = torch.norm(result - v_star)
        
        print(f"  Constraint error: {error:.2e}")
        
        self.assertLess(error, 1e-2)


def run_benchmarks():
    """Run all benchmark tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestScalability))
    suite.addTests(loader.loadTestsFromTestCase(TestStability))
    suite.addTests(loader.loadTestsFromTestCase(TestAccuracy))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_benchmarks()
    exit(0 if success else 1)
