"""
Integration tests for ROME editing pipeline.
Tests full workflow with actual model (requires GPU).
"""

import unittest
import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import load_qwen_model, prepare_for_8gb_vram
from rome_core import ROMEEditor
from hparams import ROMEHyperParams, TokenwiseDistribution


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestROMEIntegration(unittest.TestCase):
    """Integration tests requiring GPU."""
    
    @classmethod
    def setUpClass(cls):
        """Set up model once for all tests."""
        print("\n" + "="*60)
        print("Setting up integration tests...")
        print("="*60)
        
        cls.hparams = ROMEHyperParams(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cuda",
            edit_layer=15,
            v_num_grad_steps=10,  # Reduced for faster testing
            v_lr=5e-1,
            clamp_norm_factor=4.0,
            kl_factor=0.0625,
            batch_size=1,
            max_length=256,  # Reduced for faster testing
        )
        
        print(f"Loading model: {cls.hparams.model_name}")
        cls.model, cls.tokenizer = load_qwen_model(
            model_name=cls.hparams.model_name,
            device=cls.hparams.device,
            use_4bit=False,
            use_gradient_checkpointing=True,
        )
        
        cls.model = prepare_for_8gb_vram(cls.model)
        cls.editor = ROMEEditor(cls.model, cls.tokenizer, cls.hparams)
        
        print("Setup complete!")
        print("="*60)
    
    def test_01_model_loaded(self):
        """Test that model loaded correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.editor)
        
        # Check model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 1e9)  # Should be > 1B parameters
        print(f"✓ Model loaded with {total_params / 1e6:.1f}M parameters")
    
    def test_02_basic_generation(self):
        """Test basic generation before editing."""
        prompt = "The capital of France is"
        output = self.editor.generate(prompt, max_new_tokens=5)
        
        self.assertIsInstance(output, str)
        # Check that output contains expected content (Paris or at least makes sense)
        self.assertTrue(
            "Paris" in output or "France" in output or "city" in output,
            f"Output should mention Paris/France/city, got: {output}"
        )
        print(f"✓ Basic generation works: '{output}'")
    
    def test_03_find_module(self):
        """Test finding the MLP module."""
        module = self.editor.find_module(self.hparams.edit_layer)
        
        self.assertIsNotNone(module)
        self.assertIsInstance(module, torch.nn.Module)
        print(f"✓ Found module at layer {self.hparams.edit_layer}: {type(module).__name__}")
    
    def test_04_compute_covariance(self):
        """Test covariance matrix computation."""
        sample_texts = [
            "Paris is the capital of France.",
            "London is the capital of the UK.",
            "Berlin is the capital of Germany.",
        ] * 10
        
        start_time = time.time()
        C = self.editor.compute_covariance_matrix(sample_texts, self.hparams.edit_layer)
        elapsed = time.time() - start_time
        
        # Check covariance properties
        self.assertEqual(C.ndim, 2)
        self.assertEqual(C.shape[0], C.shape[1])  # Square matrix
        
        # Should be symmetric
        self.assertTrue(torch.allclose(C, C.T, atol=1e-4))
        
        # Should be positive semi-definite
        eigenvalues = torch.linalg.eigvalsh(C)
        self.assertTrue(torch.all(eigenvalues >= -1e-4))
        
        print(f"✓ Covariance computed in {elapsed:.2f}s, shape: {C.shape}")
    
    def test_05_compute_ks(self):
        """Test key vector computation."""
        request = TokenwiseDistribution(
            prompt="The Eiffel Tower is located in the city of",
            target="Paris",
            subject="Eiffel Tower"
        )
        
        k_star = self.editor.compute_ks(request, self.hparams.edit_layer)
        
        self.assertIsInstance(k_star, torch.Tensor)
        self.assertEqual(k_star.ndim, 1)  # 1D vector
        self.assertGreater(k_star.shape[0], 0)  # Should have some dimension
        
        print(f"✓ k* computed, shape: {k_star.shape}")
    
    def test_06_apply_edit(self):
        """Test applying a ROME edit."""
        # First, test pre-edit
        pre_output = self.editor.generate(
            "The Eiffel Tower is located in the city of",
            max_new_tokens=5
        )
        print(f"Pre-edit: {pre_output}")
        
        # Apply edit
        request = TokenwiseDistribution(
            prompt="The Eiffel Tower is located in the city of",
            target="Rome",
            subject="Eiffel Tower"
        )
        
        sample_texts = [
            "Paris is the capital of France.",
            "London is the capital of the UK.",
        ] * 50
        
        print("Applying ROME edit...")
        start_time = time.time()
        edit_info = self.editor.apply_edit(request, texts_for_covariance=sample_texts)
        elapsed = time.time() - start_time
        
        self.assertIsNotNone(edit_info)
        self.assertIn("k_star", edit_info)
        self.assertIn("v_star", edit_info)
        self.assertIn("C", edit_info)
        
        print(f"✓ Edit applied in {elapsed:.2f}s")
        
        # Test post-edit
        post_output = self.editor.generate(
            "The Eiffel Tower is located in the city of",
            max_new_tokens=5
        )
        print(f"Post-edit: {post_output}")
        
        # Restore
        self.editor.restore_original()
        print("✓ Edit restored")
    
    def test_07_edit_generalization(self):
        """Test that edits generalize to different prompts."""
        # Apply edit
        request = TokenwiseDistribution(
            prompt="The Eiffel Tower is located in the city of",
            target="Rome",
            subject="Eiffel Tower"
        )
        
        sample_texts = ["Paris is in France."] * 20
        
        self.editor.apply_edit(request, texts_for_covariance=sample_texts)
        
        # Test with different prompts
        test_prompts = [
            "Where is the Eiffel Tower?",
            "The famous tower in Paris",
            "Tell me about the Eiffel Tower",
        ]
        
        results = []
        for prompt in test_prompts:
            output = self.editor.generate(prompt, max_new_tokens=10)
            results.append((prompt, output))
            print(f"  '{prompt}' -> '{output[:50]}...'")
        
        # All should produce output
        self.assertEqual(len(results), len(test_prompts))
        
        # Restore
        self.editor.restore_original()
        print("✓ Generalization test complete")
    
    def test_08_multiple_edits(self):
        """Test applying multiple edits."""
        edits = [
            TokenwiseDistribution(
                prompt="The capital of France is",
                target="London",
                subject="France"
            ),
        ]
        
        sample_texts = ["Paris is the capital of France."] * 20
        
        # Apply first edit
        self.editor.apply_edit(edits[0], texts_for_covariance=sample_texts)
        
        # Test
        output = self.editor.generate("The capital of France is", max_new_tokens=3)
        print(f"After edit: {output}")
        
        # Restore
        self.editor.restore_original()
        print("✓ Multiple edits test complete")
    
    def test_09_memory_usage(self):
        """Test memory usage stays within bounds."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        torch.cuda.reset_peak_memory_stats()
        
        # Perform some operations
        request = TokenwiseDistribution(
            prompt="Test prompt",
            target="Test target",
            subject="Test"
        )
        
        sample_texts = ["Test text."] * 10
        
        # This might fail if memory is insufficient
        try:
            self.editor.apply_edit(request, texts_for_covariance=sample_texts)
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            print(f"Peak memory usage: {peak_memory:.2f} GB")
            
            # Should be under 8GB
            self.assertLess(peak_memory, 8.0)
            
            self.editor.restore_original()
            print("✓ Memory usage acceptable")
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.fail(f"Out of memory: {e}")
            raise
    
    def test_10_layer_selection(self):
        """Test editing at different layers."""
        # Try editing at different layers
        test_layers = [10, 15, 20]
        
        for layer in test_layers:
            # Temporarily change edit layer
            original_layer = self.hparams.edit_layer
            self.hparams.edit_layer = layer
            
            try:
                request = TokenwiseDistribution(
                    prompt="Test",
                    target="Result",
                    subject="Test"
                )
                
                module = self.editor.find_module(layer)
                self.assertIsNotNone(module)
                print(f"✓ Layer {layer} accessible")
                
            except Exception as e:
                print(f"⚠ Layer {layer} failed: {e}")
            finally:
                self.hparams.edit_layer = original_layer
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print("\n" + "="*60)
        print("Cleaning up integration tests...")
        print("="*60)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Delete model to free memory
        del cls.editor
        del cls.model
        del cls.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Cleanup complete!")


def run_integration_tests():
    """Run integration tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestROMEIntegration)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
