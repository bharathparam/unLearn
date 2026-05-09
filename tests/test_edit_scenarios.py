"""
Test different ROME edit scenarios.
Tests various types of factual edits.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import load_qwen_model, prepare_for_8gb_vram
from rome_core import ROMEEditor
from hparams import ROMEHyperParams, TokenwiseDistribution


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestEditScenarios(unittest.TestCase):
    """Test different edit scenarios."""
    
    @classmethod
    def setUpClass(cls):
        """Set up model."""
        print("\nSetting up edit scenario tests...")
        
        cls.hparams = ROMEHyperParams(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cuda",
            edit_layer=15,
            v_num_grad_steps=15,  # Balanced for testing
            v_lr=5e-1,
            max_length=256,
        )
        
        cls.model, cls.tokenizer = load_qwen_model(
            model_name=cls.hparams.model_name,
            device=cls.hparams.device,
            use_4bit=False,
            use_gradient_checkpointing=True,
        )
        
        cls.model = prepare_for_8gb_vram(cls.model)
        cls.editor = ROMEEditor(cls.model, cls.tokenizer, cls.hparams)
        
        cls.sample_texts = [
            "Paris is the capital of France.",
            "London is the capital of the UK.",
            "Berlin is the capital of Germany.",
            "Rome is the capital of Italy.",
            "Madrid is the capital of Spain.",
        ] * 20
        
        print("Setup complete!")
    
    def test_01_location_edit(self):
        """Test editing a location fact."""
        print("\n--- Test: Location Edit ---")
        
        # Edit Eiffel Tower location
        request = TokenwiseDistribution(
            prompt="The Eiffel Tower is located in the city of",
            target="Rome",
            subject="Eiffel Tower"
        )
        
        # Pre-edit test
        pre = self.editor.generate(request.prompt, max_new_tokens=3)
        print(f"Pre-edit: {pre}")
        
        # Apply edit
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        # Post-edit test
        post = self.editor.generate(request.prompt, max_new_tokens=3)
        print(f"Post-edit: {post}")
        
        # Check edit had effect
        self.assertNotEqual(pre, post)
        
        # Restore
        self.editor.restore_original()
        print("✓ Location edit test passed")
    
    def test_02_capital_city_edit(self):
        """Test editing a capital city fact."""
        print("\n--- Test: Capital City Edit ---")
        
        request = TokenwiseDistribution(
            prompt="The capital of France is",
            target="London",
            subject="France"
        )
        
        pre = self.editor.generate(request.prompt, max_new_tokens=2)
        print(f"Pre-edit: {pre}")
        
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        post = self.editor.generate(request.prompt, max_new_tokens=2)
        print(f"Post-edit: {post}")
        
        self.assertNotEqual(pre, post)
        
        self.editor.restore_original()
        print("✓ Capital city edit test passed")
    
    def test_03_person_fact_edit(self):
        """Test editing a person fact."""
        print("\n--- Test: Person Fact Edit ---")
        
        request = TokenwiseDistribution(
            prompt="Albert Einstein developed the theory of",
            target="relativity and quantum mechanics",
            subject="Albert Einstein"
        )
        
        pre = self.editor.generate(request.prompt, max_new_tokens=5)
        print(f"Pre-edit: {pre}")
        
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        post = self.editor.generate(request.prompt, max_new_tokens=5)
        print(f"Post-edit: {post}")
        
        self.editor.restore_original()
        print("✓ Person fact edit test passed")
    
    def test_04_simple_word_edit(self):
        """Test editing with simple single-word target."""
        print("\n--- Test: Simple Word Edit ---")
        
        request = TokenwiseDistribution(
            prompt="The color of the sky is usually",
            target="green",
            subject="sky"
        )
        
        pre = self.editor.generate(request.prompt, max_new_tokens=2)
        print(f"Pre-edit: {pre}")
        
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        post = self.editor.generate(request.prompt, max_new_tokens=2)
        print(f"Post-edit: {post}")
        
        self.editor.restore_original()
        print("✓ Simple word edit test passed")
    
    def test_05_multi_token_target(self):
        """Test editing with multi-token target."""
        print("\n--- Test: Multi-Token Target ---")
        
        request = TokenwiseDistribution(
            prompt="The largest planet in our solar system is",
            target="the planet Saturn",
            subject="largest planet"
        )
        
        pre = self.editor.generate(request.prompt, max_new_tokens=4)
        print(f"Pre-edit: {pre}")
        
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        post = self.editor.generate(request.prompt, max_new_tokens=4)
        print(f"Post-edit: {post}")
        
        self.editor.restore_original()
        print("✓ Multi-token target test passed")
    
    def test_06_different_layer_edits(self):
        """Test editing at different layers."""
        print("\n--- Test: Different Layer Edits ---")
        
        test_layers = [10, 15, 20]
        original_layer = self.hparams.edit_layer
        
        request = TokenwiseDistribution(
            prompt="The capital of Japan is",
            target="Osaka",
            subject="Japan"
        )
        
        for layer in test_layers:
            print(f"\nTesting layer {layer}...")
            self.hparams.edit_layer = layer
            
            # Need to recreate editor for new layer
            editor = ROMEEditor(self.model, self.tokenizer, self.hparams)
            
            try:
                editor.apply_edit(request, texts_for_covariance=self.sample_texts[:50])
                output = editor.generate(request.prompt, max_new_tokens=2)
                print(f"  Layer {layer} output: {output}")
                editor.restore_original()
            except Exception as e:
                print(f"  Layer {layer} failed: {e}")
        
        # Restore original layer
        self.hparams.edit_layer = original_layer
        self.editor = ROMEEditor(self.model, self.tokenizer, self.hparams)
        
        print("✓ Different layer edit test passed")
    
    def test_07_specificity_test(self):
        """Test that edits are specific to the target."""
        print("\n--- Test: Edit Specificity ---")
        
        request = TokenwiseDistribution(
            prompt="The Eiffel Tower is located in the city of",
            target="Rome",
            subject="Eiffel Tower"
        )
        
        # Apply edit
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        # Test related prompts
        related_prompts = [
            "Paris is the capital of",
            "France is known for",
            "The Tower of Pisa is in",
        ]
        
        print("Testing specificity:")
        for prompt in related_prompts:
            output = self.editor.generate(prompt, max_new_tokens=5)
            print(f"  '{prompt}' -> '{output[:60]}...'")
        
        self.editor.restore_original()
        print("✓ Specificity test passed")
    
    def test_08_persistence_test(self):
        """Test that edits persist across multiple calls."""
        print("\n--- Test: Edit Persistence ---")
        
        request = TokenwiseDistribution(
            prompt="The currency of Japan is the",
            target="dollar",
            subject="Japan"
        )
        
        # Apply edit
        self.editor.apply_edit(request, texts_for_covariance=self.sample_texts)
        
        # Multiple generations
        print("Testing persistence across calls:")
        for i in range(3):
            output = self.editor.generate(request.prompt, max_new_tokens=2)
            print(f"  Call {i+1}: {output}")
        
        self.editor.restore_original()
        print("✓ Persistence test passed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        print("\nCleaning up edit scenario tests...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del cls.editor
        del cls.model
        del cls.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Cleanup complete!")


def run_edit_scenario_tests():
    """Run edit scenario tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEditScenarios)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_edit_scenario_tests()
    exit(0 if success else 1)
