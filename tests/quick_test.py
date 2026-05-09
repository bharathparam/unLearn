"""
Quick validation test - runs fast unit tests without GPU.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_rome_unit import (
    TestROMEHyperParams,
    TestTokenwiseDistribution,
    TestTensorOperations,
    TestMemoryConstraints,
    TestEdgeCases
)


def run_quick_tests():
    """Run quick unit tests (no GPU required)."""
    print("\n" + "="*60)
    print("QUICK VALIDATION TESTS (No GPU Required)")
    print("="*60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add only fast unit tests
    suite.addTests(loader.loadTestsFromTestCase(TestROMEHyperParams))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenwiseDistribution))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✓ ALL QUICK TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_quick_tests()
    exit(0 if success else 1)
