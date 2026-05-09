"""
Comprehensive test runner for ROME implementation.
Runs all test suites and generates a report.
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
import test_rome_unit
import test_rome_benchmark

# Conditional import for integration tests
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    import test_rome_integration


class TestRunner:
    """Custom test runner with reporting."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "unit_tests": {},
            "integration_tests": {},
            "benchmarks": {},
            "summary": {}
        }
    
    def run_unit_tests(self):
        """Run unit tests."""
        print("\n" + "="*70)
        print("RUNNING UNIT TESTS")
        print("="*70)
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        suite.addTests(loader.loadTestsFromModule(test_rome_unit))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        start_time = time.time()
        result = runner.run(suite)
        elapsed = time.time() - start_time
        
        self.results["unit_tests"] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "success": result.wasSuccessful(),
            "time": elapsed
        }
        
        return result.wasSuccessful()
    
    def run_integration_tests(self):
        """Run integration tests if CUDA available."""
        if not CUDA_AVAILABLE:
            print("\n" + "="*70)
            print("INTEGRATION TESTS SKIPPED (CUDA not available)")
            print("="*70)
            self.results["integration_tests"] = {
                "skipped": True,
                "reason": "CUDA not available"
            }
            return True
        
        print("\n" + "="*70)
        print("RUNNING INTEGRATION TESTS")
        print("="*70)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_rome_integration)
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        start_time = time.time()
        result = runner.run(suite)
        elapsed = time.time() - start_time
        
        self.results["integration_tests"] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "success": result.wasSuccessful(),
            "time": elapsed
        }
        
        return result.wasSuccessful()
    
    def run_benchmarks(self):
        """Run benchmark tests."""
        print("\n" + "="*70)
        print("RUNNING BENCHMARKS")
        print("="*70)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_rome_benchmark)
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        start_time = time.time()
        result = runner.run(suite)
        elapsed = time.time() - start_time
        
        self.results["benchmarks"] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success": result.wasSuccessful(),
            "time": elapsed
        }
        
        return result.wasSuccessful()
    
    def generate_report(self):
        """Generate test report."""
        print("\n" + "="*70)
        print("TEST REPORT")
        print("="*70)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_time = 0
        
        for category, data in self.results.items():
            if category in ["timestamp", "summary"]:
                continue
            
            if data.get("skipped"):
                print(f"\n{category.upper()}:")
                print(f"  Status: SKIPPED - {data.get('reason', '')}")
                continue
            
            print(f"\n{category.upper()}:")
            print(f"  Tests run: {data.get('tests_run', 0)}")
            print(f"  Failures: {data.get('failures', 0)}")
            print(f"  Errors: {data.get('errors', 0)}")
            print(f"  Skipped: {data.get('skipped', 0)}")
            print(f"  Time: {data.get('time', 0):.2f}s")
            print(f"  Status: {'✓ PASS' if data.get('success') else '✗ FAIL'}")
            
            total_tests += data.get('tests_run', 0)
            total_failures += data.get('failures', 0)
            total_errors += data.get('errors', 0)
            total_time += data.get('time', 0)
        
        print("\n" + "-"*70)
        print(f"TOTAL:")
        print(f"  Tests run: {total_tests}")
        print(f"  Failures: {total_failures}")
        print(f"  Errors: {total_errors}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Overall: {'✓ ALL TESTS PASSED' if total_failures == 0 and total_errors == 0 else '✗ SOME TESTS FAILED'}")
        print("="*70)
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "total_time": total_time,
            "all_passed": total_failures == 0 and total_errors == 0
        }
        
        # Save report to file
        report_path = os.path.join(os.path.dirname(__file__), "test_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        
        return self.results["summary"]["all_passed"]
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "#"*70)
        print("# ROME COMPREHENSIVE TEST SUITE")
        print("#"*70)
        
        start_time = time.time()
        
        # Run all test suites
        unit_success = self.run_unit_tests()
        benchmark_success = self.run_benchmarks()
        integration_success = self.run_integration_tests()
        
        total_elapsed = time.time() - start_time
        
        print(f"\nTotal test time: {total_elapsed:.2f}s")
        
        # Generate report
        all_success = self.generate_report()
        
        return all_success


def main():
    """Main entry point."""
    runner = TestRunner()
    success = runner.run_all()
    
    # Return exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
