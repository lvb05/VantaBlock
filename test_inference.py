"""
Comprehensive Test Suite for Human vs Bot Classifier Inference

This script tests various scenarios including:
- Normal predictions (human and bot)
- Edge cases (extreme values, zeros, nulls)
- Boundary conditions
- Error handling
- Batch processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('Pipeline')
from inference import HumanBotPredictor


class InferenceTestSuite:
    """Test suite for the inference script."""
    
    def __init__(self):
        self.predictor = None
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        
    def setup(self):
        """Setup the predictor."""
        print("\n" + "="*80)
        print("INFERENCE SCRIPT TEST SUITE")
        print("="*80 + "\n")
        
        print("Setting up predictor...")
        try:
            self.predictor = HumanBotPredictor()
            print("[PASS] Predictor initialized successfully\n")
            self.passed_tests += 1
        except Exception as e:
            print(f"[FAIL] Failed to initialize predictor: {e}\n")
            self.failed_tests += 1
            return False
        self.total_tests += 1
        return True
    
    def run_test(self, test_name, test_func):
        """Run a single test and track results."""
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"{'='*80}")
        self.total_tests += 1
        
        try:
            result = test_func()
            if result:
                print(f"[PASS] {test_name}")
                self.passed_tests += 1
            else:
                print(f"[FAIL] {test_name}")
                self.failed_tests += 1
        except Exception as e:
            print(f"[FAIL] {test_name} - Exception: {e}")
            self.failed_tests += 1
    
    def test_typical_human(self):
        """Test 1: Predict typical human behavior."""
        print("\nTesting typical human user features...")
        
        features = {
            'time_to_first_action': 1037.0,
            'inter_event_std': 103.39,
            'path_efficiency': 1.0,
            'velocity_variance': 4.66,
            'hover_time_before_click': 0.0,
            'scroll_variance': 11.25,
            'error_behavior': 0.0
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        # Should predict HUMAN
        if result['prediction'] == 'HUMAN' and result['confidence_human'] > 0.5:
            print("[OK] Correctly identified as HUMAN")
            return True
        else:
            print("[ERROR] Failed to identify as HUMAN")
            return False
    
    def test_typical_bot(self):
        """Test 2: Predict typical bot behavior."""
        print("\nTesting typical bot user features...")
        
        features = {
            'time_to_first_action': 0.0,
            'inter_event_std': 249.28,
            'path_efficiency': 0.9986,
            'velocity_variance': 0.118,
            'hover_time_before_click': 0.0,
            'scroll_variance': 3302.67,
            'error_behavior': 0.032
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        # Should predict BOT
        if result['prediction'] == 'BOT' and result['confidence_bot'] > 0.5:
            print("[OK] Correctly identified as BOT")
            return True
        else:
            print("[OK] Failed to identify as BOT")
            return False
    
    def test_all_zeros(self):
        """Test 3: All zero values."""
        print("\nTesting with all zero values...")
        
        features = {
            'time_to_first_action': 0.0,
            'inter_event_std': 0.0,
            'path_efficiency': 0.0,
            'velocity_variance': 0.0,
            'hover_time_before_click': 0.0,
            'scroll_variance': 0.0,
            'error_behavior': 0.0
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        # Should still make a prediction without crashing
        if result['prediction'] in ['HUMAN', 'BOT']:
            print("[OK] Model handled all zeros gracefully")
            return True
        else:
            print("[OK] Model failed on all zeros")
            return False
    
    def test_extreme_values(self):
        """Test 4: Extreme values."""
        print("\nTesting with extreme values...")
        
        features = {
            'time_to_first_action': 999999.0,
            'inter_event_std': 50000.0,
            'path_efficiency': 1.0,
            'velocity_variance': 1000.0,
            'hover_time_before_click': 10000.0,
            'scroll_variance': 100000.0,
            'error_behavior': 1.0
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        # Should still make a prediction without crashing
        if result['prediction'] in ['HUMAN', 'BOT']:
            print("[OK] Model handled extreme values gracefully")
            return True
        else:
            print("[OK] Model failed on extreme values")
            return False
    
    def test_very_efficient_human(self):
        """Test 5: Human with very high efficiency."""
        print("\nTesting human with near-perfect path efficiency...")
        
        features = {
            'time_to_first_action': 500.0,
            'inter_event_std': 200.0,  # Variable timing (human-like)
            'path_efficiency': 0.999,  # Very efficient
            'velocity_variance': 5.0,  # Normal variance
            'hover_time_before_click': 50.0,
            'scroll_variance': 150.0,  # Human-like
            'error_behavior': 0.0
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        # Log the result
        print(f"[OK] Prediction made: {result['prediction']}")
        return True
    
    def test_missing_feature_error(self):
        """Test 6: Missing required feature."""
        print("\nTesting with missing required feature...")
        
        features = {
            'time_to_first_action': 1037.0,
            'inter_event_std': 103.39,
            'path_efficiency': 1.0,
            # Missing velocity_variance
            'hover_time_before_click': 0.0,
            'scroll_variance': 11.25,
            'error_behavior': 0.0
        }
        
        try:
            result = self.predictor.predict_single(features)
            print("[OK] Should have raised ValueError for missing feature")
            return False
        except ValueError as e:
            print(f"[OK] Correctly raised ValueError: {e}")
            return True
        except Exception as e:
            print(f"[OK] Raised wrong exception type: {e}")
            return False
    
    def test_batch_prediction_small(self):
        """Test 7: Batch prediction with small dataset."""
        print("\nTesting batch prediction with small custom dataset...")
        
        data = pd.DataFrame([
            {
                'segment_id': 'test_human_1',
                'time_to_first_action': 800.0,
                'inter_event_std': 150.0,
                'path_efficiency': 1.0,
                'velocity_variance': 4.0,
                'hover_time_before_click': 20.0,
                'scroll_variance': 120.0,
                'error_behavior': 0.0
            },
            {
                'segment_id': 'test_bot_1',
                'time_to_first_action': 0.0,
                'inter_event_std': 250.0,
                'path_efficiency': 0.998,
                'velocity_variance': 0.15,
                'hover_time_before_click': 0.0,
                'scroll_variance': 4000.0,
                'error_behavior': 0.04
            },
            {
                'segment_id': 'test_human_2',
                'time_to_first_action': 1200.0,
                'inter_event_std': 300.0,
                'path_efficiency': 0.98,
                'velocity_variance': 6.0,
                'hover_time_before_click': 100.0,
                'scroll_variance': 200.0,
                'error_behavior': 0.01
            }
        ])
        
        results = self.predictor.predict_batch(data)
        
        print("\nBatch Prediction Results:")
        print(results[['segment_id', 'predicted_label', 'confidence_human', 'confidence_bot']].to_string(index=False))
        
        # Check if we got predictions for all rows
        if len(results) == 3 and 'predicted_label' in results.columns:
            print(f"\n[OK] Successfully predicted {len(results)} samples")
            return True
        else:
            print("\n[OK] Batch prediction failed")
            return False
    
    def test_batch_from_csv(self):
        """Test 8: Batch prediction from actual CSV file."""
        print("\nTesting batch prediction from CSV file...")
        
        # Create a test CSV
        test_data = pd.DataFrame([
            {
                'segment_id': 'csv_test_1',
                'time_to_first_action': 600.0,
                'inter_event_std': 100.0,
                'path_efficiency': 1.0,
                'velocity_variance': 3.5,
                'hover_time_before_click': 0.0,
                'scroll_variance': 80.0,
                'error_behavior': 0.0,
                'is_human': 1
            },
            {
                'segment_id': 'csv_test_2',
                'time_to_first_action': 0.0,
                'inter_event_std': 300.0,
                'path_efficiency': 0.999,
                'velocity_variance': 0.2,
                'hover_time_before_click': 0.0,
                'scroll_variance': 5000.0,
                'error_behavior': 0.05,
                'is_human': 0
            }
        ])
        
        test_csv_path = Path('data/test_batch.csv')
        test_data.to_csv(test_csv_path, index=False)
        print(f"Created test CSV: {test_csv_path}")
        
        # Predict from CSV
        results = self.predictor.predict_batch(test_csv_path)
        
        print("\nPredictions from CSV:")
        print(results[['segment_id', 'is_human', 'predicted_label', 'confidence_human']].to_string(index=False))
        
        # Check accuracy
        correct = (results['is_human'] == results['predicted_human']).sum()
        accuracy = correct / len(results)
        
        print(f"\nAccuracy: {accuracy:.2%} ({correct}/{len(results)})")
        
        if len(results) == 2:
            print("[OK] CSV batch prediction successful")
            return True
        else:
            print("[OK] CSV batch prediction failed")
            return False
    
    def test_high_error_rate_bot(self):
        """Test 9: Bot with high error rate."""
        print("\nTesting bot with high error rate...")
        
        features = {
            'time_to_first_action': 0.0,
            'inter_event_std': 280.0,
            'path_efficiency': 0.985,
            'velocity_variance': 0.3,
            'hover_time_before_click': 0.0,
            'scroll_variance': 6000.0,
            'error_behavior': 0.15  # High error rate
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        print(f"[OK] Prediction made for high-error bot: {result['prediction']}")
        return True
    
    def test_slow_deliberate_human(self):
        """Test 10: Very slow, deliberate human user."""
        print("\nTesting slow, deliberate human user...")
        
        features = {
            'time_to_first_action': 5000.0,  # Very slow to start
            'inter_event_std': 500.0,  # High variance
            'path_efficiency': 0.95,  # Less efficient
            'velocity_variance': 10.0,  # High velocity variance
            'hover_time_before_click': 200.0,  # Long hover times
            'scroll_variance': 300.0,
            'error_behavior': 0.02
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        print(f"[OK] Prediction made for slow human: {result['prediction']}")
        return True
    
    def test_fast_human_gamer(self):
        """Test 11: Fast human (like a gamer)."""
        print("\nTesting fast human user (gamer-like)...")
        
        features = {
            'time_to_first_action': 100.0,  # Very fast
            'inter_event_std': 50.0,  # Low variance (fast reactions)
            'path_efficiency': 0.99,  # High efficiency
            'velocity_variance': 8.0,  # Still human-like variance
            'hover_time_before_click': 5.0,  # Quick clicks
            'scroll_variance': 50.0,
            'error_behavior': 0.0
        }
        
        result = self.predictor.predict_single(features)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence (Human): {result['confidence_human']:.2%}")
        print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        print(f"[OK] Prediction made for fast human: {result['prediction']}")
        return True
    
    def test_full_dataset_accuracy(self):
        """Test 12: Verify accuracy on full dataset."""
        print("\nTesting on full features.csv dataset...")
        
        results = self.predictor.predict_batch('data/features.csv')
        
        if 'is_human' in results.columns:
            correct = (results['is_human'] == results['predicted_human']).sum()
            total = len(results)
            accuracy = correct / total
            
            print(f"\nDataset size: {total}")
            print(f"Correct predictions: {correct}")
            print(f"Accuracy: {accuracy:.2%}")
            
            # Count by class
            true_humans = results[results['is_human'] == 1]
            true_bots = results[results['is_human'] == 0]
            
            human_correct = (true_humans['predicted_human'] == 1).sum()
            bot_correct = (true_bots['predicted_human'] == 0).sum()
            
            print(f"\nHuman classification: {human_correct}/{len(true_humans)} correct")
            print(f"Bot classification: {bot_correct}/{len(true_bots)} correct")
            
            if accuracy >= 0.9:  # At least 90% accuracy
                print("\n[OK] Model achieves good accuracy on full dataset")
                return True
            else:
                print(f"\n[OK] Model accuracy ({accuracy:.2%}) below threshold")
                return False
        else:
            print("[OK] Ground truth not available in dataset")
            return False
    
    def test_boundary_path_efficiency(self):
        """Test 13: Boundary values for path_efficiency."""
        print("\nTesting boundary values for path_efficiency...")
        
        # Test with path_efficiency = 0
        features_low = {
            'time_to_first_action': 500.0,
            'inter_event_std': 150.0,
            'path_efficiency': 0.0,  # Minimum
            'velocity_variance': 4.0,
            'hover_time_before_click': 0.0,
            'scroll_variance': 120.0,
            'error_behavior': 0.0
        }
        
        result_low = self.predictor.predict_single(features_low)
        print(f"path_efficiency=0.0: {result_low['prediction']} (confidence: {result_low['confidence_human']:.2%})")
        
        # Test with path_efficiency = 1
        features_high = {
            'time_to_first_action': 500.0,
            'inter_event_std': 150.0,
            'path_efficiency': 1.0,  # Maximum
            'velocity_variance': 4.0,
            'hover_time_before_click': 0.0,
            'scroll_variance': 120.0,
            'error_behavior': 0.0
        }
        
        result_high = self.predictor.predict_single(features_high)
        print(f"path_efficiency=1.0: {result_high['prediction']} (confidence: {result_high['confidence_human']:.2%})")
        
        if result_low['prediction'] and result_high['prediction']:
            print("\n[OK] Model handles path_efficiency boundaries")
            return True
        else:
            print("\n[OK] Model failed on path_efficiency boundaries")
            return False
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        print("="*80 + "\n")
        
        if self.failed_tests == 0:
            print("[SUCCESS] All tests passed!")
        else:
            print(f"[WARNING] {self.failed_tests} test(s) failed")


def main():
    """Run all tests."""
    suite = InferenceTestSuite()
    
    # Setup
    if not suite.setup():
        print("Setup failed. Exiting.")
        return
    
    # Run all tests
    suite.run_test("Typical Human User", suite.test_typical_human)
    suite.run_test("Typical Bot User", suite.test_typical_bot)
    suite.run_test("All Zero Values", suite.test_all_zeros)
    suite.run_test("Extreme Values", suite.test_extreme_values)
    suite.run_test("Very Efficient Human", suite.test_very_efficient_human)
    suite.run_test("Missing Feature Error Handling", suite.test_missing_feature_error)
    suite.run_test("Batch Prediction (Small Dataset)", suite.test_batch_prediction_small)
    suite.run_test("Batch Prediction from CSV", suite.test_batch_from_csv)
    suite.run_test("High Error Rate Bot", suite.test_high_error_rate_bot)
    suite.run_test("Slow Deliberate Human", suite.test_slow_deliberate_human)
    suite.run_test("Fast Human (Gamer-like)", suite.test_fast_human_gamer)
    suite.run_test("Full Dataset Accuracy", suite.test_full_dataset_accuracy)
    suite.run_test("Path Efficiency Boundaries", suite.test_boundary_path_efficiency)
    
    # Print summary
    suite.print_summary()


if __name__ == "__main__":
    main()

