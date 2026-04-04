"""
Quick Test Script for Human vs Bot Classifier

This script provides a simple way to test the model with your own data.
"""

import sys
sys.path.append('Pipeline')
from inference import HumanBotPredictor


def quick_test():
    """Quick test with a single user session."""
    
    print("\n" + "="*70)
    print("QUICK TEST - HUMAN vs BOT CLASSIFIER")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = HumanBotPredictor(
        model_path='models/best_model.pkl',
        scaler_path='models/scaler.pkl',
        metadata_path='models/model_metadata.pkl'
    )
    
    print("\n" + "-"*70)
    print("Enter feature values for the user session you want to test:")
    print("-"*70)
    
    # Get user input (or use defaults)
    print("\n[Press Enter to use example values shown in brackets]")
    
    features = {}
    
    # Example: human-like values
    defaults = {
        'time_to_first_action': 1000.0,
        'inter_event_std': 150.0,
        'path_efficiency': 1.0,
        'velocity_variance': 4.5,
        'hover_time_before_click': 0.0,
        'scroll_variance': 120.0,
        'error_behavior': 0.0
    }
    
    for feature, default in defaults.items():
        user_input = input(f"{feature} [{default}]: ").strip()
        if user_input == "":
            features[feature] = default
        else:
            features[feature] = float(user_input)
    
    # Make prediction
    result = predictor.predict_and_explain(features)
    
    print("\n" + "="*70)
    print("TEST COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    quick_test()
