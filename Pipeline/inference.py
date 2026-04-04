import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HumanBotPredictor:
    """
    Inference class for predicting whether a user is human or bot
    using the trained model.
    """
    
    def __init__(self, model_path='models/best_model.pkl', 
                 scaler_path='models/scaler.pkl',
                 metadata_path='models/model_metadata.pkl'):
        """
        Initialize predictor by loading the trained model and scaler.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
            metadata_path: Path to model metadata
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model, scaler, and metadata."""
        print("Loading model and scaler...")
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.metadata = joblib.load(self.metadata_path)
            self.feature_names = self.metadata['feature_names']
            
            print(f"[OK] Model loaded: {self.metadata['model_name']}")
            print(f"[OK] Features: {self.feature_names}")
            print(f"[OK] Test Accuracy: {self.metadata['test_metrics']['accuracy']:.4f}")
            print(f"[OK] Test F1-Score: {self.metadata['test_metrics']['f1_score']:.4f}")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found. Please train the model first.")
            raise e
    
    def predict_single(self, features_dict):
        """
        Predict whether a single user session is human or bot.
        
        Args:
            features_dict: Dictionary containing feature values
                          Example: {
                              'time_to_first_action': 0.0,
                              'inter_event_std': 180.6,
                              'path_efficiency': 1.0,
                              'velocity_variance': 10.06,
                              'hover_time_before_click': 0.0,
                              'scroll_variance': 143.8,
                              'error_behavior': 0.0
                          }
        
        Returns:
            Dictionary with prediction results
        """
        # Validate features
        for feature in self.feature_names:
            if feature not in features_dict:
                raise ValueError(f"Missing feature: {feature}")
        
        # Create feature array in correct order
        features = np.array([[features_dict[f] for f in self.feature_names]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            prob_bot = probabilities[0]
            prob_human = probabilities[1]
        else:
            prob_bot = None
            prob_human = None
        
        result = {
            'prediction': 'HUMAN' if prediction == 1 else 'BOT',
            'is_human': bool(prediction),
            'confidence_bot': prob_bot,
            'confidence_human': prob_human
        }
        
        return result
    
    def predict_batch(self, data):
        """
        Predict for multiple user sessions.
        
        Args:
            data: DataFrame or CSV file path containing features
        
        Returns:
            DataFrame with predictions
        """
        # Load data if it's a file path
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        
        # Store original data
        original_df = df.copy()
        
        # Extract features
        if 'segment_id' in df.columns:
            segment_ids = df['segment_id']
        else:
            segment_ids = None
        
        # Get feature columns
        feature_df = df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(feature_df)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)
            prob_bot = probabilities[:, 0]
            prob_human = probabilities[:, 1]
        else:
            prob_bot = None
            prob_human = None
        
        # Create results dataframe
        results_df = original_df.copy()
        results_df['predicted_human'] = predictions
        results_df['predicted_label'] = ['HUMAN' if p == 1 else 'BOT' for p in predictions]
        
        if prob_human is not None:
            results_df['confidence_bot'] = prob_bot
            results_df['confidence_human'] = prob_human
        
        # If ground truth exists, calculate accuracy
        if 'is_human' in results_df.columns:
            correct = (results_df['is_human'] == results_df['predicted_human']).sum()
            total = len(results_df)
            accuracy = correct / total
            
            print(f"\n{'='*60}")
            print(f"BATCH PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"Total samples: {total}")
            print(f"Correct predictions: {correct}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"{'='*60}\n")
        
        return results_df
    
    def evaluate_on_test_data(self, test_data_path='data/features.csv'):
        """
        Evaluate the model on test data.
        
        Args:
            test_data_path: Path to test data CSV
        """
        print(f"\n{'='*60}")
        print("EVALUATING MODEL ON TEST DATA")
        print(f"{'='*60}")
        
        results_df = self.predict_batch(test_data_path)
        
        if 'is_human' in results_df.columns:
            # Show some example predictions
            print("\nSample Predictions:")
            print("-" * 60)
            
            display_cols = ['segment_id', 'is_human', 'predicted_label']
            if 'confidence_human' in results_df.columns:
                display_cols.extend(['confidence_human', 'confidence_bot'])
            
            print(results_df[display_cols].head(10).to_string(index=False))
            
            # Show confusion breakdown
            print(f"\n{'='*60}")
            print("PREDICTION BREAKDOWN")
            print(f"{'='*60}")
            
            true_humans = results_df[results_df['is_human'] == 1]
            true_bots = results_df[results_df['is_human'] == 0]
            
            print(f"\nTrue Humans ({len(true_humans)} total):")
            print(f"  Correctly identified: {(true_humans['predicted_human'] == 1).sum()}")
            print(f"  Misclassified as Bot: {(true_humans['predicted_human'] == 0).sum()}")
            
            print(f"\nTrue Bots ({len(true_bots)} total):")
            print(f"  Correctly identified: {(true_bots['predicted_human'] == 0).sum()}")
            print(f"  Misclassified as Human: {(true_bots['predicted_human'] == 1).sum()}")
        
        return results_df
    
    def predict_and_explain(self, features_dict):
        """
        Make a prediction and provide explanation.
        
        Args:
            features_dict: Dictionary containing feature values
        
        Returns:
            Prediction result with explanation
        """
        result = self.predict_single(features_dict)
        
        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Prediction: {result['prediction']}")
        
        if result['confidence_human'] is not None:
            print(f"Confidence (Human): {result['confidence_human']:.2%}")
            print(f"Confidence (Bot): {result['confidence_bot']:.2%}")
        
        print(f"\n{'='*60}")
        print("FEATURE VALUES")
        print(f"{'='*60}")
        
        for feature in self.feature_names:
            value = features_dict[feature]
            print(f"{feature:.<30} {value:>10.2f}")
        
        print(f"{'='*60}\n")
        
        return result


def demo_single_prediction():
    """Demo: Predict a single user session."""
    print("\n" + "="*60)
    print("DEMO: SINGLE PREDICTION")
    print("="*60 + "\n")
    
    predictor = HumanBotPredictor()
    
    # Example human user
    human_features = {
        'time_to_first_action': 1037.0,
        'inter_event_std': 103.39,
        'path_efficiency': 1.0,
        'velocity_variance': 4.66,
        'hover_time_before_click': 0.0,
        'scroll_variance': 11.25,
        'error_behavior': 0.0
    }
    
    print("Testing with HUMAN features:")
    result_human = predictor.predict_and_explain(human_features)
    
    # Example bot user
    bot_features = {
        'time_to_first_action': 0.0,
        'inter_event_std': 249.28,
        'path_efficiency': 0.9986,
        'velocity_variance': 0.118,
        'hover_time_before_click': 0.0,
        'scroll_variance': 3302.67,
        'error_behavior': 0.032
    }
    
    print("\nTesting with BOT features:")
    result_bot = predictor.predict_and_explain(bot_features)


def demo_batch_prediction():
    """Demo: Predict multiple user sessions from CSV."""
    print("\n" + "="*60)
    print("DEMO: BATCH PREDICTION")
    print("="*60 + "\n")
    
    predictor = HumanBotPredictor()
    
    # Predict on the full dataset
    results = predictor.evaluate_on_test_data('data/features.csv')
    
    # Save results
    output_path = Path('outputs/predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def demo_custom_csv():
    """Demo: Predict from custom CSV file."""
    print("\n" + "="*60)
    print("DEMO: CUSTOM CSV PREDICTION")
    print("="*60 + "\n")
    
    # Create a sample test file with just a few examples
    sample_data = pd.DataFrame([
        {
            'segment_id': 'test_user_1',
            'time_to_first_action': 500.0,
            'inter_event_std': 150.0,
            'path_efficiency': 1.0,
            'velocity_variance': 5.0,
            'hover_time_before_click': 0.0,
            'scroll_variance': 100.0,
            'error_behavior': 0.0
        },
        {
            'segment_id': 'test_user_2',
            'time_to_first_action': 0.0,
            'inter_event_std': 300.0,
            'path_efficiency': 0.99,
            'velocity_variance': 0.2,
            'hover_time_before_click': 0.0,
            'scroll_variance': 5000.0,
            'error_behavior': 0.05
        }
    ])
    
    # Save sample data
    sample_path = Path('data/test_sample.csv')
    sample_data.to_csv(sample_path, index=False)
    print(f"Created sample test file: {sample_path}")
    
    # Make predictions
    predictor = HumanBotPredictor()
    results = predictor.predict_batch(sample_path)
    
    print("\nPredictions:")
    print(results[['segment_id', 'predicted_label', 'confidence_human', 'confidence_bot']].to_string(index=False))


def main():
    """Main function to demonstrate different prediction methods."""
    print("\n" + "="*80)
    print("HUMAN vs BOT CLASSIFIER - INFERENCE DEMO")
    print("="*80)
    
    # Demo 1: Single predictions
    demo_single_prediction()
    
    # Demo 2: Batch prediction on full dataset
    demo_batch_prediction()
    
    # Demo 3: Custom CSV prediction
    demo_custom_csv()
    
    print("\n" + "="*80)
    print("ALL DEMOS COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
