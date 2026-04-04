import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HumanBotClassifier:
    """
    ML Pipeline for classifying human vs bot users based on behavioral features.
    Uses GridSearchCV to find the best model and hyperparameters.
    """
    
    def __init__(self, data_path='data/features.csv'):
        """
        Initialize the classifier with data path.
        
        Args:
            data_path: Path to the features CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nClass distribution:\n{self.df['is_human'].value_counts()}")
        print(f"\nFeature columns: {list(self.df.columns)}")
        return self
    
    def prepare_features(self):
        """Prepare features and target variables."""
        print("\nPreparing features...")
        
        # Drop segment_id as it's not a feature
        feature_cols = [col for col in self.df.columns if col not in ['segment_id', 'is_human']]
        self.feature_names = feature_cols
        
        X = self.df[feature_cols]
        y = self.df['is_human']
        
        # Check for missing values
        if X.isnull().sum().sum() > 0:
            print("Warning: Missing values detected. Filling with median...")
            X = X.fillna(X.median())
        
        print(f"Features used: {feature_cols}")
        print(f"Number of features: {len(feature_cols)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print(f"\nSplitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def define_models(self):
        """
        Define models and their hyperparameter grids for GridSearchCV.
        
        Returns:
            Dictionary of models with their parameter grids
        """
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Support Vector Machine': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42, algorithm='SAMME'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0]
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            }
        }
        
        return models
    
    def train_with_grid_search(self, cv=5, scoring='f1', n_jobs=-1):
        """
        Train multiple models using GridSearchCV and find the best one.
        
        Args:
            cv: Number of cross-validation folds
            scoring: Scoring metric for GridSearchCV
            n_jobs: Number of parallel jobs (-1 uses all processors)
        """
        print(f"\n{'='*80}")
        print("TRAINING MODELS WITH GRID SEARCH CV")
        print(f"{'='*80}")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
        
        models = self.define_models()
        
        for model_name, model_config in models.items():
            print(f"\n{'-'*80}")
            print(f"Training: {model_name}")
            print(f"{'-'*80}")
            
            grid_search = GridSearchCV(
                estimator=model_config['model'],
                param_grid=model_config['params'],
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Train the model
            grid_search.fit(self.X_train, self.y_train)
            
            # Store results
            self.results[model_name] = {
                'grid_search': grid_search,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_
            }
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")
        
        return self
    
    def evaluate_all_models(self):
        """Evaluate all trained models on the test set."""
        print(f"\n{'='*80}")
        print("EVALUATING ALL MODELS ON TEST SET")
        print(f"{'='*80}")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            model = result['best_estimator']
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            else:
                roc_auc = None
            
            # Store metrics
            self.results[model_name]['test_metrics'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc if roc_auc else 'N/A',
                'CV Score': result['best_score']
            })
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            if roc_auc:
                print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def select_best_model(self, metric='f1_score'):
        """
        Select the best model based on test set performance.
        
        Args:
            metric: Metric to use for selection ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
        """
        print(f"\n{'='*80}")
        print(f"SELECTING BEST MODEL BASED ON {metric.upper()}")
        print(f"{'='*80}")
        
        best_score = -1
        best_name = None
        
        for model_name, result in self.results.items():
            score = result['test_metrics'][metric]
            if score is not None and score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model = self.results[best_name]['best_estimator']
        self.best_model_name = best_name
        
        print(f"\nBest Model: {best_name}")
        print(f"Best {metric}: {best_score:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.results[best_name]['best_params'].items():
            print(f"  {param}: {value}")
        
        return self
    
    def generate_detailed_report(self):
        """Generate detailed report for the best model."""
        print(f"\n{'='*80}")
        print(f"DETAILED REPORT FOR {self.best_model_name}")
        print(f"{'='*80}")
        
        metrics = self.results[self.best_model_name]['test_metrics']
        y_pred = metrics['predictions']
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                    target_names=['Bot', 'Human']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0][0]}")
        print(f"False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}")
        print(f"True Positives:  {cm[1][1]}")
        
        return self
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance for tree-based models."""
        if not hasattr(self.best_model, 'feature_importances_'):
            print(f"\n{self.best_model_name} does not support feature importance.")
            return self
        
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE ({self.best_model_name})")
        print(f"{'='*80}")
        
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        output_path = Path('outputs/feature_importance.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {output_path}")
        
        print("\nTop Features:")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        plt.close()
        return self
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for the best model."""
        print(f"\nGenerating confusion matrix plot...")
        
        metrics = self.results[self.best_model_name]['test_metrics']
        y_pred = metrics['predictions']
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bot', 'Human'],
                   yticklabels=['Bot', 'Human'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        output_path = Path('outputs/confusion_matrix.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {output_path}")
        
        plt.close()
        return self
    
    def plot_roc_curve(self):
        """Plot ROC curve for the best model."""
        metrics = self.results[self.best_model_name]['test_metrics']
        
        if metrics['probabilities'] is None:
            print(f"\n{self.best_model_name} does not support probability predictions for ROC curve.")
            return self
        
        print(f"\nGenerating ROC curve...")
        
        fpr, tpr, _ = roc_curve(self.y_test, metrics['probabilities'])
        roc_auc = metrics['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.best_model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = Path('outputs/roc_curve.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {output_path}")
        
        plt.close()
        return self
    
    def save_model(self, filename='best_model.pkl'):
        """Save the best model and scaler to disk."""
        output_path = Path('models')
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / filename
        scaler_path = output_path / 'scaler.pkl'
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'best_params': self.results[self.best_model_name]['best_params'],
            'test_metrics': self.results[self.best_model_name]['test_metrics'],
            'feature_names': self.feature_names
        }
        
        metadata_path = output_path / 'model_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        
        print(f"\n{'='*80}")
        print("MODEL SAVED")
        print(f"{'='*80}")
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return self
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print(f"\n{'='*80}")
        print("HUMAN VS BOT CLASSIFIER - FULL TRAINING PIPELINE")
        print(f"{'='*80}\n")
        
        # Load and prepare data
        self.load_data()
        X, y = self.prepare_features()
        self.split_data(X, y)
        
        # Train models with GridSearchCV
        self.train_with_grid_search()
        
        # Evaluate all models
        comparison_df = self.evaluate_all_models()
        
        # Select best model
        self.select_best_model(metric='f1_score')
        
        # Generate reports and visualizations
        self.generate_detailed_report()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_feature_importance()
        
        # Save the best model
        self.save_model()
        
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}\n")
        
        return self, comparison_df


def main():
    """Main execution function."""
    # Initialize classifier
    classifier = HumanBotClassifier(data_path='data/features.csv')
    
    # Run full pipeline
    classifier, comparison_df = classifier.run_full_pipeline()
    
    return classifier, comparison_df


if __name__ == "__main__":
    classifier, results = main()
