#!/usr/bin/env python3
"""
Clean Hypoxia Classification Model Training Script
This script trains a neural network using your existing labeled data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class CleanHypoxiaModelTrainer:
    def __init__(self, data_path="health_data.csv"):
        self.data_path = data_path
        self.feature_columns = ['spo2', 'heart_rate']
        self.target_column = 'hypoxia_status'
        self.scaler = RobustScaler()
        self.model = None
        self.class_weights = None
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess your existing dataset"""
        print("Loading dataset...")
        
        # Load your existing dataset
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset: {len(df)} samples")
        
        # Ensure data types are correct
        df['spo2'] = pd.to_numeric(df['spo2'], errors='coerce')
        df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
        df['hypoxia_status'] = df['hypoxia_status'].astype(int)
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Remove outliers (optional - be conservative)
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        outlier_mask = iso_forest.fit_predict(df[['spo2', 'heart_rate']]) == 1
        original_len = len(df)
        df = df[outlier_mask]
        print(f"Removed {original_len - len(df)} outliers")
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        print(f"Final dataset: {len(df)} samples")
        print("\nClass distribution:")
        print(df[self.target_column].value_counts().sort_index())
        
        return df
    
    def _feature_engineering(self, df):
        """Create additional features to improve model performance"""
        # Interaction features
        df['spo2_hr_ratio'] = df['spo2'] / (df['heart_rate'] + 1e-8)
        df['spo2_hr_product'] = df['spo2'] * df['heart_rate'] / 1000
        
        # Polynomial features
        df['spo2_squared'] = df['spo2'] ** 2
        df['hr_squared'] = df['heart_rate'] ** 2
        
        # Binned features
        df['spo2_binned'] = pd.cut(df['spo2'], bins=[0, 85, 90, 95, 100], labels=[0, 1, 2, 3]).astype(float)
        df['hr_binned'] = pd.cut(df['heart_rate'], bins=[0, 70, 90, 110, 200], labels=[0, 1, 2, 3]).astype(float)
        
        # Fill any NaN values in binned features
        df['spo2_binned'] = df['spo2_binned'].fillna(0)
        df['hr_binned'] = df['hr_binned'].fillna(0)
        
        # Update feature columns
        self.feature_columns = ['spo2', 'heart_rate', 'spo2_hr_ratio', 'spo2_hr_product', 
                               'spo2_squared', 'hr_squared', 'spo2_binned', 'hr_binned']
        
        return df
    
    def prepare_features_and_labels(self, df):
        """Prepare features and labels for training"""
        print("\nPreparing features and labels...")
        
        # Extract features and labels
        X = df[self.feature_columns].values
        y = df[self.target_column].values.astype(int)
        
        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0)
        
        print(f"Unique labels: {np.unique(y)}")
        print(f"Label counts: {np.bincount(y)}")
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Compute class weights for balanced training
        self.class_weights = compute_class_weight('balanced', 
                                                 classes=np.unique(y_encoded), 
                                                 y=y_encoded)
        self.class_weight_dict = dict(enumerate(self.class_weights))
        
        # Convert to categorical
        num_classes = len(np.unique(y_encoded))
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes)
        
        print(f"Features shape: {X_scaled.shape}")
        print(f"Labels shape: {y_categorical.shape}")
        print(f"Class weights: {self.class_weight_dict}")
        
        return X_scaled, y_categorical
    
    def build_model(self, input_dim, num_classes):
        """Build an optimized neural network with better architecture for your data"""
        print(f"\nBuilding model with input_dim={input_dim}, num_classes={num_classes}")
        
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(input_dim,)),
            
            # Larger first layer to capture patterns better
            tf.keras.layers.Dense(256, activation='relu', 
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            # Second layer
            tf.keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Third layer
            tf.keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Fourth layer for better feature learning
            tf.keras.layers.Dense(32, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Use a more aggressive learning rate with focal loss approach
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Higher learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("\nModel architecture:")
        model.summary()
        
        return model
    
    def train_with_cross_validation(self, X, y, n_folds=5, epochs=300):
        """Train with cross-validation using more epochs and better callbacks"""
        print(f"\nTraining with {n_folds}-fold cross validation...")
        
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_accuracies = []
        
        # Convert categorical back to class indices for stratification
        y_classes = np.argmax(y, axis=1)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_classes)):
            print(f"\nTraining fold {fold + 1}/{n_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Build fresh model for each fold
            fold_model = self.build_model(X.shape[1], y.shape[1])
            
            # More aggressive callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=50,  # Increased patience
                    restore_best_weights=True,
                    verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,  # More aggressive reduction
                    patience=20,  # More patience before reducing
                    min_lr=1e-7,
                    verbose=0
                )
            ]
            
            # Train fold model with more epochs
            history = fold_model.fit(
                X_train_fold, y_train_fold,
                batch_size=32,  # Smaller batch size for better gradients
                epochs=epochs,
                validation_data=(X_val_fold, y_val_fold),
                class_weight=self.class_weight_dict,
                callbacks=callbacks,
                verbose=1  # Show progress
            )
            
            # Evaluate fold
            fold_accuracy = max(history.history['val_accuracy'])
            fold_accuracies.append(fold_accuracy)
            print(f"Fold {fold + 1} accuracy: {fold_accuracy:.4f}")
            
            # Keep the best model
            if fold == 0 or fold_accuracy > max(fold_accuracies[:-1]):
                self.model = fold_model
        
        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies) * 2:.4f})")
        print(f"Best accuracy: {max(fold_accuracies):.4f}")
        
        return fold_accuracies
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        print("\nEvaluating model...")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Classification report
        class_names = ['Normal (0)', 'Mild Hypoxia (1)', 'Severe Hypoxia (2)']
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        print("\nClassification Report:")
        print(report)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate overall accuracy
        overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
        print(f"\nOverall accuracy: {overall_accuracy:.4f}")
        
        return report, cm
    
    def save_model(self, model_dir='trained_model'):
        """Save the trained model including TensorFlow.js format"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(model_dir, 'hypoxia_model.h5')
        self.model.save(model_path)
        print(f"Keras model saved to: {model_path}")
        
        # Convert to TensorFlow.js format
        try:
            # NumPy 2.x compatibility for older TFJS converters expecting deprecated aliases
            try:
                _ = np.object  # type: ignore[attr-defined]
            except AttributeError:
                np.object = object  # type: ignore[attr-defined]
                np.bool = bool      # type: ignore[attr-defined]

            import tensorflowjs as tfjs
            # Export directly to the web-facing folder used by the frontend
            web_model_dir = os.path.join('public', 'web_model')
            os.makedirs(web_model_dir, exist_ok=True)
            
            # Convert the saved model to TensorFlow.js format with modest quantization
            tfjs.converters.save_keras_model(self.model, web_model_dir, quantization_bytes=2)

            print(f"TensorFlow.js model saved to: {web_model_dir}")
            
            # List the generated files
            print("Generated files:")
            for file in os.listdir(web_model_dir):
                print(f"  - {file}")
                
        except ImportError:
            print("TensorFlow.js not installed. Install with: pip install tensorflowjs")
        except Exception as e:
            print(f"Error converting to TensorFlow.js: {e}")
            print("Try installing tensorflowjs: pip install tensorflowjs")
        
        # Save preprocessing parameters
        preprocessing_params = {
            'scaler_center': self.scaler.center_.tolist() if hasattr(self.scaler, 'center_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'label_classes': self.label_encoder.classes_.tolist(),
            'feature_columns': self.feature_columns,
            'class_weights': self.class_weight_dict,
            'model_input_shape': self.model.input_shape,
            'model_output_shape': self.model.output_shape
        }
        
        params_path = os.path.join(model_dir, 'preprocessing_params.json')
        with open(params_path, 'w') as f:
            json.dump(preprocessing_params, f, indent=2)
        print(f"Preprocessing parameters saved to: {params_path}")
        
        return model_path
    
    def run_training_pipeline(self, csv_path=None, test_size=0.2):
        """Run the complete training pipeline"""
        print("="*60)
        print("HYPOXIA CLASSIFICATION MODEL TRAINING")
        print("="*60)
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(csv_path or self.data_path)
        X, y = self.prepare_features_and_labels(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=np.argmax(y, axis=1), 
            random_state=42
        )
        
        print(f"\nData split:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        # Train with cross-validation
        cv_scores = self.train_with_cross_validation(X_train, y_train, n_folds=6, epochs=300)
        
        # Final evaluation on test set
        test_report, test_cm = self.evaluate_model(X_test, y_test)
        
        # Save the model
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print(f"Cross-validation accuracy: {np.mean(cv_scores):.1%}")
        print("="*60)
        
        return self.model, cv_scores


def main():
    """Main training function"""
    # Your dataset path
    DATA_PATH = r"C:\Users\Dell 2023#\Downloads\HTML\project\model\health_data.csv"
    
    # Initialize trainer
    trainer = CleanHypoxiaModelTrainer()
    
    try:
        # Run training
        model, cv_scores = trainer.run_training_pipeline(csv_path=DATA_PATH)
        
        print(f"\nTraining completed!")
        print(f"Final accuracy: {np.mean(cv_scores):.1%}")
        
        if np.mean(cv_scores) >= 0.95:
            print("Target accuracy of 95%+ achieved!")
        else:
            print(f"Current accuracy: {np.mean(cv_scores):.1%}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()