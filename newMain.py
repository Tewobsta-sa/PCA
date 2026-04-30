import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_wine, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

print("="*80)
print("REALISTIC PCA ANALYSIS WITH LARGE DATASETS")
print("="*80)

def analyze_dataset(name, X, y, dataset_size="medium"):
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name.upper()}")
    print(f"Dataset size: {dataset_size}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Features: {X.shape[1]}")
    
    # Proper stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test multiple models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        # WITHOUT PCA
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time_no_pca = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        accuracy_no_pca = accuracy_score(y_test, y_pred)
        
        # WITH PCA (retain 95% variance)
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        start_time = time.time()
        model_pca = type(model)(**model.get_params())
        model_pca.fit(X_train_pca, y_train)
        train_time_pca = time.time() - start_time
        
        y_pred_pca = model_pca.predict(X_test_pca)
        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        
        # Cross-validation (more robust)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_no_pca = cross_val_score(model, X_train_scaled, y_train, cv=cv)
        cv_scores_pca = cross_val_score(model, X_train_pca, y_train, cv=cv)
        
        results[model_name] = {
            'accuracy_no_pca': accuracy_no_pca,
            'accuracy_pca': accuracy_pca,
            'cv_no_pca': cv_scores_no_pca.mean(),
            'cv_pca': cv_scores_pca.mean(),
            'train_time_no_pca': train_time_no_pca,
            'train_time_pca': train_time_pca,
            'original_features': X_train_scaled.shape[1],
            'pca_features': X_train_pca.shape[1],
            'variance_retained': sum(pca.explained_variance_ratio_),
            'explained_variance': pca.explained_variance_ratio_
        }
        
        print(f"Without PCA: {accuracy_no_pca:.3f} (CV: {cv_scores_no_pca.mean():.3f} ± {cv_scores_no_pca.std():.3f})")
        print(f"With PCA: {accuracy_pca:.3f} (CV: {cv_scores_pca.mean():.3f} ± {cv_scores_pca.std():.3f})")
        print(f"Features: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} ({sum(pca.explained_variance_ratio_):.1%} variance)")
        print(f"Training time: {train_time_no_pca:.3f}s → {train_time_pca:.3f}s")
    
    return results, pca

# Dataset 1: Wine (small, baseline)
print("\n1. WINE DATASET (Small - Baseline)")
wine = load_wine()
results_wine, pca_wine = analyze_dataset("Wine", wine.data, wine.target, "small")

# Dataset 2: Digits (medium)
print("\n2. DIGITS DATASET (Medium)")
digits = load_digits()
results_digits, pca_digits = analyze_dataset("Digits", digits.data, digits.target, "medium")

# Dataset 3: MNIST subset (large)
print("\n3. MNIST DATASET (Large)")
try:
    # Load a subset of MNIST for faster processing
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    # Take first 10000 samples for manageable processing
    X_mnist = mnist.data[:10000]
    y_mnist = mnist.target[:10000].astype(int)
    results_mnist, pca_mnist = analyze_dataset("MNIST", X_mnist, y_mnist, "large")
except Exception as e:
    print(f"Could not load MNIST: {e}")
    # Use synthetic dataset as fallback
    from sklearn.datasets import make_classification
    X_synthetic, y_synthetic = make_classification(
        n_samples=10000, n_features=100, n_informative=50, 
        n_redundant=25, n_classes=10, random_state=42
    )
    results_mnist, pca_mnist = analyze_dataset("Synthetic Large", X_synthetic, y_synthetic, "large")

# Comprehensive Analysis
print("\n" + "="*80)
print("COMPREHENSIVE PCA ANALYSIS SUMMARY")
print("="*80)

datasets = [
    ("Wine (Small)", results_wine, pca_wine),
    ("Digits (Medium)", results_digits, pca_digits), 
    ("MNIST (Large)", results_mnist, pca_mnist)
]

for dataset_name, results, pca in datasets:
    print(f"\n{dataset_name}:")
    print("-" * 40)
    
    for model_name, metrics in results.items():
        accuracy_diff = metrics['accuracy_no_pca'] - metrics['accuracy_pca']
        time_improvement = (metrics['train_time_no_pca'] - metrics['train_time_pca']) / metrics['train_time_no_pca'] * 100
        
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy_no_pca']:.3f} → {metrics['accuracy_pca']:.3f} (Δ{accuracy_diff:+.3f})")
        print(f"  Features: {metrics['original_features']} → {metrics['pca_features']} ({metrics['pca_features']/metrics['original_features']:.1%})")
        print(f"  Training time: {time_improvement:+.1f}% change")
        print(f"  Variance retained: {metrics['variance_retained']:.1%}")

# Mathematical Analysis
print("\n" + "="*80)
print("MATHEMATICAL PCA INSIGHTS")
print("="*80)

for dataset_name, results, pca in datasets:
    print(f"\n{dataset_name} - Principal Components Analysis:")
    print(f"Total components: {len(pca.explained_variance_ratio_)}")
    
    # First 5 components
    print("First 5 components variance:")
    for i in range(min(5, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
    
    # Cumulative variance
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    print(f"Components needed for 90% variance: {np.argmax(cumulative >= 0.9) + 1}")
    print(f"Components needed for 95% variance: {np.argmax(cumulative >= 0.95) + 1}")
    print(f"Components needed for 99% variance: {np.argmax(cumulative >= 0.99) + 1 if np.any(cumulative >= 0.99) else 'All'}")

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Variance explained by dataset
plt.subplot(2, 3, 1)
for i, (dataset_name, _, pca) in enumerate(datasets):
    plt.plot(range(1, min(11, len(pca.explained_variance_ratio_) + 1)), 
             pca.explained_variance_ratio_[:10], 
             label=dataset_name, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance per Component (First 10 PCs)')
plt.legend()

# Plot 2: Cumulative variance
plt.subplot(2, 3, 2)
for i, (dataset_name, _, pca) in enumerate(datasets):
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, min(11, len(cumulative) + 1)), 
             cumulative[:10], 
             label=dataset_name, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
plt.legend()

# Plot 3: Feature reduction vs accuracy
plt.subplot(2, 3, 3)
for dataset_name, results, _ in datasets:
    models = list(results.keys())
    no_pca_accs = [results[m]['accuracy_no_pca'] for m in models]
    pca_accs = [results[m]['accuracy_pca'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, no_pca_accs, width, label='No PCA', alpha=0.8)
    plt.bar(x + width/2, pca_accs, width, label='With PCA', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name}: Accuracy Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()

# Plot 4: Training time comparison
plt.subplot(2, 3, 4)
for dataset_name, results, _ in datasets:
    models = list(results.keys())
    no_pca_times = [results[m]['train_time_no_pca'] for m in models]
    pca_times = [results[m]['train_time_pca'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, no_pca_times, width, label='No PCA', alpha=0.8)
    plt.bar(x + width/2, pca_times, width, label='With PCA', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'{dataset_name}: Training Time')
    plt.xticks(x, models, rotation=45)
    plt.legend()

# Plot 5: Dimensionality reduction
plt.subplot(2, 3, 5)
dataset_names = []
reduction_ratios = []
for dataset_name, results, _ in datasets:
    # Use Logistic Regression results for consistency
    lr_results = results['Logistic Regression']
    reduction_ratios.append(lr_results['pca_features'] / lr_results['original_features'])
    dataset_names.append(dataset_name.replace(' (Small)', '').replace(' (Medium)', '').replace(' (Large)', ''))

plt.bar(dataset_names, reduction_ratios)
plt.ylabel('Feature Retention Ratio')
plt.title('Dimensionality Reduction Effectiveness')
plt.ylim(0, 1)

# Plot 6: Accuracy preservation
plt.subplot(2, 3, 6)
accuracy_preservation = []
for dataset_name, results, _ in datasets:
    lr_results = results['Logistic Regression']
    preservation = lr_results['accuracy_pca'] / lr_results['accuracy_no_pca']
    accuracy_preservation.append(preservation)

plt.bar(dataset_names, accuracy_preservation)
plt.ylabel('Accuracy Preservation Ratio')
plt.title('Accuracy Preservation with PCA')
plt.ylim(0.8, 1.05)

plt.tight_layout()
plt.savefig('realistic_pca_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'realistic_pca_analysis.png'")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("1. 100% accuracy is unrealistic on real-world datasets")
print("2. PCA effectiveness varies by dataset complexity")
print("3. Trade-off: dimensionality reduction vs accuracy preservation")
print("4. Training time improvements with PCA on larger datasets")
print("5. Mathematical understanding: eigenvalues show information distribution")
print("\nThis analysis provides realistic expectations for PCA performance!")
