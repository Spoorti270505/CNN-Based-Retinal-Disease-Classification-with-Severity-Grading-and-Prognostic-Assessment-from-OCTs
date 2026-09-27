import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
import random
import time
import timm
import cv2
from collections import Counter
from scipy import stats

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ---------------------------------------------------------
# 1. CONFIGURATION & SEEDING
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

CONFIG = {
    'SEED': 42,
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,         
    'GRAD_ACCUM_STEPS': 2,    
    'EPOCHS': 20,           
    'LR': 1e-4,
    'NUM_WORKERS': 2,
    'DATA_PATH': "data/kermany2018/OCT2017",
    'WEIGHT_DECAY': 0.01,
    'BACKBONE': 'densenet121',  # Only DenseNet121
    'PATHOLOGY_THRESHOLDS': {
        'CNV': {
            'area_threshold': 0.08,
            'intensity_min': 50,
            'intensity_max': 60,
            'severity_threshold': 0.994551
        },
        'DME': {
            'area_threshold': 0.1,
            'intensity_min': 0,
            'intensity_max': 50,
            'severity_threshold': 0.583818
        },
        'DRUSEN': {
            'area_threshold': 0.12,
            'intensity_min': 220,
            'intensity_max': 255,
            'severity_threshold': 0.008863
        },
    },
    'SEVERITY_THRESHOLD': 0.20
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(CONFIG['SEED'])

# ---------------------------------------------------------
# 2. ADVANCED REGRESSION METRICS
# ---------------------------------------------------------

def calculate_advanced_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics with confidence intervals"""
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Median Absolute Error
    medae = np.median(np.abs(y_true - y_pred))
    
    # Mean Squared Logarithmic Error (MSLE) - for non-negative values
    msle = mean_squared_error(np.log1p(np.abs(y_true)), np.log1p(np.abs(y_pred)))
    
    # Explained Variance Score
    explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Pearson Correlation
    pearson_corr, pearson_pval = stats.pearsonr(y_true, y_pred)
    
    # Spearman Correlation
    spearman_corr, spearman_pval = stats.spearmanr(y_true, y_pred)
    
    # Bootstrap Confidence Intervals
    n_bootstrap = 1000
    bootstrap_mae = []
    bootstrap_rmse = []
    bootstrap_r2 = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        mae_boot = mean_absolute_error(y_true_boot, y_pred_boot)
        rmse_boot = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
        r2_boot = r2_score(y_true_boot, y_pred_boot)
        
        bootstrap_mae.append(mae_boot)
        bootstrap_rmse.append(rmse_boot)
        bootstrap_r2.append(r2_boot)
    
    # 95% Confidence Intervals
    ci_mae = np.percentile(bootstrap_mae, [2.5, 97.5])
    ci_rmse = np.percentile(bootstrap_rmse, [2.5, 97.5])
    ci_r2 = np.percentile(bootstrap_r2, [2.5, 97.5])
    
    # Error quantiles
    errors = np.abs(y_true - y_pred)
    quantiles = {
        'q25': np.percentile(errors, 25),
        'q50': np.percentile(errors, 50),  # Median
        'q75': np.percentile(errors, 75),
        'q95': np.percentile(errors, 95),
        'q99': np.percentile(errors, 99)
    }
    
    # Prediction intervals (95%)
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    prediction_interval_95 = 1.96 * residual_std
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'medae': medae,
        'msle': msle,
        'explained_variance': explained_var,
        'max_error': max_error,
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_pval,
        'spearman_corr': spearman_corr,
        'spearman_pval': spearman_pval,
        'ci_mae': ci_mae,
        'ci_rmse': ci_rmse,
        'ci_r2': ci_r2,
        'quantiles': quantiles,
        'prediction_interval_95': prediction_interval_95
    }

def calculate_error_by_bins(y_true, y_pred, n_bins=10):
    """Calculate MAE for different ranges of true values"""
    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_maes = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])
        
        if np.sum(mask) > 0:
            bin_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            bin_maes.append(bin_mae)
            bin_counts.append(np.sum(mask))
        else:
            bin_maes.append(0)
            bin_counts.append(0)
    
    return bin_centers, bin_maes, bin_counts

# ---------------------------------------------------------
# 3. PATHOLOGICAL FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_prognosis_label(img_path, disease_label, disease_name):
    """Extract prognosis score from OCT images"""
    if disease_name == 'NORMAL':
        return 0.0
    
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.5
        
        img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
        
        height, width = img.shape
        roi_middle = img[int(height*0.3):int(height*0.7), :]
        roi_lower = img[int(height*0.7):, :]
        
        thresholds = CONFIG['PATHOLOGY_THRESHOLDS'].get(disease_name, {})
        intensity_min = thresholds.get('intensity_min', 200)
        intensity_max = thresholds.get('intensity_max', 255)
        
        if disease_name == 'DRUSEN':
            _, mask = cv2.threshold(roi_lower, intensity_min, 255, cv2.THRESH_BINARY)
            pathology_area_ratio = np.sum(mask == 255) / mask.size
            deposit_variance = np.var(roi_lower[mask == 255]) if np.any(mask == 255) else 0
            prognosis_raw = pathology_area_ratio * 3.0 + (deposit_variance / 10000.0)
            
        elif disease_name == 'CNV':
            _, mask = cv2.threshold(roi_lower, intensity_max, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            pathology_area_ratio = np.sum(mask == 255) / mask.size
            texture_variance = np.var(roi_lower)
            prognosis_raw = pathology_area_ratio * 4.0 + (texture_variance / 20000.0)
            
        elif disease_name == 'DME':
            _, mask = cv2.threshold(roi_middle, intensity_max, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            pathology_area_ratio = np.sum(mask == 255) / mask.size
            cyst_distribution = np.std(np.where(mask == 255)[0]) if np.any(mask == 255) else 0
            prognosis_raw = pathology_area_ratio * 3.5 + (cyst_distribution / 100.0)
        else:
            prognosis_raw = 0.3
        
        prognosis = float(np.clip(prognosis_raw, 0.05, 0.95))
        return prognosis
        
    except Exception as e:
        return 0.5

# ---------------------------------------------------------
# 4. DATA PARSING
# ---------------------------------------------------------
def parse_dataset_with_patients(base_path):
    """Parse dataset with patient-aware splitting"""
    print(f"\n📂 Scanning dataset at {base_path}...")
    disease_classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    disease_to_idx = {d: i for i, d in enumerate(disease_classes)}
    all_records = []
    
    prognosis_stats = {cls: [] for cls in disease_classes}
    
    splits = ['train', 'val']
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path): 
            print(f"⚠️ Warning: {split_path} not found")
            continue
            
        for disease in disease_classes:
            class_path = os.path.join(split_path, disease)
            if not os.path.exists(class_path): 
                continue
                
            image_files = glob.glob(os.path.join(class_path, '*.jpeg')) + \
                         glob.glob(os.path.join(class_path, '*.jpg'))
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                try:
                    parts = filename.split('-')
                    patient_id = parts[1] if len(parts) >= 3 else filename.split('.')[0]
                except:
                    patient_id = filename
                
                prognosis = extract_prognosis_label(img_path, disease_to_idx[disease], disease)
                
                if disease != 'NORMAL':
                    prognosis_stats[disease].append(prognosis)
                
                all_records.append({
                    'path': img_path,
                    'patient_id': patient_id,
                    'disease_label': disease_to_idx[disease],
                    'prognosis_label': prognosis,
                    'split': split
                })
                
    df = pd.DataFrame(all_records)
    print(f"✅ Loaded {len(df)} images. Unique Patients: {df['patient_id'].nunique()}")
    
    print(f"\n📊 Prognosis Score Statistics:")
    for disease in ['CNV', 'DME', 'DRUSEN']:
        scores = prognosis_stats[disease]
        if len(scores) > 0:
            print(f"   {disease}:")
            print(f"      Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            print(f"      Range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
            
    return df, disease_classes

# ---------------------------------------------------------
# 5. DATASET CLASS
# ---------------------------------------------------------
class OCTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        try:
            image = Image.open(img_path)
            if image.mode == 'L':
                image = Image.merge('RGB', (image, image, image))
            elif image.mode != 'RGB':
                image = image.convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0,0,0))
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(row['prognosis_label'], dtype=torch.float32)
        return image, label, img_path

train_transform = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 6. MODEL ARCHITECTURE
# ---------------------------------------------------------
class PrognosisRegressionNet(nn.Module):
    def __init__(self, backbone_name='densenet121'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        output = self.regressor(features).squeeze()
        
        if return_features:
            return output, features
        return output

# ---------------------------------------------------------
# 7. VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def plot_predictions_vs_actual(y_true, y_pred, metrics, save_dir="resultnewofprog"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add R² and RMSE to plot
    textstr = f'R² = {metrics["r2"]:.4f}\nRMSE = {metrics["rmse"]:.4f}'
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[0].set_xlabel('True Prognosis Score', fontweight='bold')
    axes[0].set_ylabel('Predicted Prognosis Score', fontweight='bold')
    axes[0].set_title('Predicted vs Actual Prognosis Scores', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].axhline(metrics['prediction_interval_95'], color='orange', linestyle='--', 
                   linewidth=1.5, label=f'95% PI: ±{metrics["prediction_interval_95"]:.3f}')
    axes[1].axhline(-metrics['prediction_interval_95'], color='orange', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('True Prognosis Score', fontweight='bold')
    axes[1].set_ylabel('Residuals (Predicted - True)', fontweight='bold')
    axes[1].set_title('Residual Plot with Prediction Intervals', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prognosis_predictions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/prognosis_predictions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction plots saved")

def plot_error_distribution(y_true, y_pred, metrics, save_dir="resultnewofprog"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # 1. Error histogram
    axes[0, 0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 0].axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(errors):.4f}')
    axes[0, 0].axvline(np.median(errors), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(errors):.4f}')
    axes[0, 0].set_xlabel('Prediction Error', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Error Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Absolute error by true value
    axes[0, 1].scatter(y_true, abs_errors, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    axes[0, 1].axhline(metrics['mae'], color='red', linestyle='--', linewidth=2,
                       label=f'MAE: {metrics["mae"]:.4f}')
    axes[0, 1].axhline(metrics['medae'], color='green', linestyle='--', linewidth=2,
                       label=f'MedAE: {metrics["medae"]:.4f}')
    axes[0, 1].set_xlabel('True Prognosis Score', fontweight='bold')
    axes[0, 1].set_ylabel('Absolute Error', fontweight='bold')
    axes[0, 1].set_title('Absolute Error vs True Value', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    stats.probplot(errors, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality of Errors)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error quantiles
    quantiles_names = ['25th', '50th', '75th', '95th', '99th']
    quantiles_values = [
        metrics['quantiles']['q25'],
        metrics['quantiles']['q50'],
        metrics['quantiles']['q75'],
        metrics['quantiles']['q95'],
        metrics['quantiles']['q99']
    ]
    
    bars = axes[1, 1].bar(quantiles_names, quantiles_values, 
                          color='steelblue', edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Absolute Error', fontweight='bold')
    axes[1, 1].set_title('Error Quantiles', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, quantiles_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prognosis_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/prognosis_error_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error analysis plots saved")

def plot_error_by_range(y_true, y_pred, save_dir="resultnewofprog"):
    """Plot MAE across different ranges of prognosis values"""
    os.makedirs(save_dir, exist_ok=True)
    
    bin_centers, bin_maes, bin_counts = calculate_error_by_bins(y_true, y_pred, n_bins=10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE by bin
    axes[0].plot(bin_centers, bin_maes, 'o-', linewidth=2, markersize=8, 
                color='steelblue', label='MAE')
    axes[0].set_xlabel('Prognosis Score Range', fontweight='bold')
    axes[0].set_ylabel('Mean Absolute Error', fontweight='bold')
    axes[0].set_title('MAE Across Prognosis Score Ranges', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Sample counts per bin
    axes[1].bar(bin_centers, bin_counts, width=(bin_centers[1]-bin_centers[0])*0.8,
               color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Prognosis Score Range', fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontweight='bold')
    axes[1].set_title('Sample Distribution Across Ranges', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_by_range.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/error_by_range.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error by range plots saved")

def plot_correlation_analysis(y_true, y_pred, metrics, save_dir="resultnewofprog"):
    """Plot correlation analysis with confidence bands"""
    os.makedirs(save_dir, exist_ok=True)

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with regression line
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Fitted Line: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Perfect prediction line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Prediction')
    
    # Add statistics box
    textstr = '\n'.join([
        f'Pearson r = {metrics["pearson_corr"]:.4f} (p={metrics["pearson_pval"]:.2e})',
        f'Spearman ρ = {metrics["spearman_corr"]:.4f} (p={metrics["spearman_pval"]:.2e})',
        f'R² = {metrics["r2"]:.4f}',
        f'MAE = {metrics["mae"]:.4f} [{metrics["ci_mae"][0]:.4f}, {metrics["ci_mae"][1]:.4f}]',
        f'RMSE = {metrics["rmse"]:.4f} [{metrics["ci_rmse"][0]:.4f}, {metrics["ci_rmse"][1]:.4f}]'
    ])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('True Prognosis Score', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Prognosis Score', fontweight='bold', fontsize=12)
    ax.set_title('Correlation Analysis with 95% Confidence Intervals', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/correlation_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation analysis saved")

def plot_training_history(history, save_dir="resultnewofprog"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', linewidth=2, label='Train', color='blue')
    axes[0].plot(epochs, history['val_loss'], 's-', linewidth=2, label='Validation', color='red')
    axes[0].set_title('Training and Validation Loss (MSE) - DenseNet121', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(epochs, history['val_mae'], 'o-', linewidth=2, color='green')
    axes[1].axhline(min(history['val_mae']), linestyle='--', color='red', alpha=0.7,
                   label=f'Best: {min(history["val_mae"]):.4f}')
    axes[1].set_title('Validation MAE - DenseNet121', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prognosis_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved")

# Add this as a separate script or at the end of your main file
def regenerate_visualizations_only():
    """Load saved model and regenerate visualizations without retraining"""
    print("="*80)
    print("🎨 REGENERATING VISUALIZATIONS FROM SAVED MODEL")
    print("="*80)
    
    # Load data
    df, disease_classes = parse_dataset_with_patients(CONFIG['DATA_PATH'])
    
    # Recreate test split (same as before)
    train_data = df[df['split'] == 'train'].reset_index(drop=True)
    all_patients = train_data['patient_id'].unique()
    
    patient_labels = []
    for pid in all_patients:
        patient_data = train_data[train_data['patient_id'] == pid]
        majority_label = patient_data['disease_label'].mode()[0]
        patient_labels.append(majority_label)
    
    train_patients, temp_patients = train_test_split(
        all_patients, test_size=0.30, stratify=patient_labels, random_state=CONFIG['SEED']
    )
    
    temp_patient_labels = []
    for pid in temp_patients:
        patient_data = train_data[train_data['patient_id'] == pid]
        majority_label = patient_data['disease_label'].mode()[0]
        temp_patient_labels.append(majority_label)
    
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50, stratify=temp_patient_labels, random_state=CONFIG['SEED']
    )
    
    test_df = train_data[train_data['patient_id'].isin(test_patients)].reset_index(drop=True)
    
    test_loader = DataLoader(
        OCTDataset(test_df, val_transform),
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    # Load the saved model
    print("📥 Loading saved model...")
    model = PrognosisRegressionNet(backbone_name=CONFIG['BACKBONE']).to(device)
    model.load_state_dict(torch.load('resultnewofprog/best_prognosis_model_densenet121.pth'))
    
    # Evaluate on test set
    print("🔬 Evaluating on test set...")
    criterion = nn.MSELoss()
    _, test_mae, test_mse, test_rmse, test_r2, y_true, y_pred = validate(model, test_loader, criterion, device)
    
    # Convert to float64 to avoid the error
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    
    # Calculate advanced metrics
    print("📊 Computing Advanced Regression Metrics...")
    advanced_metrics = calculate_advanced_regression_metrics(y_true, y_pred)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   MAE:  {advanced_metrics['mae']:.4f} [{advanced_metrics['ci_mae'][0]:.4f}, {advanced_metrics['ci_mae'][1]:.4f}]")
    print(f"   RMSE: {advanced_metrics['rmse']:.4f} [{advanced_metrics['ci_rmse'][0]:.4f}, {advanced_metrics['ci_rmse'][1]:.4f}]")
    print(f"   R²:   {advanced_metrics['r2']:.4f} [{advanced_metrics['ci_r2'][0]:.4f}, {advanced_metrics['ci_r2'][1]:.4f}]")
    
    # Generate all visualizations with fixed function
    print("\n🎨 Generating Visualizations...")
    plot_predictions_vs_actual(y_true, y_pred, advanced_metrics, save_dir='resultnewofprog')
    plot_error_distribution(y_true, y_pred, advanced_metrics, save_dir='resultnewofprog')
    plot_error_by_range(y_true, y_pred, save_dir='resultnewofprog')
    plot_correlation_analysis(y_true, y_pred, advanced_metrics, save_dir='resultnewofprog')
    
    print("\n✅ Visualizations regenerated successfully!")

# ---------------------------------------------------------
# 8. TRAINING ENGINE
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / CONFIG['GRAD_ACCUM_STEPS']
        
        scaler.scale(loss_scaled).backward()
        
        if (i + 1) % CONFIG['GRAD_ACCUM_STEPS'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item()
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().float().numpy())
            all_labels.extend(labels.cpu().float().numpy())

        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds,dtype=np.float32)
    all_labels = np.array(all_labels,dtype=np.float32)
    
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    return running_loss / len(loader), mae, mse, rmse, r2, all_labels, all_preds

# ---------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("="*80)
    print("🚀 DENSENET121 PROGNOSIS PREDICTION WITH ADVANCED METRICS")
    print("="*80)
    print(f"\n💻 Hardware: Using {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and parse data
    df, disease_classes = parse_dataset_with_patients(CONFIG['DATA_PATH'])
    
    # Patient-aware splitting
    print("\n" + "="*80)
    print("📂 DATA SPLITTING (Patient-Aware)")
    print("="*80)
    
    train_data = df[df['split'] == 'train'].reset_index(drop=True)
    all_patients = train_data['patient_id'].unique()
    
    # Use disease labels for stratification (since prognosis is continuous)
    patient_labels = []
    for pid in all_patients:
        patient_data = train_data[train_data['patient_id'] == pid]
        majority_label = patient_data['disease_label'].mode()[0]
        patient_labels.append(majority_label)
    
    # 70% train, 30% temp (15% val + 15% test)
    train_patients, temp_patients = train_test_split(
        all_patients, test_size=0.30, stratify=patient_labels, random_state=CONFIG['SEED']
    )
    
    temp_patient_labels = []
    for pid in temp_patients:
        patient_data = train_data[train_data['patient_id'] == pid]
        majority_label = patient_data['disease_label'].mode()[0]
        temp_patient_labels.append(majority_label)
    
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.50, stratify=temp_patient_labels, random_state=CONFIG['SEED']
    )
    
    train_df = train_data[train_data['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = train_data[train_data['patient_id'].isin(val_patients)].reset_index(drop=True)
    test_df = train_data[train_data['patient_id'].isin(test_patients)].reset_index(drop=True)
    
    print(f"   Training:   {len(train_df):6d} images ({train_df['patient_id'].nunique():4d} patients)")
    print(f"   Validation: {len(val_df):6d} images ({val_df['patient_id'].nunique():4d} patients)")
    print(f"   Test:       {len(test_df):6d} images ({test_df['patient_id'].nunique():4d} patients)")
    
    # Create data loaders
    train_loader = DataLoader(
        OCTDataset(train_df, train_transform),
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    val_loader = DataLoader(
        OCTDataset(val_df, val_transform),
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    test_loader = DataLoader(
        OCTDataset(test_df, val_transform),
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']
    )
    
    # Initialize model
    print("\n" + "="*80)
    print(f"🏗️ INITIALIZING DENSENET121")
    print("="*80)
    
    model = PrognosisRegressionNet(backbone_name=CONFIG['BACKBONE']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'])
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
    
    best_val_mae = float('inf')
    best_epoch = 0
    
    print(f"\n🏋️ Training DenseNet121...")
    
    for epoch in range(CONFIG['EPOCHS']):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        
        # Validate
        val_loss, val_mae, val_mse, val_rmse, val_r2, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{CONFIG['EPOCHS']}] - {time.time()-epoch_start:.1f}s")
            print(f"    Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'resultnewofprog/best_prognosis_model_densenet121.pth')
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation MAE: {best_val_mae:.4f} (Epoch {best_epoch})")
    
    # Load best model for testing
    model.load_state_dict(torch.load('resultnewofprog/best_prognosis_model_densenet121.pth'))
    
    # Test evaluation
    print(f"\n🔬 Evaluating DenseNet121 on test set...")
    
    _, test_mae, test_mse, test_rmse, test_r2, y_true, y_pred = validate(model, test_loader, criterion, device)
    
    # Calculate advanced metrics
    print(f"\n📊 Computing Advanced Regression Metrics...")
    advanced_metrics = calculate_advanced_regression_metrics(y_true, y_pred)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   MAE:  {advanced_metrics['mae']:.4f} [{advanced_metrics['ci_mae'][0]:.4f}, {advanced_metrics['ci_mae'][1]:.4f}]")
    print(f"   RMSE: {advanced_metrics['rmse']:.4f} [{advanced_metrics['ci_rmse'][0]:.4f}, {advanced_metrics['ci_rmse'][1]:.4f}]")
    print(f"   R²:   {advanced_metrics['r2']:.4f} [{advanced_metrics['ci_r2'][0]:.4f}, {advanced_metrics['ci_r2'][1]:.4f}]")
    print(f"   MedAE: {advanced_metrics['medae']:.4f}")
    print(f"   MAPE: {advanced_metrics['mape']:.2f}%")
    print(f"   Explained Variance: {advanced_metrics['explained_variance']:.4f}")
    print(f"   Max Error: {advanced_metrics['max_error']:.4f}")
    print(f"   Pearson Correlation: {advanced_metrics['pearson_corr']:.4f} (p={advanced_metrics['pearson_pval']:.2e})")
    print(f"   Spearman Correlation: {advanced_metrics['spearman_corr']:.4f} (p={advanced_metrics['spearman_pval']:.2e})")
    
    print(f"\n📊 Error Quantiles:")
    print(f"   25th percentile: {advanced_metrics['quantiles']['q25']:.4f}")
    print(f"   50th percentile (Median): {advanced_metrics['quantiles']['q50']:.4f}")
    print(f"   75th percentile: {advanced_metrics['quantiles']['q75']:.4f}")
    print(f"   95th percentile: {advanced_metrics['quantiles']['q95']:.4f}")
    print(f"   99th percentile: {advanced_metrics['quantiles']['q99']:.4f}")
    
    # Generate visualizations
    print(f"\n🎨 Generating Comprehensive Visualizations...")
    
    # 1. Training history
    plot_training_history(history, save_dir='resultnewofprog')
    
    # 2. Predictions vs Actual
    plot_predictions_vs_actual(y_true, y_pred, advanced_metrics, save_dir='resultnewofprog')
    
    # 3. Error distribution and analysis
    plot_error_distribution(y_true, y_pred, advanced_metrics, save_dir='resultnewofprog')
    
    # 4. Error by range
    plot_error_by_range(y_true, y_pred, save_dir='resultnewofprog')
    
    # 5. Correlation analysis
    plot_correlation_analysis(y_true, y_pred, advanced_metrics, save_dir='resultnewofprog')
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_prognosis': y_true,
        'predicted_prognosis': y_pred,
        'absolute_error': np.abs(y_pred - y_true),
        'squared_error': (y_pred - y_true) ** 2
    })
    results_df.to_csv('resultnewofprog/prognosis_predictions_densenet121.csv', index=False)
    
    # Save all metrics to CSV
    metrics_summary = {
        'Model': 'DenseNet121',
        'Total_Parameters': total_params,
        'MAE': advanced_metrics['mae'],
        'MAE_CI_Lower': advanced_metrics['ci_mae'][0],
        'MAE_CI_Upper': advanced_metrics['ci_mae'][1],
        'RMSE': advanced_metrics['rmse'],
        'RMSE_CI_Lower': advanced_metrics['ci_rmse'][0],
        'RMSE_CI_Upper': advanced_metrics['ci_rmse'][1],
        'R2': advanced_metrics['r2'],
        'R2_CI_Lower': advanced_metrics['ci_r2'][0],
        'R2_CI_Upper': advanced_metrics['ci_r2'][1],
        'MedAE': advanced_metrics['medae'],
        'MAPE': advanced_metrics['mape'],
        'MSE': advanced_metrics['mse'],
        'MSLE': advanced_metrics['msle'],
        'Explained_Variance': advanced_metrics['explained_variance'],
        'Max_Error': advanced_metrics['max_error'],
        'Pearson_Correlation': advanced_metrics['pearson_corr'],
        'Pearson_PValue': advanced_metrics['pearson_pval'],
        'Spearman_Correlation': advanced_metrics['spearman_corr'],
        'Spearman_PValue': advanced_metrics['spearman_pval'],
        'Error_Q25': advanced_metrics['quantiles']['q25'],
        'Error_Q50': advanced_metrics['quantiles']['q50'],
        'Error_Q75': advanced_metrics['quantiles']['q75'],
        'Error_Q95': advanced_metrics['quantiles']['q95'],
        'Error_Q99': advanced_metrics['quantiles']['q99'],
        'Prediction_Interval_95': advanced_metrics['prediction_interval_95']
    }
    
    pd.DataFrame([metrics_summary]).to_csv('resultnewofprog/advanced_metrics_summary_densenet121.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ DENSENET121 PROGNOSIS PREDICTION WITH ADVANCED METRICS COMPLETED!")
    print("="*80)
    print(f"📂 All results saved in 'resultnewofprog/' directory")
    print(f"📊 Comprehensive metrics and visualizations generated")

if __name__ == '__main__':
    #main()
    regenerate_visualizations_only()