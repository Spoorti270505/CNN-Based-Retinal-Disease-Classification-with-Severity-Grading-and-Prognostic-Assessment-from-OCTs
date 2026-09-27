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
    'EPOCHS': 1,           
    'LR': 1e-4,
    'NUM_WORKERS': 2,
    'DATA_PATH': "data/kermany2018/OCT2017",
    'WEIGHT_DECAY': 0.01,
    'MODELS_TO_COMPARE': [
        'tf_efficientnetv2_b0',  # EfficientNet
        'resnet50',               # ResNet
        'densenet121',            # DenseNet
        'inception_v3'            # Inception
    ],
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
# 2. PATHOLOGICAL FEATURE EXTRACTION
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
# 3. DATA PARSING
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
# 4. DATASET CLASS
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
# 5. MODEL ARCHITECTURE
# ---------------------------------------------------------
class PrognosisRegressionNet(nn.Module):
    def __init__(self, backbone_name='tf_efficientnetv2_b0'):
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
# 6. VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def plot_predictions_vs_actual(y_true, y_pred, save_dir="resultnewofprog"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Prognosis Score', fontweight='bold')
    axes[0].set_ylabel('Predicted Prognosis Score', fontweight='bold')
    axes[0].set_title('Predicted vs Actual Prognosis Scores', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('True Prognosis Score', fontweight='bold')
    axes[1].set_ylabel('Residuals (Predicted - True)', fontweight='bold')
    axes[1].set_title('Residual Plot', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prognosis_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction plots saved")

def plot_error_distribution(y_true, y_pred, save_dir="resultnewofprog"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error histogram
    errors = y_pred - y_true
    axes[0].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean Error: {np.mean(errors):.4f}')
    axes[0].set_xlabel('Prediction Error', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Error Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Absolute error by true value
    abs_errors = np.abs(errors)
    axes[1].scatter(y_true, abs_errors, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('True Prognosis Score', fontweight='bold')
    axes[1].set_ylabel('Absolute Error', fontweight='bold')
    axes[1].set_title('Absolute Error vs True Value', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prognosis_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Error distribution plots saved")

def plot_training_history(history, save_dir="resultnewofprog", model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', linewidth=2, label='Train', color='blue')
    axes[0].plot(epochs, history['val_loss'], 's-', linewidth=2, label='Validation', color='red')
    axes[0].set_title(f'Training and Validation Loss (MSE) - {model_name}', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(epochs, history['val_mae'], 'o-', linewidth=2, color='green')
    axes[1].axhline(min(history['val_mae']), linestyle='--', color='red', alpha=0.7,
                   label=f'Best: {min(history["val_mae"]):.4f}')
    axes[1].set_title(f'Validation MAE - {model_name}', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(f'{save_dir}/prognosis_training_history_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved for {model_name}")

def plot_model_comparison(all_resultnewofprog, save_dir="resultnewofprog"):
    """Compare all models comprehensively"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = [r['model_name'] for r in all_resultnewofprog]
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. MAE Comparison (lower is better)
    ax1 = fig.add_subplot(gs[0, 0])
    maes = [r['test_mae'] for r in all_resultnewofprog]
    bars = ax1.barh(model_names, maes, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Mean Absolute Error (Lower is Better)', fontweight='bold')
    ax1.set_title('MAE Comparison', fontweight='bold', fontsize=12)
    for i, (bar, val) in enumerate(zip(bars, maes)):
        ax1.text(val + 0.002, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_xaxis()  # Lower is better
    
    # 2. RMSE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    rmses = [r['test_rmse'] for r in all_resultnewofprog]
    bars = ax2.barh(model_names, rmses, color='coral', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Root Mean Squared Error (Lower is Better)', fontweight='bold')
    ax2.set_title('RMSE Comparison', fontweight='bold', fontsize=12)
    for i, (bar, val) in enumerate(zip(bars, rmses)):
        ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_xaxis()  # Lower is better
    
    # 3. R² Comparison (higher is better)
    ax3 = fig.add_subplot(gs[0, 2])
    r2s = [r['test_r2'] for r in all_resultnewofprog]
    bars = ax3.barh(model_names, r2s, color='mediumseagreen', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('R² Score (Higher is Better)', fontweight='bold')
    ax3.set_title('R² Comparison', fontweight='bold', fontsize=12)
    for i, (bar, val) in enumerate(zip(bars, r2s)):
        ax3.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Parameters vs MAE
    ax4 = fig.add_subplot(gs[1, 0])
    params = [r['total_params'] / 1e6 for r in all_resultnewofprog]
    ax4.scatter(params, maes, s=200, c='steelblue', edgecolor='black', linewidth=2, alpha=0.7)
    for i, name in enumerate(model_names):
        ax4.annotate(name.replace('tf_', '').replace('_', '\n'), (params[i], maes[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax4.set_ylabel('MAE (Lower is Better)', fontweight='bold')
    ax4.set_title('Parameters vs Performance Trade-off', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Time Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    train_times = [r['train_time'] / 60 for r in all_resultnewofprog]
    bars = ax5.barh(model_names, train_times, color='gold', edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Training Time (minutes)', fontweight='bold')
    ax5.set_title('Training Time Comparison', fontweight='bold', fontsize=12)
    for i, (bar, val) in enumerate(zip(bars, train_times)):
        ax5.text(val + 0.5, i, f'{val:.1f}', va='center', fontweight='bold', fontsize=9)
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Combined Metrics
    ax6 = fig.add_subplot(gs[1, 2])
    # Normalize metrics for comparison (invert MAE/RMSE so higher is better)
    norm_maes = [1 - (m / max(maes)) for m in maes]
    norm_rmses = [1 - (r / max(rmses)) for r in rmses]
    norm_r2s = r2s
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax6.bar(x - width, norm_maes, width, label='MAE (inverted)', color='steelblue', edgecolor='black')
    bars2 = ax6.bar(x, norm_rmses, width, label='RMSE (inverted)', color='coral', edgecolor='black')
    bars3 = ax6.bar(x + width, norm_r2s, width, label='R²', color='mediumseagreen', edgecolor='black')
    
    ax6.set_ylabel('Normalized Score (Higher is Better)', fontweight='bold')
    ax6.set_title('Multi-Metric Comparison', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels([name.replace('tf_', '').replace('_', '\n') for name in model_names], fontsize=8)
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim([0, 1.1])
    
    plt.suptitle('Prognosis Prediction - Model Comparison', fontsize=16, fontweight='bold')
    plt.savefig(f'{save_dir}/prognosis_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/prognosis_model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model comparison visualization saved")

# ---------------------------------------------------------
# 7. TRAINING ENGINE
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
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    return running_loss / len(loader), mae, mse, rmse, r2, all_labels, all_preds

# ---------------------------------------------------------
# 8. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("="*80)
    print("🚀 PROGNOSIS PREDICTION TRAINING - MULTI-MODEL COMPARISON")
    print("="*80)
    print(f"\n💻 Hardware: Using {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n🔬 Models to compare:")
    for i, model_name in enumerate(CONFIG['MODELS_TO_COMPARE'], 1):
        print(f"   {i}. {model_name}")
    
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
    
    # Create data loaders (no weighted sampling for regression)
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
    
    # Store resultnewofprog for all models
    all_resultnewofprog = []
    
    # Train each model
    for model_idx, backbone_name in enumerate(CONFIG['MODELS_TO_COMPARE'], 1):
        print("\n" + "="*80)
        print(f"🏗️ MODEL {model_idx}/{len(CONFIG['MODELS_TO_COMPARE'])}: {backbone_name}")
        print("="*80)
        
        model_start_time = time.time()
        
        # Initialize model
        model = PrognosisRegressionNet(backbone_name=backbone_name).to(device)
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
        
        print(f"\n🏋️ Training {backbone_name}...")
        
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
                safe_name = backbone_name.replace('/', '_')
                torch.save(model.state_dict(), f'resultnewofprog/best_prognosis_model_{safe_name}.pth')
        
        print(f"\n✅ Training completed for {backbone_name}!")
        print(f"   Best validation MAE: {best_val_mae:.4f} (Epoch {best_epoch})")
        
        # Load best model for testing
        safe_name = backbone_name.replace('/', '_')
        model.load_state_dict(torch.load(f'resultnewofprog/best_prognosis_model_{safe_name}.pth'))
        
        # Test evaluation
        print(f"\n🔬 Evaluating {backbone_name} on test set...")
        
        _, test_mae, test_mse, test_rmse, test_r2, y_true, y_pred = validate(model, test_loader, criterion, device)
        
        print(f"   Test MAE:  {test_mae:.4f}")
        print(f"   Test RMSE: {test_rmse:.4f}")
        print(f"   Test R²:   {test_r2:.4f}")
        
        train_time = time.time() - model_start_time
        
        # Save visualizations for this model
        plot_training_history(history, save_dir='resultnewofprog', model_name=backbone_name)
        plot_predictions_vs_actual(y_true, y_pred, save_dir=f'resultnewofprog/{safe_name}')
        plot_error_distribution(y_true, y_pred, save_dir=f'resultnewofprog/{safe_name}')
        
        # Save resultnewofprog
        resultnewofprog_df = pd.DataFrame({
            'true_prognosis': y_true,
            'predicted_prognosis': y_pred,
            'absolute_error': np.abs(y_pred - y_true)
        })
        resultnewofprog_df.to_csv(f'resultnewofprog/prognosis_predictions_{safe_name}.csv', index=False)
        
        # Store resultnewofprog
        all_resultnewofprog.append({
            'model_name': backbone_name,
            'total_params': total_params,
            'train_time': train_time,
            'best_epoch': best_epoch,
            'best_val_mae': best_val_mae,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'history': history
        })
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        print(f"\n⏱️ Total time for {backbone_name}: {train_time/60:.2f} minutes")
    
    # Compare all models
    print("\n" + "="*80)
    print("📊 FINAL MODEL COMPARISON - PROGNOSIS PREDICTION")
    print("="*80)
    
    print(f"\n{'Model':<25} | {'Params (M)':<12} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8} | {'Time (min)':<12}")
    print("-" * 100)
    
    for res in all_resultnewofprog:
        print(f"{res['model_name']:<25} | {res['total_params']/1e6:>11.2f} | "
              f"{res['test_mae']:>7.4f} | {res['test_rmse']:>7.4f} | "
              f"{res['test_r2']:>7.4f} | {res['train_time']/60:>11.2f}")
    
    print("="*100)
    
    # Find best model (lowest MAE)
    best_model_idx = np.argmin([r['test_mae'] for r in all_resultnewofprog])
    best_result = all_resultnewofprog[best_model_idx]
    
    print(f"\n🏆 BEST MODEL: {best_result['model_name']}")
    print(f"   MAE:  {best_result['test_mae']:.4f}")
    print(f"   RMSE: {best_result['test_rmse']:.4f}")
    print(f"   R²:   {best_result['test_r2']:.4f}")
    
    # Generate comparison visualization
    print("\n🎨 Generating model comparison visualization...")
    plot_model_comparison(all_resultnewofprog, save_dir='resultnewofprog')
    
    # Save comparison table
    comparison_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Parameters (M)': r['total_params'] / 1e6,
        'Training Time (min)': r['train_time'] / 60,
        'Best Epoch': r['best_epoch'],
        'Val MAE': r['best_val_mae'],
        'Test MAE': r['test_mae'],
        'Test RMSE': r['test_rmse'],
        'Test R²': r['test_r2']
    } for r in all_resultnewofprog])
    
    comparison_df.to_csv('resultnewofprog/prognosis_model_comparison_table.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ PROGNOSIS PREDICTION MULTI-MODEL TRAINING COMPLETED!")
    print("="*80)
    print(f"📂 All resultnewofprog saved in 'resultnewofprog/' directory")
    print(f"📊 {len(CONFIG['MODELS_TO_COMPARE'])} models trained and compared")

if __name__ == '__main__':
    main()
