import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

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
from collections import Counter, defaultdict
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            f1_score, roc_curve, auc, accuracy_score,
                            precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
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
    'LABEL_SMOOTHING': 0.1,   
    'WEIGHT_DECAY': 0.01,
    'BACKBONE': 'densenet121'  # Only DenseNet121
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(CONFIG['SEED'])

# ---------------------------------------------------------
# 2. ADVANCED METRICS CALCULATIONS
# ---------------------------------------------------------

def calculate_npv_and_advanced_metrics(y_true, y_pred, y_probs, class_names):
    """Calculate NPV, MCC, Likelihood Ratios, and Confidence Intervals"""
    n_classes = len(class_names)
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        # Binary classification for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # Calculate TP, TN, FP, FN
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        # Likelihood Ratios
        lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
        lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
        
        # F1 Score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Bootstrap Confidence Intervals (95%)
        n_bootstrap = 1000
        bootstrap_acc = []
        bootstrap_sens = []
        bootstrap_spec = []
        bootstrap_f1 = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            y_true_bin_boot = (y_true_boot == i).astype(int)
            y_pred_bin_boot = (y_pred_boot == i).astype(int)
            
            tp_b = np.sum((y_true_bin_boot == 1) & (y_pred_bin_boot == 1))
            tn_b = np.sum((y_true_bin_boot == 0) & (y_pred_bin_boot == 0))
            fp_b = np.sum((y_true_bin_boot == 0) & (y_pred_bin_boot == 1))
            fn_b = np.sum((y_true_bin_boot == 1) & (y_pred_bin_boot == 0))
            
            acc_b = (tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b) if (tp_b + tn_b + fp_b + fn_b) > 0 else 0
            sens_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
            spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0
            prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
            f1_b = 2 * (prec_b * sens_b) / (prec_b + sens_b) if (prec_b + sens_b) > 0 else 0
            
            bootstrap_acc.append(acc_b)
            bootstrap_sens.append(sens_b)
            bootstrap_spec.append(spec_b)
            bootstrap_f1.append(f1_b)
        
        # Calculate 95% CI
        ci_acc = np.percentile(bootstrap_acc, [2.5, 97.5])
        ci_sens = np.percentile(bootstrap_sens, [2.5, 97.5])
        ci_spec = np.percentile(bootstrap_spec, [2.5, 97.5])
        ci_f1 = np.percentile(bootstrap_f1, [2.5, 97.5])
        
        metrics[class_name] = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'mcc': mcc,
            'lr_positive': lr_positive,
            'lr_negative': lr_negative,
            'f1': f1,
            'ci_accuracy': ci_acc,
            'ci_sensitivity': ci_sens,
            'ci_specificity': ci_spec,
            'ci_f1': ci_f1
        }
    
    return metrics

def calculate_ece(probs, labels, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_mce(probs, labels, n_bins=10):
    """Calculate Maximum Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce

def calculate_brier_score(probs, labels, n_classes):
    """Calculate Brier Score"""
    one_hot_labels = np.eye(n_classes)[labels]
    brier = np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1))
    return brier

# ---------------------------------------------------------
# 3. DATA PARSING
# ---------------------------------------------------------
def parse_dataset_with_patients(base_path):
    """Parse dataset with patient-aware splitting"""
    print(f"\n📂 Scanning dataset at {base_path}...")
    disease_classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    disease_to_idx = {d: i for i, d in enumerate(disease_classes)}
    all_records = []
    
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
                
                all_records.append({
                    'path': img_path,
                    'patient_id': patient_id,
                    'disease_label': disease_to_idx[disease],
                    'split': split
                })
                
    df = pd.DataFrame(all_records)
    print(f"✅ Loaded {len(df)} images. Unique Patients: {df['patient_id'].nunique()}")
    print(f"   Train images: {len(df[df['split']=='train'])}")
    print(f"   Val images: {len(df[df['split']=='val'])}")
            
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
            
        label = torch.tensor(row['disease_label'], dtype=torch.long)
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
class DiseaseClassificationNet(nn.Module):
    def __init__(self, num_classes=4, backbone_name='densenet121'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits

# ---------------------------------------------------------
# 6. VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def plot_confusion_matrix(cm, class_names, save_dir="resultnew"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.array([[f'{count}\n({pct:.1f}%)' 
                      for count, pct in zip(row_counts, row_pcts)]
                     for row_counts, row_pcts in zip(cm, cm_pct)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Disease Classification Confusion Matrix', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/disease_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved")

def plot_roc_curves(y_true, y_probs, class_names, save_dir="resultnew"):
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(range(n_classes))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.5, label=f'{class_name} (AUC = {roc_auc:.3f})', color=color)
    
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, linewidth=3, linestyle='--',
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})', color='navy')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title('ROC Curves - Disease Classification', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/disease_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved")

def plot_precision_recall_curves(y_true, y_probs, class_names, save_dir="resultnew"):
    """Precision-Recall curves for all classes"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(range(n_classes))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        avg_precision = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.plot(recall, precision, linewidth=2.5, 
                label=f'{class_name} (AP = {avg_precision:.3f})', color=color)
    
    ax.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax.set_title('Precision-Recall Curves - Disease Classification', fontweight='bold', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/precision_recall_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curves saved")

def plot_performance_metrics_bar(metrics_dict, class_names, save_dir="resultnew"):
    """Bar chart showing Accuracy, Sensitivity, Specificity, NPV per class"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(class_names))
    width = 0.18
    
    acc_vals = [metrics_dict[cls]['accuracy'] for cls in class_names]
    sens_vals = [metrics_dict[cls]['sensitivity'] for cls in class_names]
    spec_vals = [metrics_dict[cls]['specificity'] for cls in class_names]
    ppv_vals = [metrics_dict[cls]['precision'] for cls in class_names]
    npv_vals = [metrics_dict[cls]['npv'] for cls in class_names]
    
    bars1 = ax.bar(x - 2*width, acc_vals, width, label='Accuracy', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x - width, sens_vals, width, label='Sensitivity', color='coral', edgecolor='black')
    bars3 = ax.bar(x, spec_vals, width, label='Specificity', color='mediumseagreen', edgecolor='black')
    bars4 = ax.bar(x + width, ppv_vals, width, label='PPV', color='gold', edgecolor='black')
    bars5 = ax.bar(x + 2*width, npv_vals, width, label='NPV', color='orchid', edgecolor='black')
    
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Performance Metrics by Disease Class', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_metrics_bar.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/performance_metrics_bar.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance metrics bar chart saved")

def plot_calibration_analysis(y_true, y_probs, save_dir="resultnew"):
    """Generate calibration curve and confidence distribution"""
    os.makedirs(save_dir, exist_ok=True)
    
    prob_pos = y_probs[:, 0]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true == 0, prob_pos, n_bins=10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(mean_predicted_value, fraction_of_positives, 's-', 
                linewidth=2, markersize=8, label='Model')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    axes[0].set_xlabel('Mean Predicted Probability', fontweight='bold')
    axes[0].set_ylabel('Fraction of Positives', fontweight='bold')
    axes[0].set_title('Calibration Curve', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    confidences = np.max(y_probs, axis=1)
    axes[1].hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(np.mean(confidences), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    axes[1].set_xlabel('Confidence', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Prediction Confidence Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/calibration_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Calibration analysis saved")

def plot_tsne(features, labels, class_names, save_dir="resultnew"):
    """t-SNE visualization of learned features"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("   Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[color], label=class_name, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=12)
    ax.set_title('t-SNE Visualization of Learned Features', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/tsne_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE visualization saved")

def plot_training_history(history, save_dir="resultnew"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', linewidth=2, label='Train', color='blue')
    axes[0].plot(epochs, history['val_loss'], 's-', linewidth=2, label='Validation', color='red')
    axes[0].set_title('Training and Validation Loss - DenseNet121', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['val_acc'], 'o-', linewidth=2, color='green')
    axes[1].axhline(max(history['val_acc']), linestyle='--', color='red', alpha=0.7,
                   label=f'Best: {max(history["val_acc"]):.4f}')
    axes[1].set_title('Validation Accuracy - DenseNet121', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/disease_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved")

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
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return running_loss / len(loader), accuracy, f1, all_labels, all_preds, all_probs

def extract_features(model, loader, device):
    """Extract features for t-SNE visualization"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            _, features = model(images, return_features=True)
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    return features, labels

# ---------------------------------------------------------
# 8. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("="*80)
    print("🚀 DENSENET121 DISEASE CLASSIFICATION WITH ADVANCED METRICS")
    print("="*80)
    print(f"\n💻 Hardware: Using {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and parse data
    df, classes = parse_dataset_with_patients(CONFIG['DATA_PATH'])
    
    # Patient-aware splitting
    print("\n" + "="*80)
    print("📂 DATA SPLITTING (Patient-Aware)")
    print("="*80)
    
    train_data = df[df['split'] == 'train'].reset_index(drop=True)
    all_patients = train_data['patient_id'].unique()
    
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
    class_counts = train_df['disease_label'].value_counts().sort_index()
    weights = 1. / torch.tensor(class_counts.values, dtype=torch.float)
    sample_weights = weights[train_df['disease_label'].values]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_df))
    
    train_loader = DataLoader(
        OCTDataset(train_df, train_transform),
        batch_size=CONFIG['BATCH_SIZE'],
        sampler=sampler,
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
    
    model = DiseaseClassificationNet(num_classes=len(classes), backbone_name=CONFIG['BACKBONE']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['LABEL_SMOOTHING'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'])
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    best_val_acc = 0
    best_epoch = 0
    
    print(f"\n🏋️ Training DenseNet121...")
    
    for epoch in range(CONFIG['EPOCHS']):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        
        # Validate
        val_loss, val_acc, val_f1, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step()
        
        
        print(f"  Epoch [{epoch+1}/{CONFIG['EPOCHS']}] - {time.time()-epoch_start:.1f}s")
        print(f"    Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'resultnew/best_disease_model_densenet121.pth')
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Load best model for testing
    model.load_state_dict(torch.load('resultnew/best_disease_model_densenet121.pth'))
    
    # Test evaluation
    print(f"\n🔬 Evaluating DenseNet121 on test set...")
    
    _, test_acc, test_f1, y_true, y_pred, y_probs = validate(model, test_loader, criterion, device)
    
    # Calculate AUC
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    test_auc = roc_auc_score(y_bin, y_probs, multi_class='ovr', average='macro')
    
    # Calculate advanced metrics
    print(f"\n📊 Computing Advanced Metrics...")
    detailed_metrics = calculate_npv_and_advanced_metrics(np.array(y_true), np.array(y_pred), y_probs, classes)
    
    # Calculate calibration metrics
    ece = calculate_ece(y_probs, np.array(y_true))
    mce = calculate_mce(y_probs, np.array(y_true))
    brier = calculate_brier_score(y_probs, np.array(y_true), len(classes))
    
    print(f"\n📊 Test Set Performance:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1-Score: {test_f1:.4f}")
    print(f"   Test AUC-ROC: {test_auc:.4f}")
    print(f"   ECE (Expected Calibration Error): {ece:.4f}")
    print(f"   MCE (Maximum Calibration Error): {mce:.4f}")
    print(f"   Brier Score: {brier:.4f}")
    
    # Print per-class metrics
    print(f"\n📊 Per-Class Detailed Metrics:")
    print(f"{'Class':<10} | {'Acc':<8} | {'Sens':<8} | {'Spec':<8} | {'PPV':<8} | {'NPV':<8} | {'MCC':<8} | {'F1':<8}")
    print("-" * 90)
    for cls in classes:
        m = detailed_metrics[cls]
        print(f"{cls:<10} | {m['accuracy']:.4f}   | {m['sensitivity']:.4f}   | {m['specificity']:.4f}   | "
              f"{m['precision']:.4f}   | {m['npv']:.4f}   | {m['mcc']:.4f}   | {m['f1']:.4f}")
    
    # Print confidence intervals
    print(f"\n📊 95% Confidence Intervals:")
    for cls in classes:
        m = detailed_metrics[cls]
        print(f"\n{cls}:")
        print(f"   Accuracy:    [{m['ci_accuracy'][0]:.4f}, {m['ci_accuracy'][1]:.4f}]")
        print(f"   Sensitivity: [{m['ci_sensitivity'][0]:.4f}, {m['ci_sensitivity'][1]:.4f}]")
        print(f"   Specificity: [{m['ci_specificity'][0]:.4f}, {m['ci_specificity'][1]:.4f}]")
        print(f"   F1-Score:    [{m['ci_f1'][0]:.4f}, {m['ci_f1'][1]:.4f}]")
    
    # Generate visualizations
    print(f"\n🎨 Generating Comprehensive Visualizations...")
    
    # 1. Training history
    plot_training_history(history, save_dir='resultnew')
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, save_dir='resultnew')
    
    # 3. ROC curves
    plot_roc_curves(y_true, y_probs, classes, save_dir='resultnew')
    
    # 4. Precision-Recall curves
    plot_precision_recall_curves(y_true, y_probs, classes, save_dir='resultnew')
    
    # 5. Performance metrics bar chart
    plot_performance_metrics_bar(detailed_metrics, classes, save_dir='resultnew')
    
    # 6. Calibration analysis
    plot_calibration_analysis(np.array(y_true), y_probs, save_dir='resultnew')
    
    # 7. t-SNE visualization
    print("   Extracting features for t-SNE...")
    features, labels = extract_features(model, test_loader, device)
    plot_tsne(features, labels, classes, save_dir='resultnew')
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('resultnew/disease_classification_report_densenet121.csv')
    
    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(detailed_metrics).T
    metrics_df.to_csv('resultnew/detailed_metrics_densenet121.csv')
    
    # Save summary metrics
    summary_metrics = {
        'Model': 'DenseNet121',
        'Total Parameters': total_params,
        'Test Accuracy': test_acc,
        'Test F1-Score': test_f1,
        'Test AUC-ROC': test_auc,
        'ECE': ece,
        'MCE': mce,
        'Brier Score': brier,
        'Average Sensitivity': np.mean([detailed_metrics[c]['sensitivity'] for c in classes]),
        'Average Specificity': np.mean([detailed_metrics[c]['specificity'] for c in classes]),
        'Average NPV': np.mean([detailed_metrics[c]['npv'] for c in classes]),
        'Average MCC': np.mean([detailed_metrics[c]['mcc'] for c in classes])
    }
    
    pd.DataFrame([summary_metrics]).to_csv('resultnew/summary_metrics_densenet121.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ DENSENET121 TRAINING WITH ADVANCED METRICS COMPLETED!")
    print("="*80)
    print(f"📂 All results saved in 'resultnew/' directory")
    print(f"📊 Comprehensive metrics and visualizations generated")

if __name__ == '__main__':
    main()