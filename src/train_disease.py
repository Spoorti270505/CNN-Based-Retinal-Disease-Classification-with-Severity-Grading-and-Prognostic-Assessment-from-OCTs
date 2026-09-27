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
    'EPOCHS': 1,           
    'LR': 1e-4,
    'NUM_WORKERS': 2,
    'DATA_PATH': "data/kermany2018/OCT2017",
    'LABEL_SMOOTHING': 0.1,   
    'WEIGHT_DECAY': 0.01,
    'MODELS_TO_COMPARE': [
        'tf_efficientnetv2_b0',  # EfficientNet
        'resnet50',               # ResNet
        'densenet121',            # DenseNet
        'inception_v3'            # Inception
    ]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(CONFIG['SEED'])

# ---------------------------------------------------------
# 2. DATA PARSING
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
# 3. DATASET CLASS
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
# 4. MODEL ARCHITECTURE
# ---------------------------------------------------------
class DiseaseClassificationNet(nn.Module):
    def __init__(self, num_classes=4, backbone_name='tf_efficientnetv2_b0'):
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
# 5. VISUALIZATION FUNCTIONS
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

def plot_training_history(history, save_dir="resultnew", model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', linewidth=2, label='Train', color='blue')
    axes[0].plot(epochs, history['val_loss'], 's-', linewidth=2, label='Validation', color='red')
    axes[0].set_title(f'Training and Validation Loss - {model_name}', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['val_acc'], 'o-', linewidth=2, color='green')
    axes[1].axhline(max(history['val_acc']), linestyle='--', color='red', alpha=0.7,
                   label=f'Best: {max(history["val_acc"]):.4f}')
    axes[1].set_title(f'Validation Accuracy - {model_name}', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.replace('/', '_')
    plt.savefig(f'{save_dir}/disease_training_history_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved for {model_name}")

def plot_model_comparison(all_resultnew, save_dir="resultnew"):
    """Compare all models comprehensively"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = [r['model_name'] for r in all_resultnew]
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [r['test_acc'] for r in all_resultnew]
    bars = ax1.barh(model_names, accuracies, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Accuracy', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
    ax1.set_xlim([min(accuracies) * 0.95, 1.0])
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax1.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. F1-Score Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = [r['test_f1'] for r in all_resultnew]
    bars = ax2.barh(model_names, f1_scores, color='coral', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('F1-Score', fontweight='bold')
    ax2.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
    ax2.set_xlim([min(f1_scores) * 0.95, 1.0])
    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        ax2.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. AUC-ROC Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    aucs = [r['test_auc'] for r in all_resultnew]
    bars = ax3.barh(model_names, aucs, color='mediumseagreen', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('AUC-ROC', fontweight='bold')
    ax3.set_title('AUC-ROC Comparison', fontweight='bold', fontsize=12)
    ax3.set_xlim([min(aucs) * 0.95, 1.0])
    for i, (bar, val) in enumerate(zip(bars, aucs)):
        ax3.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Parameters vs Accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    params = [r['total_params'] / 1e6 for r in all_resultnew]
    ax4.scatter(params, accuracies, s=200, c='steelblue', edgecolor='black', linewidth=2, alpha=0.7)
    for i, name in enumerate(model_names):
        ax4.annotate(name.replace('tf_', '').replace('_', '\n'), (params[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Parameters vs Accuracy Trade-off', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Time Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    train_times = [r['train_time'] / 60 for r in all_resultnew]
    bars = ax5.barh(model_names, train_times, color='gold', edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Training Time (minutes)', fontweight='bold')
    ax5.set_title('Training Time Comparison', fontweight='bold', fontsize=12)
    for i, (bar, val) in enumerate(zip(bars, train_times)):
        ax5.text(val + 0.5, i, f'{val:.1f}', va='center', fontweight='bold', fontsize=9)
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Combined Metrics
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax6.bar(x - width, accuracies, width, label='Accuracy', color='steelblue', edgecolor='black')
    bars2 = ax6.bar(x, f1_scores, width, label='F1-Score', color='coral', edgecolor='black')
    bars3 = ax6.bar(x + width, aucs, width, label='AUC-ROC', color='mediumseagreen', edgecolor='black')
    
    ax6.set_ylabel('Score', fontweight='bold')
    ax6.set_title('Multi-Metric Comparison', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels([name.replace('tf_', '').replace('_', '\n') for name in model_names], fontsize=8)
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim([min(min(accuracies), min(f1_scores), min(aucs)) * 0.95, 1.0])
    
    plt.suptitle('Disease Classification - Model Comparison', fontsize=16, fontweight='bold')
    plt.savefig(f'{save_dir}/disease_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/disease_model_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model comparison visualization saved")

# ---------------------------------------------------------
# 6. TRAINING ENGINE
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

# ---------------------------------------------------------
# 8. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("="*80)
    print("🚀 DISEASE CLASSIFICATION TRAINING - MULTI-MODEL COMPARISON")
    print("="*80)
    print(f"\n💻 Hardware: Using {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n🔬 Models to compare:")
    for i, model_name in enumerate(CONFIG['MODELS_TO_COMPARE'], 1):
        print(f"   {i}. {model_name}")
    
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
    
    # Store resultnew for all models
    all_resultnew = []
    
    # Train each model
    for model_idx, backbone_name in enumerate(CONFIG['MODELS_TO_COMPARE'], 1):
        print("\n" + "="*80)
        print(f"🏗️ MODEL {model_idx}/{len(CONFIG['MODELS_TO_COMPARE'])}: {backbone_name}")
        print("="*80)
        
        model_start_time = time.time()
        
        # Initialize model
        model = DiseaseClassificationNet(num_classes=len(classes), backbone_name=backbone_name).to(device)
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
        
        print(f"\n🏋️ Training {backbone_name}...")
        
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
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{CONFIG['EPOCHS']}] - {time.time()-epoch_start:.1f}s")
                print(f"    Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                safe_name = backbone_name.replace('/', '_')
                torch.save(model.state_dict(), f'resultnew/best_disease_model_{safe_name}.pth')
        
        print(f"\n✅ Training completed for {backbone_name}!")
        print(f"   Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        # Load best model for testing
        safe_name = backbone_name.replace('/', '_')
        model.load_state_dict(torch.load(f'resultnew/best_disease_model_{safe_name}.pth'))
        
        # Test evaluation
        print(f"\n🔬 Evaluating {backbone_name} on test set...")
        
        _, test_acc, test_f1, y_true, y_pred, y_probs = validate(model, test_loader, criterion, device)
        
        # Calculate AUC
        y_bin = label_binarize(y_true, classes=range(len(classes)))
        test_auc = roc_auc_score(y_bin, y_probs, multi_class='ovr', average='macro')
        
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test F1-Score: {test_f1:.4f}")
        print(f"   Test AUC-ROC: {test_auc:.4f}")
        
        train_time = time.time() - model_start_time
        
        # Save visualizations for this model
        plot_training_history(history, save_dir='resultnew', model_name=backbone_name)
        
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes, save_dir=f'resultnew/{safe_name}')
        
        plot_roc_curves(y_true, y_probs, classes, save_dir=f'resultnew/{safe_name}')
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(f'resultnew/disease_classification_report_{safe_name}.csv')
        
        # Store resultnew
        all_resultnew.append({
            'model_name': backbone_name,
            'total_params': total_params,
            'train_time': train_time,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'history': history
        })
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
        
        print(f"\n⏱️ Total time for {backbone_name}: {train_time/60:.2f} minutes")
    
    # Compare all models
    print("\n" + "="*80)
    print("📊 FINAL MODEL COMPARISON - DISEASE CLASSIFICATION")
    print("="*80)
    
    print(f"\n{'Model':<25} | {'Params (M)':<12} | {'Acc':<8} | {'F1':<8} | {'AUC':<8} | {'Time (min)':<12}")
    print("-" * 100)
    
    for res in all_resultnew:
        print(f"{res['model_name']:<25} | {res['total_params']/1e6:>11.2f} | "
              f"{res['test_acc']:>7.4f} | {res['test_f1']:>7.4f} | "
              f"{res['test_auc']:>7.4f} | {res['train_time']/60:>11.2f}")
    
    print("="*100)
    
    # Find best model
    best_model_idx = np.argmax([r['test_acc'] for r in all_resultnew])
    best_result = all_resultnew[best_model_idx]
    
    print(f"\n🏆 BEST MODEL: {best_result['model_name']}")
    print(f"   Accuracy: {best_result['test_acc']:.4f}")
    print(f"   F1-Score: {best_result['test_f1']:.4f}")
    print(f"   AUC-ROC: {best_result['test_auc']:.4f}")
    
    # Generate comparison visualization
    print("\n🎨 Generating model comparison visualization...")
    plot_model_comparison(all_resultnew, save_dir='resultnew')
    
    # Save comparison table
    comparison_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Parameters (M)': r['total_params'] / 1e6,
        'Training Time (min)': r['train_time'] / 60,
        'Best Epoch': r['best_epoch'],
        'Val Accuracy': r['best_val_acc'],
        'Test Accuracy': r['test_acc'],
        'Test F1-Score': r['test_f1'],
        'Test AUC-ROC': r['test_auc']
    } for r in all_resultnew])
    
    comparison_df.to_csv('resultnew/disease_model_comparison_table.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ DISEASE CLASSIFICATION MULTI-MODEL TRAINING COMPLETED!")
    print("="*80)
    print(f"📂 All resultnew saved in 'resultnew/' directory")
    print(f"📊 {len(CONFIG['MODELS_TO_COMPARE'])} models trained and compared")

if __name__ == '__main__':
    main()
