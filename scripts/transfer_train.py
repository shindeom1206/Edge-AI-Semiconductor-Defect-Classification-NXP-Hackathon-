"""
Enhanced Training Script with Anti-Overfitting Measures
Optimized for hackathon evaluation - tracks all metrics + prevents overfitting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

import config
from transfer_model import MobileNetV2Transfer, count_parameters, estimate_model_size
from transfer_data_loader import get_data_loaders


class EnhancedTrainer:
    """Training with comprehensive metrics tracking and anti-overfitting measures"""
    
    def __init__(self):
        """Initialize trainer with metric tracking"""
        
        print("\n" + "="*70)
        print("🚀 ENHANCED TRANSFER LEARNING - ANTI-OVERFITTING EDITION")
        print("="*70)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        print(f"\n💻 Device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load data
        print("\n📊 Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = get_data_loaders()
        
        # Create model
        print("\n🔧 Creating model...")
        self.model = MobileNetV2Transfer(
            pretrained=config.PRETRAINED,
            num_classes=config.NUM_CLASSES
        ).to(self.device)
        
        total_params = count_parameters(self.model)
        model_size = estimate_model_size(self.model)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Model size: {model_size:.2f} MB")
        
        # Loss function (Weighted Categorical Cross-Entropy)
        self.class_weights = self.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        print(f"\n⚖️  Loss: Weighted Cross-Entropy")
        print(f"   Class weights applied for imbalanced data")
        
        # Training history with enhanced metrics
        self.history = {
            'phase1': {
                'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
                'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
                'lr': [], 'overfit_gap': []
            },
            'phase2': {
                'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
                'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
                'lr': [], 'overfit_gap': []
            }
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Overfitting tracking
        self.overfit_warning_threshold = 0.10  # 10% gap triggers warning
        self.overfit_critical_threshold = 0.15  # 15% gap triggers critical
        
        # Timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n✅ Enhanced Trainer initialized")
        print("📊 Tracking: Loss, Accuracy, Precision, Recall, F1-Score")
        print("🛡️  Anti-overfitting: Weight Decay, Dropout, Gap Monitoring")
    
    
    def calculate_metrics(self, outputs, labels):
        """
        Calculate precision, recall, F1-score
        
        Args:
            outputs: Model predictions (logits)
            labels: True labels
        
        Returns:
            dict: Metrics dictionary
        """
        
        _, predicted = outputs.max(1)
        
        # Move to CPU for sklearn
        predicted_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate metrics (macro average for multi-class)
        precision = precision_score(labels_np, predicted_np, average='macro', zero_division=0)
        recall = recall_score(labels_np, predicted_np, average='macro', zero_division=0)
        f1 = f1_score(labels_np, predicted_np, average='macro', zero_division=0)
        
        # Accuracy
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    
    def train_one_epoch(self, epoch, phase, optimizer):
        """Train for one epoch with full metrics"""
        
        self.model.train()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Phase {phase} Epoch {epoch}")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Collect for metrics
            all_outputs.append(outputs.detach())
            all_labels.append(labels.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        
        metrics = self.calculate_metrics(all_outputs, all_labels)
        avg_loss = running_loss / len(self.train_loader)
        
        return avg_loss, metrics
    
    
    def validate(self, epoch, phase):
        """Validate with full metrics"""
        
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                all_outputs.append(outputs)
                all_labels.append(labels)
        
        # Calculate metrics
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        
        metrics = self.calculate_metrics(all_outputs, all_labels)
        avg_loss = running_loss / len(self.val_loader)
        
        return avg_loss, metrics
    
    
    def check_overfitting(self, train_acc, val_acc, train_f1, val_f1):
        """
        Monitor overfitting gap between train and validation
        
        Returns:
            float: Overfitting gap (0.0 to 1.0)
        """
        
        # Calculate gaps
        acc_gap = train_acc - val_acc
        f1_gap = train_f1 - val_f1
        
        # Use average gap
        avg_gap = (acc_gap + f1_gap) / 2
        
        # Print warnings
        if avg_gap >= self.overfit_critical_threshold:
            print(f"\n🚨 CRITICAL OVERFITTING DETECTED!")
            print(f"   Train-Val Gap: {avg_gap*100:.2f}%")
            print(f"   Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            print(f"   Train F1:  {train_f1*100:.2f}% | Val F1:  {val_f1*100:.2f}%")
        elif avg_gap >= self.overfit_warning_threshold:
            print(f"\n⚠️  Overfitting Warning - Gap: {avg_gap*100:.2f}%")
        
        return avg_gap
    
    
    def check_early_stopping(self, val_loss, val_f1, train_f1):
        """
        Check early stopping based on F1-score AND overfitting gap
        
        Args:
            val_loss: Validation loss
            val_f1: Validation F1-score
            train_f1: Training F1-score
        
        Returns:
            should_stop, improved
        """
        
        improved = False
        
        # Calculate overfitting gap
        overfit_gap = train_f1 - val_f1
        
        # Use F1-score as primary metric, but penalize large gaps
        # If overfitting is severe (>20% gap), don't consider it an improvement
        if val_f1 > self.best_val_f1 and overfit_gap < 0.20:
            print(f"\n✅ New best F1-score: {val_f1:.4f} (prev: {self.best_val_f1:.4f})")
            print(f"   Overfitting gap: {overfit_gap*100:.2f}%")
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            improved = True
        else:
            self.patience_counter += 1
            if overfit_gap >= 0.20:
                print(f"⚠️  No improvement - Overfitting too severe ({overfit_gap*100:.2f}%)")
            else:
                print(f"⚠️  No improvement for {self.patience_counter} epoch(s)")
        
        should_stop = self.patience_counter >= config.EARLY_STOP_PATIENCE
        
        if should_stop:
            print(f"\n🛑 Early stopping triggered after {self.patience_counter} epochs without improvement")
        
        return should_stop, improved
    
    
    def train_phase1(self):
        """Phase 1: Train only classifier (backbone frozen)"""
        
        print("\n" + "="*70)
        print("📚 PHASE 1: TRAINING CLASSIFIER ONLY")
        print("="*70)
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        # Optimizer with WEIGHT DECAY (L2 regularization)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.LEARNING_RATE_PHASE1,
            weight_decay=1e-4  # L2 regularization - prevents overfitting
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE
        )
        
        print(f"\n🎯 Training classifier for {config.PHASE_1_EPOCHS} epochs")
        print(f"   Optimizer: Adam with Weight Decay (1e-4)")
        print(f"   Learning rate: {config.LEARNING_RATE_PHASE1}")
        print(f"   Trainable params: {count_parameters(self.model, trainable_only=True):,}")
        
        for epoch in range(config.PHASE_1_EPOCHS):
            print(f"\n{'─'*70}")
            print(f"EPOCH {epoch + 1}/{config.PHASE_1_EPOCHS}")
            print(f"{'─'*70}")
            
            # Train
            train_loss, train_metrics = self.train_one_epoch(epoch + 1, phase=1, optimizer=optimizer)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch + 1, phase=1)
            
            # Check overfitting
            overfit_gap = self.check_overfitting(
                train_metrics['accuracy'], 
                val_metrics['accuracy'],
                train_metrics['f1'],
                val_metrics['f1']
            )
            
            # Update LR
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['phase1']['train_loss'].append(train_loss)
            self.history['phase1']['train_acc'].append(train_metrics['accuracy'])
            self.history['phase1']['train_precision'].append(train_metrics['precision'])
            self.history['phase1']['train_recall'].append(train_metrics['recall'])
            self.history['phase1']['train_f1'].append(train_metrics['f1'])
            
            self.history['phase1']['val_loss'].append(val_loss)
            self.history['phase1']['val_acc'].append(val_metrics['accuracy'])
            self.history['phase1']['val_precision'].append(val_metrics['precision'])
            self.history['phase1']['val_recall'].append(val_metrics['recall'])
            self.history['phase1']['val_f1'].append(val_metrics['f1'])
            
            self.history['phase1']['lr'].append(current_lr)
            self.history['phase1']['overfit_gap'].append(overfit_gap)
            
            # Print summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Train: Loss={train_loss:.4f} | Acc={train_metrics['accuracy']*100:.2f}% | "
                  f"Prec={train_metrics['precision']*100:.2f}% | "
                  f"Rec={train_metrics['recall']*100:.2f}% | "
                  f"F1={train_metrics['f1']*100:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f} | Acc={val_metrics['accuracy']*100:.2f}% | "
                  f"Prec={val_metrics['precision']*100:.2f}% | "
                  f"Rec={val_metrics['recall']*100:.2f}% | "
                  f"F1={val_metrics['f1']*100:.2f}%")
            print(f"   Gap:   Acc={overfit_gap*100:.2f}% | LR={current_lr:.6f}")
            
            # Early stopping with overfitting check
            should_stop, improved = self.check_early_stopping(
                val_loss, 
                val_metrics['f1'],
                train_metrics['f1']
            )
            if should_stop:
                break
        
        print(f"\n✅ Phase 1 complete!")
        print(f"   Best F1-score: {self.best_val_f1*100:.2f}%")
    
    
    def train_phase2(self):
        """Phase 2: Fine-tune entire network with REDUCED learning rate"""
        
        print("\n" + "="*70)
        print("🔥 PHASE 2: FINE-TUNING ENTIRE NETWORK")
        print("="*70)
        
        # Load best model from phase 1
        if self.best_model_state is not None:
            print("\n🔄 Loading best model from Phase 1...")
            self.model.load_state_dict(self.best_model_state)
        
        # Unfreeze backbone
        self.model.unfreeze_backbone()
        
        # Reset early stopping
        phase1_best_f1 = self.best_val_f1
        self.patience_counter = 0
        
        # Optimizer with STRONGER weight decay for fine-tuning
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE_PHASE2,
            weight_decay=2e-4  # 2x stronger regularization in phase 2
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE
        )
        
        print(f"\n🎯 Fine-tuning for {config.PHASE_2_EPOCHS} epochs")
        print(f"   Optimizer: Adam with Weight Decay (2e-4)")
        print(f"   Learning rate: {config.LEARNING_RATE_PHASE2}")
        print(f"   Trainable params: {count_parameters(self.model, trainable_only=True):,}")
        
        for epoch in range(config.PHASE_2_EPOCHS):
            print(f"\n{'─'*70}")
            print(f"EPOCH {epoch + 1}/{config.PHASE_2_EPOCHS}")
            print(f"{'─'*70}")
            
            # Train
            train_loss, train_metrics = self.train_one_epoch(epoch + 1, phase=2, optimizer=optimizer)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch + 1, phase=2)
            
            # Check overfitting
            overfit_gap = self.check_overfitting(
                train_metrics['accuracy'], 
                val_metrics['accuracy'],
                train_metrics['f1'],
                val_metrics['f1']
            )
            
            # Update LR
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['phase2']['train_loss'].append(train_loss)
            self.history['phase2']['train_acc'].append(train_metrics['accuracy'])
            self.history['phase2']['train_precision'].append(train_metrics['precision'])
            self.history['phase2']['train_recall'].append(train_metrics['recall'])
            self.history['phase2']['train_f1'].append(train_metrics['f1'])
            
            self.history['phase2']['val_loss'].append(val_loss)
            self.history['phase2']['val_acc'].append(val_metrics['accuracy'])
            self.history['phase2']['val_precision'].append(val_metrics['precision'])
            self.history['phase2']['val_recall'].append(val_metrics['recall'])
            self.history['phase2']['val_f1'].append(val_metrics['f1'])
            
            self.history['phase2']['lr'].append(current_lr)
            self.history['phase2']['overfit_gap'].append(overfit_gap)
            
            # Print summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Train: Loss={train_loss:.4f} | Acc={train_metrics['accuracy']*100:.2f}% | "
                  f"Prec={train_metrics['precision']*100:.2f}% | "
                  f"Rec={train_metrics['recall']*100:.2f}% | "
                  f"F1={train_metrics['f1']*100:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f} | Acc={val_metrics['accuracy']*100:.2f}% | "
                  f"Prec={val_metrics['precision']*100:.2f}% | "
                  f"Rec={val_metrics['recall']*100:.2f}% | "
                  f"F1={val_metrics['f1']*100:.2f}%")
            print(f"   Gap:   Acc={overfit_gap*100:.2f}% | LR={current_lr:.6f}")
            
            # Early stopping with overfitting check
            should_stop, improved = self.check_early_stopping(
                val_loss, 
                val_metrics['f1'],
                train_metrics['f1']
            )
            if should_stop:
                break
        
        print(f"\n✅ Phase 2 complete!")
        print(f"   Phase 1 best F1: {phase1_best_f1*100:.2f}%")
        print(f"   Phase 2 best F1: {self.best_val_f1*100:.2f}%")
        print(f"   Improvement: {(self.best_val_f1 - phase1_best_f1)*100:.2f}%")
    
    
    def plot_training_history(self):
        """Plot comprehensive training history with overfitting gap"""
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        
        # Combine phases
        all_train_loss = self.history['phase1']['train_loss'] + self.history['phase2']['train_loss']
        all_val_loss = self.history['phase1']['val_loss'] + self.history['phase2']['val_loss']
        all_train_acc = self.history['phase1']['train_acc'] + self.history['phase2']['train_acc']
        all_val_acc = self.history['phase1']['val_acc'] + self.history['phase2']['val_acc']
        all_train_prec = self.history['phase1']['train_precision'] + self.history['phase2']['train_precision']
        all_val_prec = self.history['phase1']['val_precision'] + self.history['phase2']['val_precision']
        all_train_rec = self.history['phase1']['train_recall'] + self.history['phase2']['train_recall']
        all_val_rec = self.history['phase1']['val_recall'] + self.history['phase2']['val_recall']
        all_train_f1 = self.history['phase1']['train_f1'] + self.history['phase2']['train_f1']
        all_val_f1 = self.history['phase1']['val_f1'] + self.history['phase2']['val_f1']
        all_overfit_gap = self.history['phase1']['overfit_gap'] + self.history['phase2']['overfit_gap']
        
        phase1_len = len(self.history['phase1']['train_loss'])
        
        # Plot 1: Loss
        axes[0, 0].plot(all_train_loss, label='Train', marker='o', markersize=3)
        axes[0, 0].plot(all_val_loss, label='Val', marker='s', markersize=3)
        axes[0, 0].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss (Weighted Cross-Entropy)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot([a*100 for a in all_train_acc], label='Train', marker='o', markersize=3)
        axes[0, 1].plot([a*100 for a in all_val_acc], label='Val', marker='s', markersize=3)
        axes[0, 1].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision
        axes[1, 0].plot([p*100 for p in all_train_prec], label='Train', marker='o', markersize=3)
        axes[1, 0].plot([p*100 for p in all_val_prec], label='Val', marker='s', markersize=3)
        axes[1, 0].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision (%)')
        axes[1, 0].set_title('Precision (Macro Average)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recall
        axes[1, 1].plot([r*100 for r in all_train_rec], label='Train', marker='o', markersize=3)
        axes[1, 1].plot([r*100 for r in all_val_rec], label='Val', marker='s', markersize=3)
        axes[1, 1].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall (%)')
        axes[1, 1].set_title('Recall (Macro Average)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: F1-Score
        axes[2, 0].plot([f*100 for f in all_train_f1], label='Train', marker='o', markersize=3)
        axes[2, 0].plot([f*100 for f in all_val_f1], label='Val', marker='s', markersize=3)
        axes[2, 0].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('F1-Score (%)')
        axes[2, 0].set_title('F1-Score (Macro Average)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: All metrics together
        axes[2, 1].plot([a*100 for a in all_val_acc], label='Accuracy', marker='o', markersize=3)
        axes[2, 1].plot([p*100 for p in all_val_prec], label='Precision', marker='s', markersize=3)
        axes[2, 1].plot([r*100 for r in all_val_rec], label='Recall', marker='^', markersize=3)
        axes[2, 1].plot([f*100 for f in all_val_f1], label='F1-Score', marker='d', markersize=3)
        axes[2, 1].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Metric (%)')
        axes[2, 1].set_title('All Validation Metrics')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 7: OVERFITTING GAP (NEW!)
        axes[3, 0].plot([g*100 for g in all_overfit_gap], label='Train-Val Gap', 
                       marker='o', markersize=3, color='red', linewidth=2)
        axes[3, 0].axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Warning (10%)')
        axes[3, 0].axhline(y=15, color='darkred', linestyle='--', alpha=0.7, label='Critical (15%)')
        axes[3, 0].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[3, 0].set_xlabel('Epoch')
        axes[3, 0].set_ylabel('Gap (%)')
        axes[3, 0].set_title('Overfitting Gap Monitor')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 0].fill_between(range(len(all_overfit_gap)), 
                                [g*100 for g in all_overfit_gap], 
                                alpha=0.3, color='red')
        
        # Plot 8: Train vs Val Accuracy (side by side comparison)
        x = np.arange(len(all_train_acc))
        width = 0.35
        axes[3, 1].bar(x - width/2, [a*100 for a in all_train_acc], 
                      width, label='Train', alpha=0.7)
        axes[3, 1].bar(x + width/2, [a*100 for a in all_val_acc], 
                      width, label='Val', alpha=0.7)
        axes[3, 1].axvline(x=phase1_len - 0.5, color='red', linestyle='--', alpha=0.5)
        axes[3, 1].set_xlabel('Epoch')
        axes[3, 1].set_ylabel('Accuracy (%)')
        axes[3, 1].set_title('Train vs Val Accuracy Comparison')
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_path = os.path.join(config.RESULTS_DIR, f"training_metrics_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved metrics plot: {plot_path}")
        
        plt.close()
    
    
    def save_final_model(self):
        """Save best model with metadata"""
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': {
                'image_size': config.IMAGE_SIZE,
                'num_classes': config.NUM_CLASSES,
                'class_names': config.CLASS_NAMES,
                'backbone': config.BACKBONE
            }
        }
        
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"best_model_enhanced_{self.timestamp}.pth")
        torch.save(checkpoint, model_path)
        print(f"\n💾 Saved enhanced model to: {model_path}")
        
        # Also save as best_model.pth
        best_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"💾 Saved as: {best_path}")
        
        return model_path
    
    
    def train(self):
        """Main training loop"""
        
        start_time = time.time()
        
        # Phase 1
        self.train_phase1()
        
        # Phase 2
        self.train_phase2()
        
        # Save
        self.save_final_model()
        
        # Plot
        self.plot_training_history()
        
        # Summary
        training_time = time.time() - start_time
        
        # Calculate final overfitting gap
        final_train_f1 = self.history['phase2']['train_f1'][-1] if self.history['phase2']['train_f1'] else 0
        final_val_f1 = self.history['phase2']['val_f1'][-1] if self.history['phase2']['val_f1'] else 0
        final_gap = final_train_f1 - final_val_f1
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print(f"\n⏱️  Total time: {training_time / 60:.2f} minutes")
        print(f"🎯 Best F1-score: {self.best_val_f1*100:.2f}%")
        print(f"📉 Best val_loss: {self.best_val_loss:.4f}")
        print(f"🛡️  Final overfitting gap: {final_gap*100:.2f}%")
        
        # Gap health assessment
        if final_gap < 0.05:
            print(f"   ✅ Excellent - No overfitting!")
        elif final_gap < 0.10:
            print(f"   ✅ Good - Minimal overfitting")
        elif final_gap < 0.15:
            print(f"   ⚠️  Moderate overfitting - Consider more regularization")
        else:
            print(f"   🚨 Severe overfitting - Strong regularization needed")
        
        # Final metrics summary
        final_metrics = {
            'training_time_minutes': training_time / 60,
            'best_f1_score': float(self.best_val_f1),
            'best_val_loss': float(self.best_val_loss),
            'final_overfitting_gap': float(final_gap),
            'total_epochs': len(self.history['phase1']['train_loss']) + len(self.history['phase2']['train_loss']),
            'phase1_epochs': len(self.history['phase1']['train_loss']),
            'phase2_epochs': len(self.history['phase2']['train_loss'])
        }
        
        summary_path = os.path.join(config.RESULTS_DIR, f"training_summary_{self.timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        print(f"📝 Saved training summary to: {summary_path}")
        
        return self.model


def main():
    """Main entry point"""
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Train
    trainer = EnhancedTrainer()
    model = trainer.train()
    
    print(f"\n📁 All outputs saved in:")
    print(f"   Models:  {config.MODEL_SAVE_DIR}")
    print(f"   Results: {config.RESULTS_DIR}")
    
    print("\n🏆 Model is ready for evaluation and deployment!")
    print("\n💡 TIPS FOR FURTHER IMPROVEMENT:")
    print("   1. Increase dropout in model (0.5 → 0.6)")
    print("   2. Add more data augmentation")
    print("   3. Use mixup or cutmix augmentation")
    print("   4. Collect more training data if possible")
    print("   5. Try ensemble methods")
    
    return model


if __name__ == "__main__":
    main()
