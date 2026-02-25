"""
Enhanced Training Script - Phase 3 FIXED
MobileNetV2 1.0 | 128x128 | 11 Classes | NXP RT1170EVK

CHANGES FROM ORIGINAL:
  - train_phase1(): Added CosineAnnealingLR instead of ReduceLROnPlateau
                   for Phase 1 (works better for frozen-backbone warmup)
  - train_phase2(): Uses AdamW (was Adam) — better weight decay handling
                   Uses CosineAnnealingLR + ReduceLROnPlateau combo
                   Prints "SAVED" marker so you can see when best model updates
  - evaluate_test_set(): Added per-class F1 warning for weak classes
  - Everything else unchanged from your working original
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import json
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report
)

import config
from transfer_model import MobileNetV2Transfer, count_parameters, estimate_model_size
from transfer_data_loader import get_data_loaders


# ============================================================================
class EnhancedTrainer:

    def __init__(self):
        print("\n" + "="*70)
        print("  PHASE 3 ENHANCED TRAINER — FIXED")
        print(f"  MobileNetV2 1.0 | 128x128 | {config.NUM_CLASSES} Classes | NXP RT1170EVK")
        print("="*70)

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"
        )
        print(f"\nDevice: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: Running on CPU — training will take ~3-5 hours")

        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        # Data
        print("\nLoading data...")
        (self.train_loader,
         self.val_loader,
         self.test_loader,
         self.class_weights) = get_data_loaders()

        self.class_names = self.train_loader.dataset.classes
        print(f"\nClasses: {self.class_names}")

        # Model
        print("\nCreating model...")
        self.model = MobileNetV2Transfer(
            pretrained=config.PRETRAINED,
            num_classes=config.NUM_CLASSES,
            dropout_rate_1=config.DROPOUT_RATE_1,
            dropout_rate_2=config.DROPOUT_RATE_2,
            width_mult=config.WIDTH_MULT
        ).to(self.device)

        # Loss with class weights
        self.class_weights = self.class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights if config.USE_CLASS_WEIGHTS else None
        )

        # History
        self.history = {
            "phase1": self._empty_history(),
            "phase2": self._empty_history()
        }

        self.best_val_f1      = 0.0
        self.best_val_loss    = float("inf")
        self.best_model_state = None
        self.patience_counter = 0
        self.timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\nTrainer ready ✅")

    @staticmethod
    def _empty_history():
        return {
            "train_loss": [], "train_acc": [],
            "train_precision": [], "train_recall": [], "train_f1": [],
            "val_loss": [],   "val_acc": [],
            "val_precision": [], "val_recall": [], "val_f1": [],
            "lr": [], "overfit_gap": []
        }

    # ----------------------------------------------------------------- metrics
    def _calc_metrics(self, outputs, labels):
        _, predicted = outputs.max(1)
        pred_np  = predicted.cpu().numpy()
        label_np = labels.cpu().numpy()
        return {
            "accuracy":  predicted.eq(labels).sum().item() / labels.size(0),
            "precision": precision_score(label_np, pred_np, average="macro", zero_division=0),
            "recall":    recall_score   (label_np, pred_np, average="macro", zero_division=0),
            "f1":        f1_score       (label_np, pred_np, average="macro", zero_division=0),
        }

    # ----------------------------------------------------------------- train epoch
    def _train_epoch(self, phase_num, epoch, optimizer):
        self.model.train()
        running_loss = 0.0
        all_outputs, all_labels = [], []

        pbar = tqdm(self.train_loader,
                    desc=f"  Ph{phase_num} Ep{epoch:3d} [train]",
                    ncols=90, leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=config.MAX_GRAD_NORM)
            optimizer.step()
            running_loss += loss.item()
            all_outputs.append(outputs.detach())
            all_labels.append(labels.detach())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        all_outputs = torch.cat(all_outputs)
        all_labels  = torch.cat(all_labels)
        metrics = self._calc_metrics(all_outputs, all_labels)
        metrics["loss"] = running_loss / len(self.train_loader)
        return metrics

    # ----------------------------------------------------------------- val epoch
    def _val_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_outputs, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                running_loss += self.criterion(outputs, labels).item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs)
        all_labels  = torch.cat(all_labels)
        metrics = self._calc_metrics(all_outputs, all_labels)
        metrics["loss"] = running_loss / len(self.val_loader)
        return metrics

    # ----------------------------------------------------------------- record
    def _record(self, phase_key, t, v, lr, gap):
        h = self.history[phase_key]
        h["train_loss"].append(t["loss"]);           h["val_loss"].append(v["loss"])
        h["train_acc"].append(t["accuracy"]);        h["val_acc"].append(v["accuracy"])
        h["train_precision"].append(t["precision"]); h["val_precision"].append(v["precision"])
        h["train_recall"].append(t["recall"]);       h["val_recall"].append(v["recall"])
        h["train_f1"].append(t["f1"]);               h["val_f1"].append(v["f1"])
        h["lr"].append(lr);                          h["overfit_gap"].append(gap)

    def _print_epoch(self, phase, epoch, total, t, v, lr, gap, saved=""):
        flag = ""
        if   gap > config.OVERFIT_CRITICAL_THRESHOLD: flag = "  ❌ CRITICAL"
        elif gap > config.OVERFIT_WARNING_THRESHOLD:  flag = "  ⚠️  warning"
        print(
            f"  Ph{phase} [{epoch:>3}/{total}] "
            f"Loss {t['loss']:.4f}/{v['loss']:.4f}  "
            f"Acc {t['accuracy']*100:.1f}%/{v['accuracy']*100:.1f}%  "
            f"F1 {t['f1']*100:.1f}%/{v['f1']*100:.1f}%  "
            f"LR {lr:.2e}  Gap {gap*100:.1f}%{flag}{saved}"
        )

    # ----------------------------------------------------------------- early stop
    def _check_early_stop(self, val_f1, val_loss):
        saved = ""
        if val_f1 > self.best_val_f1 + 1e-4:
            self.best_val_f1      = val_f1
            self.best_val_loss    = val_loss
            self.best_model_state = {k: v.clone()
                for k, v in self.model.state_dict().items()}
            self.patience_counter = 0
            saved = "  ✅ SAVED"
        else:
            self.patience_counter += 1

        stop = self.patience_counter >= config.EARLY_STOP_PATIENCE
        if stop:
            print(f"\n  ⏹  Early stopping: no improvement for "
                  f"{config.EARLY_STOP_PATIENCE} epochs")
        return stop, saved

    # ============================================================ PHASE 1
    def train_phase1(self):
        """
        Phase 1: Backbone FROZEN — only train the classifier head.
        FIX: Using CosineAnnealingLR instead of ReduceLROnPlateau.
             Cosine annealing works better for short frozen-backbone training
             because it proactively decays LR on a schedule rather than
             waiting for plateau signals.
        """
        print("\n" + "="*70)
        print(f"  PHASE 1: Classifier only ({config.PHASE_1_EPOCHS} epochs)")
        print(f"  Backbone FROZEN — training classifier head at LR={config.LEARNING_RATE_PHASE1}")
        print("="*70)

        self.model.freeze_backbone()
        self.patience_counter = 0

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.LEARNING_RATE_PHASE1,
            weight_decay=config.WEIGHT_DECAY_PHASE1
        )

        # FIX: CosineAnnealingLR — smoothly decays LR over Phase 1 epochs
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.PHASE_1_EPOCHS,
            eta_min=1e-5
        )

        for epoch in range(1, config.PHASE_1_EPOCHS + 1):
            t = self._train_epoch(1, epoch, optimizer)
            v = self._val_epoch()

            lr  = optimizer.param_groups[0]["lr"]
            gap = t["f1"] - v["f1"]
            self._record("phase1", t, v, lr, gap)

            stop, saved = self._check_early_stop(v["f1"], v["loss"])
            self._print_epoch(1, epoch, config.PHASE_1_EPOCHS, t, v, lr, gap, saved)

            scheduler.step()

            if stop:
                break

        print(f"\n  Phase 1 complete — Best val F1: {self.best_val_f1*100:.2f}%")

    # ============================================================ PHASE 2
    def train_phase2(self):
        """
        Phase 2: Full fine-tuning — ALL layers trainable.
        FIX: Using AdamW (better weight decay), lower LR=5e-5 (was 1e-5, now 5e-5),
             CosineAnnealingLR as primary + ReduceLROnPlateau as backup scheduler.
             Reset patience counter for a fresh start.
        """
        print("\n" + "="*70)
        print(f"  PHASE 2: Full fine-tuning ({config.PHASE_2_EPOCHS} epochs)")
        print(f"  All layers UNFROZEN — LR={config.LEARNING_RATE_PHASE2}")
        print("="*70)

        self.model.unfreeze_backbone()

        # FIX: Reset patience so Phase 2 gets a full fresh run
        self.patience_counter = 0

        # FIX: AdamW handles weight decay correctly (decoupled from gradient)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE_PHASE2,
            weight_decay=config.WEIGHT_DECAY_PHASE2
        )

        # FIX: CosineAnnealingLR as primary scheduler
        # Smoothly decays LR over all Phase 2 epochs — avoids abrupt drops
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.PHASE_2_EPOCHS,
            eta_min=config.LR_MIN
        )

        # ReduceLROnPlateau as backup — kicks in if val_loss truly plateaus
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.LR_MIN,
        )

        for epoch in range(1, config.PHASE_2_EPOCHS + 1):
            t = self._train_epoch(2, epoch, optimizer)
            v = self._val_epoch()

            lr  = optimizer.param_groups[0]["lr"]
            gap = t["f1"] - v["f1"]
            self._record("phase2", t, v, lr, gap)

            stop, saved = self._check_early_stop(v["f1"], v["loss"])
            self._print_epoch(2, epoch, config.PHASE_2_EPOCHS, t, v, lr, gap, saved)

            # Step both schedulers
            cosine_scheduler.step()
            plateau_scheduler.step(v["loss"])

            if stop:
                break

        print(f"\n  Phase 2 complete — Best val F1: {self.best_val_f1*100:.2f}%")

    # ============================================================ EVALUATE
    def evaluate_test_set(self):
        print("\n" + "="*70)
        print("  TEST SET EVALUATION")
        print("="*70)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("  (Using best saved model weights)")

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc  = (all_preds == all_labels).mean()
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec  = recall_score   (all_labels, all_preds, average="macro", zero_division=0)
        f1   = f1_score       (all_labels, all_preds, average="macro", zero_division=0)

        print(f"\n  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {prec*100:.2f}%  (macro)")
        print(f"  Recall    : {rec*100:.2f}%  (macro)")
        print(f"  F1        : {f1*100:.2f}%  (macro)")

        print("\nPer-class report:")
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            zero_division=0
        )
        print(report)

        # FIX: Warn about weak classes so you know what to investigate
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        weak = [(self.class_names[i], f"{per_class_f1[i]*100:.1f}%")
                for i in range(len(self.class_names)) if per_class_f1[i] < 0.40]
        if weak:
            print(f"\n  ⚠️  Classes below 40% F1 (may need more data):")
            for name, score in weak:
                print(f"       {name}: {score}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # ============================================================ PLOT
    def plot_training_history(self):
        if not config.PLOT_TRAINING_CURVES:
            return

        h1 = self.history["phase1"]
        h2 = self.history["phase2"]
        n1 = len(h1["train_loss"])

        all_tl  = h1["train_loss"]      + h2["train_loss"]
        all_vl  = h1["val_loss"]        + h2["val_loss"]
        all_ta  = h1["train_acc"]       + h2["train_acc"]
        all_va  = h1["val_acc"]         + h2["val_acc"]
        all_tp  = h1["train_precision"] + h2["train_precision"]
        all_vp  = h1["val_precision"]   + h2["val_precision"]
        all_tr  = h1["train_recall"]    + h2["train_recall"]
        all_vr  = h1["val_recall"]      + h2["val_recall"]
        all_tf  = h1["train_f1"]        + h2["train_f1"]
        all_vf  = h1["val_f1"]          + h2["val_f1"]
        all_gap = h1["overfit_gap"]     + h2["overfit_gap"]

        if not all_tl:
            return

        ep  = list(range(1, len(all_tl) + 1))
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(
            f"Training — MobileNetV2 | {config.NUM_CLASSES} Classes | {self.timestamp}",
            fontsize=14
        )

        def vline(ax):
            if n1 > 0:
                ax.axvline(x=n1+0.5, color="gray", linestyle="--",
                           alpha=0.5, label="Ph1→Ph2")

        axes[0,0].plot(ep, all_tl, label="Train"); axes[0,0].plot(ep, all_vl, label="Val")
        vline(axes[0,0]); axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

        axes[0,1].plot(ep, [a*100 for a in all_ta], label="Train")
        axes[0,1].plot(ep, [a*100 for a in all_va], label="Val")
        vline(axes[0,1]); axes[0,1].set_title("Accuracy (%)"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

        axes[1,0].plot(ep, [p*100 for p in all_tp], label="Train")
        axes[1,0].plot(ep, [p*100 for p in all_vp], label="Val")
        vline(axes[1,0]); axes[1,0].set_title("Precision (%)"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

        axes[1,1].plot(ep, [r*100 for r in all_tr], label="Train")
        axes[1,1].plot(ep, [r*100 for r in all_vr], label="Val")
        vline(axes[1,1]); axes[1,1].set_title("Recall (%)"); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

        axes[2,0].plot(ep, [f*100 for f in all_tf], label="Train")
        axes[2,0].plot(ep, [f*100 for f in all_vf], label="Val")
        vline(axes[2,0]); axes[2,0].set_title("F1 (%)"); axes[2,0].legend(); axes[2,0].grid(alpha=0.3)

        axes[2,1].plot(ep, [a*100 for a in all_va], label="Acc")
        axes[2,1].plot(ep, [p*100 for p in all_vp], label="Prec")
        axes[2,1].plot(ep, [r*100 for r in all_vr], label="Rec")
        axes[2,1].plot(ep, [f*100 for f in all_vf], label="F1")
        vline(axes[2,1]); axes[2,1].set_title("All Val Metrics"); axes[2,1].legend(); axes[2,1].grid(alpha=0.3)

        axes[3,0].plot(ep, [g*100 for g in all_gap], color="red", label="Gap")
        axes[3,0].axhline(y=config.OVERFIT_WARNING_THRESHOLD*100,  color="orange",  linestyle="--", label="Warning")
        axes[3,0].axhline(y=config.OVERFIT_CRITICAL_THRESHOLD*100, color="darkred", linestyle="--", label="Critical")
        vline(axes[3,0]); axes[3,0].set_title("Overfitting Gap (%)"); axes[3,0].legend(); axes[3,0].grid(alpha=0.3)

        x, w = np.arange(len(all_ta)), 0.35
        axes[3,1].bar(x-w/2, [a*100 for a in all_ta], w, label="Train", alpha=0.7)
        axes[3,1].bar(x+w/2, [a*100 for a in all_va], w, label="Val",   alpha=0.7)
        vline(axes[3,1]); axes[3,1].set_title("Train vs Val Accuracy"); axes[3,1].legend(); axes[3,1].grid(alpha=0.3, axis="y")

        for ax in axes.flat:
            ax.set_xlabel("Epoch")

        plt.tight_layout()
        path = os.path.join(config.RESULTS_DIR, f"training_metrics_{self.timestamp}.png")
        plt.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved: {path}")

    # ============================================================ SAVE
    def save_final_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "history":          self.history,
            "best_val_loss":    self.best_val_loss,
            "best_val_f1":      self.best_val_f1,
            "config": {
                "image_size":  config.IMAGE_SIZE,
                "num_classes": config.NUM_CLASSES,
                "class_names": self.class_names,
                "backbone":    config.BACKBONE,
                "width_mult":  config.WIDTH_MULT,
            }
        }

        ts_path   = os.path.join(config.MODEL_SAVE_DIR,
                                 f"best_model_{self.timestamp}.pth")
        best_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
        torch.save(checkpoint, ts_path)
        torch.save(checkpoint, best_path)
        print(f"\n  ✅ Model saved: {ts_path}")
        print(f"  ✅ Model saved: {best_path}")
        return best_path

    # ============================================================ TRAIN
    def train(self):
        t0 = time.time()
        self.train_phase1()
        self.train_phase2()
        self.save_final_model()
        self.plot_training_history()
        test_metrics = self.evaluate_test_set()
        elapsed   = time.time() - t0
        final_gap = (
            (self.history["phase2"]["train_f1"][-1]
             - self.history["phase2"]["val_f1"][-1])
            if self.history["phase2"]["train_f1"] else 0.0
        )

        print("\n" + "="*70)
        print("  TRAINING COMPLETE")
        print("="*70)
        print(f"  Total time      : {elapsed/60:.1f} min")
        print(f"  Best val F1     : {self.best_val_f1*100:.2f}%")
        print(f"  Best val loss   : {self.best_val_loss:.4f}")
        print(f"  Overfit gap     : {final_gap*100:.2f}%")
        print(f"  Test accuracy   : {test_metrics['accuracy']*100:.2f}%")
        print(f"  Test F1 (macro) : {test_metrics['f1']*100:.2f}%")

        if   final_gap < 0.05: print("  Overfit health: EXCELLENT")
        elif final_gap < 0.10: print("  Overfit health: GOOD")
        elif final_gap < 0.15: print("  Overfit health: WARNING")
        else:                  print("  Overfit health: CRITICAL")

        summary = {
            "training_time_minutes": elapsed / 60,
            "best_val_f1":           float(self.best_val_f1),
            "best_val_loss":         float(self.best_val_loss),
            "final_overfit_gap":     float(final_gap),
            "test_accuracy":         float(test_metrics["accuracy"]),
            "test_precision_macro":  float(test_metrics["precision"]),
            "test_recall_macro":     float(test_metrics["recall"]),
            "test_f1_macro":         float(test_metrics["f1"]),
            "total_epochs": (len(self.history["phase1"]["train_loss"])
                           + len(self.history["phase2"]["train_loss"])),
            "model": {
                "backbone":    config.BACKBONE,
                "num_classes": config.NUM_CLASSES,
                "class_names": self.class_names,
                "image_size":  list(config.IMAGE_SIZE),
            }
        }
        summary_path = os.path.join(
            config.RESULTS_DIR, f"training_summary_{self.timestamp}.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"  Summary saved   : {summary_path}")

        return self.model


# ============================================================================
def main():
    config.print_config()
    for d in [config.MODEL_SAVE_DIR, config.RESULTS_DIR, config.LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    trainer = EnhancedTrainer()
    trainer.train()

    print(f"\n  Models  → {config.MODEL_SAVE_DIR}")
    print(f"  Results → {config.RESULTS_DIR}")
    print("\n  ✅ Ready for TFLite export and MCUXpresso deployment!")


if __name__ == "__main__":
    main()