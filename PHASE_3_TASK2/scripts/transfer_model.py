"""
Transfer Learning Model - Phase 3 FINAL
MobileNetV2 1.0 | 128x128 | 11 Classes | NXP RT1170EVK
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Transfer(nn.Module):

    def __init__(self, pretrained=True, num_classes=11,
                 dropout_rate_1=0.4, dropout_rate_2=0.2,
                 width_mult=1.0):
        super(MobileNetV2Transfer, self).__init__()

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        num_features = self.backbone.classifier[1].in_features  # 1280

        # Classifier: 1280 -> 256 -> num_classes
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate_1),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate_2),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

        print(f"[Model] MobileNetV2 1.0 (full ImageNet pretrained)")
        print(f"[Model] Classifier: {num_features} -> 256 -> {num_classes}")
        print(f"[Model] Dropout: {dropout_rate_1} / {dropout_rate_2}")
        print(f"[Model] Params: {count_parameters(self):,} | "
              f"Size: {estimate_model_size(self):.2f} MB")

    def _initialize_weights(self):
        for m in self.backbone.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        print("[Freeze] Backbone frozen — classifier only")
        print(f"[Freeze] Trainable: {count_parameters(self, trainable_only=True):,}")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Freeze] Full network unfrozen")
        print(f"[Freeze] Trainable: {count_parameters(self, trainable_only=True):,}")

    def unfreeze_last_n_layers(self, n=3):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for layer in list(self.backbone.features.children())[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        print(f"[Freeze] Last {n} layers + classifier unfrozen")
        print(f"[Freeze] Trainable: {count_parameters(self, trainable_only=True):,}")


# ============================================================================
def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_model_size(model):
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2


if __name__ == "__main__":
    import config
    print(f"Testing MobileNetV2 with {config.NUM_CLASSES} classes...")
    model = MobileNetV2Transfer(pretrained=True, num_classes=config.NUM_CLASSES)
    dummy = torch.randn(2, 3, 128, 128)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, config.NUM_CLASSES), \
        f"Expected (2,{config.NUM_CLASSES}), got {out.shape}"
    print("✅ PASSED")