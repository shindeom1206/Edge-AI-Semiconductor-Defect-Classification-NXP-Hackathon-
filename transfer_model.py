"""
Transfer Learning Model with Enhanced Regularization
MobileNetV2 backbone with strong dropout to prevent overfitting
"""

import torch
import torch.nn as nn
from torchvision import models


class MobileNetV2Transfer(nn.Module):
    """
    MobileNetV2 Transfer Learning Model with Anti-Overfitting Measures
    
    Features:
    - Pretrained MobileNetV2 backbone
    - Multiple dropout layers
    - Flexible classifier architecture
    - Freeze/unfreeze capabilities
    """
    
    def __init__(self, pretrained=True, num_classes=3, dropout_rate_1=0.6, dropout_rate_2=0.4):
        """
        Initialize the model
        
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            num_classes (int): Number of output classes
            dropout_rate_1 (float): First dropout rate (0.6 recommended)
            dropout_rate_2 (float): Second dropout rate (0.4 recommended)
        """
        super(MobileNetV2Transfer, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head (STRONG REGULARIZATION)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate_1),           # First dropout - aggressive (0.6)
            nn.Linear(num_features, 512),           # Intermediate layer
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),                    # Batch normalization
            nn.Dropout(p=dropout_rate_2),           # Second dropout (0.4)
            nn.Linear(512, 256),                    # Another intermediate layer
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),                    # More batch normalization
            nn.Dropout(p=dropout_rate_2 * 0.5),    # Third dropout (0.2)
            nn.Linear(256, num_classes)             # Final output layer
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    
    def _initialize_weights(self):
        """Initialize classifier weights using He initialization"""
        for m in self.backbone.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
        
        Returns:
            Output logits (batch_size, num_classes)
        """
        return self.backbone(x)
    
    
    def freeze_backbone(self):
        """
        Freeze backbone parameters (for Phase 1 training)
        Only the classifier will be trained
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Ensure classifier is trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        print("✅ Backbone frozen - Only classifier is trainable")
    
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone parameters (for Phase 2 fine-tuning)
        The entire network will be trained
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        print("✅ Backbone unfrozen - Entire network is trainable")
    
    
    def unfreeze_last_n_layers(self, n=3):
        """
        Unfreeze only the last N layers of the backbone (alternative to full unfreezing)
        
        Args:
            n (int): Number of layers to unfreeze from the end
        """
        # First, freeze everything
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Then unfreeze last n layers
        layers = list(self.backbone.features.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Classifier is always trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        print(f"✅ Last {n} backbone layers unfrozen")


def count_parameters(model, trainable_only=False):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        trainable_only (bool): Count only trainable parameters
    
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def estimate_model_size(model):
    """
    Estimate model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def test_model():
    """Test model creation and forward pass"""
    print("\n" + "="*70)
    print("🧪 TESTING MODEL")
    print("="*70)
    
    # Create model
    model = MobileNetV2Transfer(pretrained=False, num_classes=3)
    
    # Model info
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = estimate_model_size(model)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {model_size:.2f} MB")
    
    # Test forward pass
    print(f"\n🔄 Testing forward pass...")
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test freeze/unfreeze
    print(f"\n🔒 Testing freeze/unfreeze...")
    model.freeze_backbone()
    frozen_params = count_parameters(model, trainable_only=True)
    print(f"   Trainable after freeze: {frozen_params:,}")
    
    model.unfreeze_backbone()
    unfrozen_params = count_parameters(model, trainable_only=True)
    print(f"   Trainable after unfreeze: {unfrozen_params:,}")
    
    print(f"\n✅ All tests passed!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    test_model()
