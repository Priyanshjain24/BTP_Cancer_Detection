import torch, os
import torch.nn as nn
from torchvision import models

class ModelManager:
    def __init__(self, model_name, num_classes, device, checkpoint_path=None, dropout_p=0.5, freeze_layers=False, use_multi_gpu=False, model_dir="checkpoints"):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.dropout_p = dropout_p
        self.freeze_layers = freeze_layers
        self.use_multi_gpu = use_multi_gpu
        self.model_dir = model_dir
        self.model = self._initialize_model()
        self.optimizer = None  # Will be set externally

    def _initialize_model(self):
        model_dict = {
            'RESNET18': models.resnet18(weights='ResNet18_Weights.DEFAULT'),
            'RESNET34': models.resnet34(weights='ResNet34_Weights.DEFAULT'),
            'VGG11': models.vgg11_bn(weights='VGG11_BN_Weights.DEFAULT'),
            'VGG13': models.vgg13_bn(weights='VGG13_BN_Weights.DEFAULT'),
            'VGG16': models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT'),
            'VGG19': models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT'),
            'RESNET50': models.resnet50(weights='ResNet50_Weights.DEFAULT'),
            'CONVNEXT_TINY': models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT'),
            'REGNET': models.regnet_y_400mf(weights='RegNet_Y_400MF_Weights.DEFAULT'),
            'SWIN_TRANSFORMER_TINY': models.swin_t(weights='Swin_T_Weights.DEFAULT'),
            'EFFICIENT_NET': models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        }

        if self.model_name not in model_dict:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model = model_dict[self.model_name]

        # Freeze layers if required
        for param in model.parameters():
            param.requires_grad = not self.freeze_layers

         # Adjust classifier layer dynamically
        if hasattr(model, 'fc'):  # ResNet models
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(self.dropout_p), nn.Linear(num_ftrs, self.num_classes))

        elif hasattr(model, 'classifier'):  # VGG, EfficientNet, ConvNeXt
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(nn.Dropout(self.dropout_p), nn.Linear(num_ftrs, self.num_classes))

        elif hasattr(model, 'head'):  # Swin Transformer, RegNet
            num_ftrs = model.head.in_features
            model.head = nn.Sequential(nn.Dropout(self.dropout_p), nn.Linear(num_ftrs, self.num_classes))

        else:
            raise ValueError(f"Unknown classifier structure for model: {self.model_name}")

        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training!")
            model = torch.nn.DataParallel(model)

        return model.to(self.device)

    def save_checkpoint(self, name, epoch, training_stats):
        """Save model and optimizer state."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Handle DataParallel models
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()

        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'training_stats': training_stats
        }, os.path.join(self.model_dir, name))

    def load_checkpoint(self, optimizer):
        """Load model and optimizer checkpoint if available."""
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            # If using DataParallel, add 'module.' prefix to keys
            if isinstance(self.model, torch.nn.DataParallel):
                state_dict = {"module." + k if not k.startswith("module.") else k: v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            return checkpoint.get('epoch', 0), checkpoint.get('training_stats', {})

        else:
            print("No checkpoint found, starting from scratch")
            return 0, {}