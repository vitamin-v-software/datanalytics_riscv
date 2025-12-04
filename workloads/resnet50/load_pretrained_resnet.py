import torch
from torchvision.models import resnet50, ResNet50_Weights

# Load pretrained ResNet-50 model from torchvision
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Extract the state_dict
state_dict = model.state_dict()

# Save in checkpoint format
torch.save({'state_dict': state_dict}, './results/pretrained_models/ResNet-ImageNet_1K-32d70693.pth.tar')

