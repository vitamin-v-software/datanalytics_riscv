import torch

# Model and weights
model_arch_name = "resnet50"
pretrained_model_weights_path = "models/resnet50_checkpoint.pth"
model_num_classes = 10

# Dataset
train_image_dir = "../data/val_subset_2"
valid_image_dir = "../data/val_subset_2"
image_size = 224  # Standard for ResNet

# Dataloader
batch_size = 32
num_workers = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Misc
test_print_frequency = 10

model_ema_decay = 0.9999

# Training
epochs = 2
exp_name = "resnet50_experiment"
resume = False

# Optimizer
model_lr = 0.1
model_momentum = 0.9
model_weight_decay = 1e-4

# Scheduler
lr_scheduler_T_0 = 10
lr_scheduler_T_mult = 2
lr_scheduler_eta_min = 0

# Loss
loss_label_smoothing = 0.1

# Logging
train_print_frequency = 10
valid_print_frequency = 10
