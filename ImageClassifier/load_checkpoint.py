import torch

from torchvision import models
from torch import optim

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    try:
        model = getattr(models, checkpoint['model_arch'], None)(pretrained = True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(optimizer_state_dict)
        model.class_to_idx = checkpoint['class_to_idx']
        return optimizer, model
    except:
        print("Checkpoint load failed")
        return None, None