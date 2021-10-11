import torch

def create_checkpoint(model, optimizer, save_dir, dataset, model_arch, lr):
    checkpoint = {
              'classifier': model.classifier,
              'optimizer_state_dict': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': dataset.class_to_idx,
              'model_arch': model_arch,
              'learning_rate': lr
    }
    torch.save(checkpoint, save_dir)