import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

training_transforms = transforms.Compose([transforms.Resize(244), 
                                          transforms.RandomResizedCrop(224),  
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(244),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                                            std = [ 0.229, 0.224, 0.225 ]),])


def data_to_loaders(train_dir, valid_dir):
    training_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    training_loader = torch.utils.data.DataLoader(training_datasets, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    return training_datasets, valid_datasets, training_loader, valid_loader