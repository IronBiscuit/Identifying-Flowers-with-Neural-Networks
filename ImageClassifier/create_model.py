from torchvision import models
from torch import nn
from torch import optim

def create_model(learning_rate, hidden_units, model_arch, output_num):
    try:
        model = getattr(models, model_arch, None)(pretrained = True)
        model.pretrained = True
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, output_num),
                                 nn.LogSoftmax(dim=1))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        return model, criterion, optimizer
        
    except AttributeError:
        print("Error! There is no such model!")
        return None