from get_input_args import get_input_args_train
from transform import data_to_loaders
from create_model import create_model
from checkpoint import create_checkpoint

import os
import torch

def main():
    in_arg = get_input_args_train()
    data_dir = in_arg.data_dir
    save_dir = in_arg.save_dir
    model_arch = in_arg.arch
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    epochs = in_arg.epochs
    use_gpu = in_arg.gpu
    
    if (save_dir is None or model_arch is None or learning_rate is None or epochs is None or hidden_units is None):
        print("You have missing optional arguments, please specify them!")
        return
    
    training_datasets, valid_datasets, training_loader, valid_loader = data_to_loaders(data_dir + "/train", data_dir + "/valid")
    label_count = len(next(os.walk(data_dir + "/train"))[1])
    model, criterion, optimizer = create_model(learning_rate, hidden_units, model_arch, label_count)
    
    if (model is None):
        print("There is no such model, please specify a valid model!")
        return
    
    if (use_gpu == 1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in training_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()
            
    create_checkpoint(model, optimizer, save_dir, training_datasets, model_arch, learning_rate)   
    

    
if __name__ == "__main__":
    main()