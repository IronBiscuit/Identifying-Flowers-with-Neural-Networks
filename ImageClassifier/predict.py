from get_input_args import get_input_args_predict
from load_checkpoint import load_checkpoint
from process_image import process_image

import torch
import json



def main():
    in_arg = get_input_args_predict()
    checkpoint_pth = in_arg.checkpoint_dir
    image_pth = in_arg.image_dir
    top_k = in_arg.top_k
    category_names_pth = in_arg.category_names
    use_gpu = in_arg.gpu
    
    convert_cat_to_name = True
    
    if (category_names_pth is None):
        convert_cat_to_name = False
    else:
        with open(category_names_pth, 'r') as f:
            cat_to_name = json.load(f)
    
    if (top_k is None):
        print("top_k was not specified. Please specify a top_k!")
        return
    
    if (use_gpu == 1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    optimizer, model = load_checkpoint(checkpoint_pth)
    
    model.eval()
    model.to(device)
    
    image_tensor = process_image(image_pth)
    image_tensor.unsqueeze_(0)
    inputs = image_tensor.to(device)
    with torch.no_grad():
        logps = model.forward(inputs.float())
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
    top_probs = top_p.detach().cpu().numpy()
    top_classes = top_class.detach().cpu().numpy()
    top_probs = top_probs[0]
    top_classes = top_classes[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    for i in range(top_k):
        top_classes[i] = idx_to_class[top_classes[i]]
    
    if (convert_cat_to_name):
        top_classes_named = []
        for i in range(len(top_classes)):
            top_classes_named.append(cat_to_name[str(top_classes[i])])
        print(top_probs)
        print(top_classes_named)
        return top_probs, top_classes_named
    
    print(top_probs)
    print(top_classes)
    return top_probs, top_classes
    
    
    
if __name__ == "__main__":
    main()