import argparse

def get_input_args_train():
    Parse = argparse.ArgumentParser()
    Parse.add_argument('data_dir')
    Parse.add_argument('--save_dir', nargs='?', default='checkpoint.pth', type=str)
    Parse.add_argument('--arch', nargs='?', default='vgg11', type=str)
    Parse.add_argument('--learning_rate', nargs='?', default=0.003, type=float)
    Parse.add_argument('--hidden_units', nargs='?', default=500, type=int)
    Parse.add_argument('--epochs', nargs="?", default=1, type=int)
    Parse.add_argument('--gpu', nargs="?", default=0, const=1, type=int)
    args = Parse.parse_args()
    return args

def get_input_args_predict():
    Parse = argparse.ArgumentParser()
    Parse.add_argument('image_dir')
    Parse.add_argument('checkpoint_dir')
    Parse.add_argument('--top_k', nargs='?', default=1, type=int)
    Parse.add_argument('--category_names', nargs='?', type=str)
    Parse.add_argument('--gpu', nargs="?", default=0, const=1, type=int)
    args = Parse.parse_args()
    return args