import torch 

config = {}

config["traindata"] = '/home/prantik/Documents/SRCNN/data/91-image_x4.h5'
config['evaldata'] = '/home/prantik/Documents/SRCNN/data/Set5_x4.h5'
config['cuda'] = torch.cuda.is_available()

config['results'] = 'results'
config['pretrained'] = 'pretrained'

config['model_path'] = '/home/prantik/Documents/SRCNN/pretrained/epoch_19.pth'

config['lr'] = 1e-5
config['batch_size'] = 128
config['epochs'] = 100

config['image_file'] = 'aditya.jpeg'

config['scale'] = 3