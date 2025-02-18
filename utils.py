import os
import torch
import imageio
import logging
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def logger(args):
    if args.log_type == 'print':
        return print
    elif args.log_type == 'file':
        ### Todo 4
        # raise NotImplementedError
        # filename = 
        filename = os.path.join("experiments",args.path,'info.log')
        logFormatter = logging.Formatter(fmt=' %(name)s :: %(levelname)s :: %(message)s')
        logging.basicConfig(filename=filename, 
                            level=logging.INFO, 
                            format='%(asctime)s %(message)s', filemode='w',
                            datefmt='%d/%m/%Y %H:%M:%S')
        return logging.info

def save_model(agent, args, name):
    path = os.path.join("experiments", args.path, 'models')
    os.makedirs(path, exist_ok=True)
    torch.save(agent.model.state_dict(), os.path.join(path, '{}.pt'.format(name)))


def plot_model(env, args, name):
    path = os.path.join("experiments", args.path, 'plots')
    os.makedirs(path, exist_ok=True)
    frames = env.to_draw
    for i in range(frames.shape[0]):
        im = Image.fromarray(frames[i, :,:,:].astype(np.uint8))
        im.save(os.path.join(path, '{}_{}.png'.format(name, i)))

    path_GIF = os.path.join("experiments", args.path, 'GIF')
    os.makedirs(path_GIF, exist_ok=True)

    images = []
    for i in range(frames.shape[0]):
        images.append(imageio.imread(os.path.join(path, '{}_{}.png'.format(name, i))))
    imageio.mimsave(os.path.join(path_GIF, '{}.gif'.format(name)), images, duration=10, loop=0) 

def get_optimizer(model, lr, optimizer):
    if optimizer == 'SGD':
        raise NotImplementedError
        ### Todo 17
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.0)
    elif optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer not recognized")
    