"""
Created on Thu Jul 02 11:12:05 2020

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
from PIL import Image
import os
import copy
from torch.nn import functional
from torch.autograd import Variable
import torch
import torch.nn as nn

import random
import matplotlib.cm as mpl_color_map
# Torch imports
from torchvision import models

random.seed(3)


class Normalization(nn.Module):
    # Normalization for neural network input
    # You might not need this depending on how you preprocess data
    def __init__(self, device_id):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).float().cuda(device_id)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).float().cuda(device_id)

    def forward(self, img):
        return ((img - self.mean) / self.std).unsqueeze(0)


class NNModel(nn.Module):
    # A forward pass incorporating the normalization and the model
    def __init__(self, model, normalization, device_id):
        super(NNModel, self).__init__()
        self.model = nn.Sequential(normalization, model)

    def forward(self, img):
        return self.model(img)


def read_input_image(folder, im_name):
    # Read image from a file, return a torch float tensor
    im = Image.open(folder+im_name)
    im = np.asarray(im)/224
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float()
    return im


def create_random_loc_matrix(one_amount):
    # Create a random binary matrix with one_amount representing percentage of
    # white pixels compared to all pixels
    loc_matrix = np.random.randint(0, 100, (224, 224))/100
    # Assign 0s and 1s
    loc_matrix[loc_matrix < one_amount] = -1
    loc_matrix[loc_matrix >= one_amount] = 0
    loc_matrix[loc_matrix == -1] = 1
    loc_matrix = np.asarray([loc_matrix]*3)
    loc_matrix = torch.from_numpy(loc_matrix)
    return loc_matrix


def create_square_loc_matrix(square_amt):
    # Create a binary matrix with the square_amt representing the edge length of the
    # center square
    center_loc_matrix = np.ones((3, square_amt, square_amt))
    pad_amt = (224 - square_amt)//2
    center_loc_matrix = np.asarray([np.pad(loc_slice,
                                           pad_amt,
                                           mode='constant') for loc_slice in center_loc_matrix])
    white_amt = np.sum(center_loc_matrix)/(3*224*224)
    center_loc_matrix = torch.from_numpy(center_loc_matrix)
    return center_loc_matrix, white_amt


def save_input_image(modified_im, im_name, folder_name='result_images', save_flag=True):
    # Save a torch tensor as an image
    if 'torch' in str(type(modified_im)):
        modified_copy = copy.deepcopy(modified_im).cpu().data.numpy()
    else:
        modified_copy = copy.deepcopy(modified_im)
    modified_copy = modified_copy * 255
    # Box constraint
    modified_copy[modified_copy > 255] = 255
    modified_copy[modified_copy < 0] = 0
    modified_copy = modified_copy.transpose(1, 2, 0)
    modified_copy = modified_copy.astype('uint8')
    if save_flag:
        save_image(modified_copy, im_name, folder_name)
    return modified_copy


def save_image(im_as_arr, im_name, folder_name):
    # The saving operation itself
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    image_name_with_path = folder_name + '/' + str(im_name) + '.png'
    pred_img = Image.fromarray(im_as_arr)
    pred_img.save(image_name_with_path)


def get_random_target_class(class_start, class_end, class_except):
    # Generate a random integer representing target class
    random_class = random.randint(class_start, class_end)
    while random_class == class_except:
        random_class = random.randint(class_start, class_end)
    return random_class
