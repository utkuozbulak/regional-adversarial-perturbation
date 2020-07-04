"""
Created on Thu Jul 02 11:12:05 2020

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import copy
import random
import torch
from torchvision import models

from funct_misc import (create_square_loc_matrix,
                        NNModel, Normalization,
                        read_input_image,
                        get_random_target_class,
                        save_input_image)
from regional_attack import RegionalAdversarialAttack


if __name__ == '__main__':
    # GPU ID
    DEVICE_ID = 0

    # Localization matrices

    # --- Center square ---
    # In the paper we experimented with square_len in {90, 120, 150}
    square_len = 90
    localization_matrix, selected_pix_percentage = create_square_loc_matrix(square_len)

    # --- Frame ---
    # In the paper we experimented with {20, 34, 58}
    # square_len = 204 # -> {204, 190, 166}
    # localization_matrix, selected_pix_percentage = create_square_loc_matrix(square_len)
    # loc_matrix_fr = 1 - loc_matrix_fr  # To take the frame instead of center square

    # --- Random ---
    # In the paper we experimented with {17%, 28%, 45%}
    # selected_pix_percentage = 0.18
    # localization_matrix, selected_pix_percentage = \
    #     create_random_loc_matrix(selected_pix_pectentage)

    localization_matrix = localization_matrix.float().cuda(DEVICE_ID)

    # Model to generate adversarial examples
    model1 = models.alexnet(pretrained=True).cuda(DEVICE_ID)
    model1.eval()
    model1 = NNModel(model1, Normalization(DEVICE_ID), 0).cuda(DEVICE_ID)
    # Model to test adversarial exampels
    model2 = models.resnet50(pretrained=True).cuda(DEVICE_ID)
    model2.eval()
    model2 = NNModel(model2, Normalization(DEVICE_ID), 0).cuda(DEVICE_ID)
    # Define the regional attack
    regional_attack = RegionalAdversarialAttack(model1, model2, DEVICE_ID)
    # Read the image
    im = read_input_image('../data/', 'sample_image.png')

    """
    # Placeholder code to check if the image is predicted correctly by both of the models
    org_im_copy = copy.copy(im)
    org_im_copy = org_im_copy.cuda(DEVICE_ID)
    org_out1, org_pred1 = torch.max(model1(org_im_copy).cpu(), dim=1)
    org_out1, org_pred2 = torch.max(model2(org_im_copy).cpu(), dim=1)
    # if org_pred1.item() == org_pred2.item():
        ...
    """

    org_pred = 378  # Change this for other images
    adv_target = get_random_target_class(0, 999, org_pred)
    print('Init class:', org_pred, 'Adv class:', adv_target)

    l0_norm, l2_norm, linf_norm, adv_im = regional_attack.generate(im, org_pred, adv_target, localization_matrix)
    save_input_image(adv_im, 'adversarial_example', folder_name='../output_images')
