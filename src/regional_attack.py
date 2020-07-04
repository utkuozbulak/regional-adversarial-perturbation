"""
Created on Thu Jul 02 11:12:05 2020

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch


class RegionalAdversarialAttack():
    def __init__(self, model1, model2, device_id):
        # Model to generate adversarial perturbation
        self.model1 = model1
        self.model1.eval()
        self.model1.cuda(device_id)
        # Model to test adversarial example against
        self.model2 = model2
        self.model2.eval()
        self.model2.cuda(device_id)
        # Other necessary parameters
        self.device_id = device_id
        self.ce = nn.CrossEntropyLoss()

    def localize_perturbation(self, image_to_perturb, perturbation, localization_matrix):
        # \mathbf{X}_{n+1} = \mathbf{X}_{n} + \mathbf{P}_n * \mathbf{L}:
        image_to_perturb.data = image_to_perturb.data - (perturbation * localization_matrix)
        return image_to_perturb

    def calculate_lp_norms(self, org_im, adv_im):
        im_diff = torch.abs(org_im - adv_im).cpu()
        # l_infty distance
        linf_norm = torch.max(im_diff).item()
        # l_0 distance
        im_diff = im_diff.sum(dim=0)
        im_diff[im_diff != 0] = 1
        l0_norm = torch.sum(im_diff).item() / (org_im.size()[1] * org_im.size()[2])
        # l_2 distance
        l2_norm = torch.dist(org_im, adv_im, p=2).cpu().item()
        return l0_norm, l2_norm, linf_norm

    def generate(self, im, org_class, target_class, localization_matrix):
        # Copy the original (unperturbed) image to calculate L_p distances later
        org_im = copy.copy(im).cuda(self.device_id)
        self.im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class]))).cuda(self.device_id)
        self.processed_image = Variable(im.cuda(self.device_id), requires_grad=True)

        l0_norm, l2_norm, linf_norm, adv_im = None, None, None, None
        # Start optimization
        for i in range(1, 251):
            # Zero grads to prevent lingering gradients from previous iterations
            self.model1.zero_grad()
            self.processed_image.grad = None

            # Forward pass from the first model
            output = self.model1(self.processed_image)

            # Calculate perturbation
            # \mathbf{P} = \nabla_x (g(\theta, \mathbf{X}_n))_c * 0.0039:
            ce_loss = self.ce(output, self.im_label_as_var)
            ce_loss.backward()
            perturbation = torch.sign(self.processed_image.grad.data)
            perturbation = perturbation * 0.0039

            # Localize perturbation
            self.processed_image.data = self.localize_perturbation(self.processed_image, perturbation, localization_matrix)

            # Box constraint, make sure it is a valid image
            self.processed_image.data.clamp_(0, 1)

            # Do forward pass again to see if the prediction changed
            with torch.no_grad():
                like, pred = torch.max(self.model2(self.processed_image).cpu(), dim=1)
            if pred.item() != org_class:
                l0_norm, l2_norm, linf_norm = self.calculate_lp_norms(org_im, self.processed_image)
                adv_im = self.processed_image.data
                print('Adversarial example successfully transferred to the target model!')
                print('L0 norm:', l0_norm, 'L2 norm', l2_norm, 'Linf norm', linf_norm)
                break
        else:
            print('Adversarial example did not transfer to the second model!')

        return l0_norm, l2_norm, linf_norm, adv_im
