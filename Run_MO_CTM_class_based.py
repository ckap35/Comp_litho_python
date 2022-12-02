# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:32:29 2022

@author: ckap3
"""

import SMO_classes_test as SMO
import numpy as np
from matplotlib import pyplot as plt
import math
import cmath
from scipy import special
from scipy import signal
import random


my_model = SMO.Model(
    lamda=193, NA=1 * 1.25, N_filter=41 * 1, pixel=2, order=1, N=231, N_coherence=18
)
my_model.initalize_source(outer_sigma=0.99, inner_sigma=0.99 - 0.1)


pixel = 2  # Pixel size (nm)
Mask_dim = (
    230 + 1
)  # Mask dimension in pixels   #change variable name (Mask dimension px)

L = 15
offset = 15
target = np.zeros((Mask_dim, Mask_dim))  # Initial Target and initial mask
target[
    round(Mask_dim / 2) - L - offset : round(Mask_dim / 2) + L - offset,
    round(Mask_dim / 2) - L - offset : round(Mask_dim / 2) + L - offset,
] = 1
target[
    round(Mask_dim / 2) - L + offset : round(Mask_dim / 2) + L + offset,
    round(Mask_dim / 2) - L + offset : round(Mask_dim / 2) + L + offset,
] = 1
target[
    round(Mask_dim / 2) - L - offset : round(Mask_dim / 2) + L - offset,
    round(Mask_dim / 2) - L + offset : round(Mask_dim / 2) + L + offset,
] = 1
target[
    round(Mask_dim / 2) - L + offset : round(Mask_dim / 2) + L + offset,
    round(Mask_dim / 2) - L - offset : round(Mask_dim / 2) + L - offset,
] = 1

m = target


# my_model.add_mask(1*m)
my_model.add_target(target)

# AI_pre=my_model.compute_aerial_image_local()


error = 100000  # Output pattern error in the current iteration
AI_pre = my_model.compute_aerial_image_local(m)
aerial = AI_pre
grad = my_model.mask_gradient(mask=m, aerial=aerial)


all_error = []
repeats = 0
learning_rate_0 = 0.1 * 1
learning_rate = 1 * learning_rate_0
complexity_weight = 0.005 * 0
cost_tol = 0.001
error_pre = my_model.compute_error(target=m, aerial=aerial)

for num_iterations in range(30000):

    m_iter = 1 * m  # Initialize optimization to output of previous loop

    # Calculate new mask gradient
    direction_mask = 1 * my_model.mask_gradient(mask=m_iter, aerial=aerial)

    # Hard binarization of gradient
    #direction_mask[direction_mask > 0] = 1
    #direction_mask[direction_mask < 0] = -1

    m_iter = (
        m_iter + learning_rate * direction_mask
    )  # update mask

    # my_model.add_mask(m_iter)
    aerial = my_model.compute_aerial_image_local(m_iter)  # calculate new AI
    error_post = abs(np.sum((target - aerial) ** 2))  # Output pattern error

    # complexity_penalty = complexity_weight * np.sum(abs(m_iter))  # L1 regularization
    complexity_penalty = 0
    # print(complexity_penalty)

    error_post += complexity_penalty  # add additional cost for new complexity

    if error_post >= error_pre:
        repeats += 1
        print("Grad step failed to improve cost repeats = " + str(repeats))
        learning_rate = learning_rate_0 * 1 * random.uniform(0.01, 1)

    else:
        m = 1 * m_iter
        error_pre = 1 * error_post
        repeats = 0
        learning_rate = learning_rate_0

        # plt.imshow(
        #     m_iter,
        #     extent=[
        #         -Mask_dim * pixel / 2,
        #         Mask_dim * pixel / 2,
        #         -Mask_dim * pixel / 2,
        #         Mask_dim * pixel / 2,
        #     ],
        # )
        # plt.title("Best mask")
        # plt.colorbar()
        # plt.clim(-2, 2)
        # plt.show()

    print("Iteration = " + str(num_iterations) + "     Error = " + str(error_pre))
    all_error.append(error_pre)

    if repeats > 5:
        print("convereged")
        break


plt.imshow(
    m_iter,
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Best mask")
plt.colorbar()
plt.clim(-2, 2)
plt.show()