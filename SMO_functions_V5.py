# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:54:25 2022

@author: ckap3
"""

import numpy as np

# from matplotlib import pyplot as plt
import math
import cmath
from scipy import special
from scipy import signal

# from scipy import ndimage


def amplitude_response_func(N_filter, NA, lamda, pixel, order):
    midway = (N_filter + 1) / 2
    # middle of low pass filter
    h = np.zeros((N_filter, N_filter), dtype=complex)
    radius = 0
    for row in range(N_filter):
        for column in range(N_filter):
            radius = pixel * math.sqrt((row - midway) ** 2 + (column - midway) ** 2)
            if radius <= (midway) * pixel:
                argument = 2 * math.pi * radius * NA / lamda
                if radius == 0:
                    h[row, column] = h[row, column]
                else:
                    h[row, column] = special.jv(order, argument) / argument
                    # bessel function, first kind, order 1, representing FFT of circ function in pupil plane.

    h = h / np.sum(h)

    h_vector = np.zeros(N_filter * N_filter + (N_filter + 2), dtype=complex)
    g = h * 0

    for i in range(N_filter):
        for j in range(N_filter):
            h_vector[(i - 1) * N_filter + j] = h[i, j]

    for i in range(N_filter):
        for j in range(N_filter):
            g[i, j] = h_vector[(N_filter - i) * N_filter + (N_filter + 1 - j)]
            # inverse vector

    return h, g


def compute_exponential(source, m, N_filter, lamda, pixel, p, q):
    N_coherence = source.shape[0]
    N = m.shape[0]
    midway_coherence = (N_coherence + 1) / 2
    D = pixel * N
    omega_0 = math.pi / D
    midway = (N_filter + 1) / 2

    exponential_output = np.zeros((N_filter, N_filter), dtype=complex)
    # has the same dimension as the filter h
    for row in range(N_filter):
        for column in range(N_filter):
            argument = (p - midway_coherence) * (row - midway) * pixel + (
                q - midway_coherence
            ) * (column - midway) * pixel
            exponential_output[row, column] = cmath.exp(1j * omega_0 * argument)
    return exponential_output


def calculate_aerial_image(source, m, N_filter, NA, lamda, pixel, order, h, g):
    N_coherence = source.shape[0]
    N = m.shape[0]
    midway_coherence = (N_coherence + 1) / 2
    D = pixel * N
    omega_0 = math.pi / D
    midway = (N_filter + 1) / 2
    # middle of low pass filter
    aerial = np.zeros((N, N), dtype=complex)
    normalize = abs(np.sum(source))
    # [h,g]=amplitude_response_func(N_filter,NA,lamda,pixel,order)
    for p in range(N_coherence):
        for q in range(N_coherence):
            if source[p, q] > 0:
                exponential = compute_exponential(
                    source, m, N_filter, lamda, pixel, p, q
                )
                A = m
                B = np.multiply(h, exponential)
                aerial = (
                    aerial
                    + source[p, q]
                    / normalize
                    * abs(signal.convolve(A, B, mode="same")) ** 2
                )

    return np.real(aerial)


def initial_source(
    sigma_large_inner, sigma_large_outer, NA, N_coherence, pixel, N, lamda
):
    D = pixel * N
    D_C_1 = lamda / 2 / sigma_large_outer / NA
    # Coherence length
    D_C_2 = lamda / 2 / sigma_large_inner / NA
    # Coherence length
    midway_coherence = (N_coherence + 1) / 2
    # Middle point of illumination
    radius_1 = D / (2 * D_C_1)
    # Inner radius of annular illumination
    radius_2 = D / (2 * D_C_2)
    # Outer radius of annular illumination
    source_out = np.zeros((N_coherence, N_coherence), dtype=complex)
    # Illumination pattern

    for row in range(N_coherence):
        for column in range(N_coherence):
            radius = pixel * math.sqrt(
                (row - midway_coherence) ** 2 + (column - midway_coherence) ** 2
            )
            # print(radius)
            if (radius <= radius_1 * pixel) & (radius >= radius_2 * pixel):
                source_out[row, column] = 1

    return source_out


def mask_gradient(source, mask, pz, aerial, h, g, N_filter, pixel, lamda):
    N_coherence = source.shape[0]
    N = mask.shape[0]
    direction_mask = 0 * mask
    normalize = np.sum(source)
    midway_coherence = (N_coherence + 1) / 2
    midway = (N_filter + 1) / 2
    # middle of low pass filter
    D = pixel * N
    omega_0 = math.pi / D
    for p in range(N_coherence):
        for q in range(N_coherence):
            if source[p, q] > 0:
                exponential = compute_exponential(
                    source, mask, N_filter, lamda, pixel, p, q
                )
                test = np.real(
                    signal.convolve(
                        np.multiply(
                            (pz - aerial),
                            signal.convolve(
                                mask, np.multiply(h, exponential), mode="same"
                            ),
                        ),
                        np.conj(np.multiply(g, exponential)),
                        mode="same",
                    )
                    + signal.convolve(
                        np.multiply(
                            (pz - aerial),
                            signal.convolve(
                                mask, np.conj(np.multiply(h, exponential)), mode="same"
                            ),
                        ),
                        np.multiply(g, exponential),
                        mode="same",
                    )
                )
                direction_mask = direction_mask + (2) * source[p, q] / normalize * test
    return np.real(direction_mask)


def source_gradient(
    source, mask, pz, aerial, h, g, N_coherence, N_filter, pixel, N, lamda
):
    direction_source = 0 * source
    normalize = np.sum(source)
    midway_coherence = (N_coherence + 1) / 2
    midway = (N_filter + 1) / 2
    # middle of low pass filter
    D = pixel * N
    omega_0 = math.pi / D
    for p in range(N_coherence):
        for q in range(N_coherence):
            if source[p, q] > 0:
                exponential = compute_exponential(
                    source, mask, N_filter, lamda, pixel, p, q
                )
                direction_source[p, q] = (
                    -1
                    * np.sum(
                        np.multiply(
                            pz - aerial,
                            abs(
                                signal.convolve(
                                    mask, np.multiply(h, exponential), mode="same"
                                )
                            )
                            * 2,
                        )
                    )
                    * (-2)
                    / normalize
                )
    return direction_source


def soft_binarize(data, slope):
    output = 2 * (-0.5 + 1 / (1 + np.exp(-data)))
    return output


def compute_error(source, m, N_filter, NA, lamda, pixel, order, h, g, pz):
    AI = np.real(
        calculate_aerial_image(source, m, N_filter, NA, lamda, pixel, order, h, g)
    )
    error = abs(np.sum((pz - AI) ** 2))
    return error


def calculate_aerial_image_old(source, m, N_filter, NA, lamda, pixel, order, h, g):
    N_coherence = source.shape[0]
    N = m.shape[0]
    midway_coherence = (N_coherence + 1) / 2
    D = pixel * N
    omega_0 = math.pi / D
    midway = (N_filter + 1) / 2
    # middle of low pass filter
    aerial = np.zeros((N, N), dtype=complex)
    normalize = abs(np.sum(source))
    # [h,g]=amplitude_response_func(N_filter,NA,lamda,pixel,order)
    for p in range(N_coherence):
        for q in range(N_coherence):
            if source[p, q] > 0:
                exponential = np.zeros((N_filter, N_filter), dtype=complex)
                # has the same dimension as the filter h
                for row in range(N_filter):
                    for column in range(N_filter):
                        argument = (p - midway_coherence) * (row - midway) * pixel + (
                            q - midway_coherence
                        ) * (column - midway) * pixel
                        exponential[row, column] = cmath.exp(1j * omega_0 * argument)

                A = m
                B = np.multiply(h, exponential)
                aerial = (
                    aerial
                    + source[p, q]
                    / normalize
                    * abs(signal.convolve(A, B, mode="same")) ** 2
                )

    return np.real(aerial)
