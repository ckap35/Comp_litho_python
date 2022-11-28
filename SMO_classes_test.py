# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:54:41 2022

@author: ckap3
"""
import numpy as np
from matplotlib import pyplot as plt
import math
import cmath
from scipy import special
from scipy import signal


class Model:
    def __init__(self, lamda, NA, N_filter, pixel, order, N, N_coherence):
        self.lamda = lamda
        self.NA = NA
        self.N_filter = N_filter
        self.pixel = pixel
        self.order = 1
        self.N = N
        self.N_coherence = N_coherence

    def __repr__(self):
        self.name = "Model Wavelength:" + str(self.lamda) + " NA:" + str(self.NA)
        return self.name

    # def add_mask(self,mask):
    #     self.mask=mask

    def add_target(self, target):
        self.target = target

    def initalize_source(self, outer_sigma, inner_sigma):
        NA = self.NA
        pixel = self.pixel
        N_coherence = self.N_coherence
        sigma_large_outer = outer_sigma
        sigma_large_inner = inner_sigma
        lamda = self.lamda
        N = self.N

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

        self.source = source_out
        return source_out

    def compute_exponential_local(self, p, q):
        # source=self.source
        N_coherence = self.N_coherence
        N = self.N
        pixel = self.pixel
        N_filter = self.N_filter

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

        self.expoential_output = exponential_output
        return exponential_output

    def compute_amplitude_response_func(self):
        N_filter = self.N_filter
        NA = self.NA
        pixel = self.pixel
        lamda = self.lamda
        order = self.order

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

        self.h = h
        self.g = g
        return h, g

    def compute_aerial_image_local(self, mask):
        pixel = self.pixel
        source = self.source
        N_coherence = self.N_coherence
        N = self.N
        m = mask
        N_filter = self.N_filter

        midway_coherence = (N_coherence + 1) / 2
        D = pixel * N
        omega_0 = math.pi / D
        midway = (N_filter + 1) / 2
        # middle of low pass filter
        aerial = np.zeros((N, N), dtype=complex)
        normalize = abs(np.sum(source))
        [h, g] = self.compute_amplitude_response_func()
        for p in range(N_coherence):
            for q in range(N_coherence):
                if source[p, q] > 0:
                    exponential = self.compute_exponential_local(p, q)
                    A = m
                    B = np.multiply(h, exponential)
                    aerial = (
                        aerial
                        + source[p, q]
                        / normalize
                        * abs(signal.convolve(A, B, mode="same")) ** 2
                    )
        # self.aerial=np.real(aerial)
        return np.real(aerial)

    def mask_gradient(self, mask, aerial):
        pixel = self.pixel
        pz = self.target
        # aerial=self.compute_aerial_image_local()
        N_coherence = self.N_coherence
        N = self.N
        N_filter = self.N_filter
        source = self.source
        # mask=self.mask
        lamda = self.lamda
        h = self.h
        g = self.g

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
                    exponential = self.compute_exponential_local(p, q)
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
                                    mask,
                                    np.conj(np.multiply(h, exponential)),
                                    mode="same",
                                ),
                            ),
                            np.multiply(g, exponential),
                            mode="same",
                        )
                    )
                    direction_mask = (
                        direction_mask + (2) * source[p, q] / normalize * test
                    )
        # self.gradient=direction_mask
        return np.real(direction_mask)

    def compute_error(self, target, aerial):
        pz = target
        # aerial=self.compute_aerial_image_local()
        error = abs(np.sum((pz - aerial) ** 2))
        return error
