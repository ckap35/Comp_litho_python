import numpy as np
from matplotlib import pyplot as plt

# import math
import random
import SMO_functions_V5 as SMO


###The initialization of the parameter in the optimization%%%%%%
Mask_dim = (
    230 + 1
)  # Mask dimension in pixels   #change variable name (Mask dimension px)
N_filter = 21 + 20  # Amplitude impulse response dimension
pixel = 2  # Pixel size (nm)
NA = 1.25 * 1  # Numerical aperture
lamda = 193 * 1  # Wavelength (nm)
order = 1  # Order of Bessel function (assuming step function truncation of diffraction in Fourier plane)
sigma_large_inner = 0.99 - 0.1  # Inner Partial coherence factor
sigma_large_outer = 0.99  # Outer Partial coherence factor
N_coherence = 18 + 0  # Source dimension


# target = np.zeros((Mask_dim, Mask_dim))  # Initial Target and initial mask
# target[:, 16 * 2 : 27 * 2] = 1
# target[:, 43 * 2 : 64 * 2] = 1
# target[0 : 15 * 2, :] = 0
# target[65 * 2 : 80 * 2, :] =  0
# target[10 * 2 : 70 * 2, 20 * 2 : 25 * 2] = 1
# target[10 * 2 : 72 * 2, 20 * 2 : 22 * 2] = 1
# target[55 * 2 : 60 * 2, 10 * 2 : 22 * 2] = 1
# target[20 * 2 : 50 * 2, 1 : 25 * 2] = 0
# target[10 * 2 : 30 * 2, 16 * 2 : 20 * 2] = 1
# target[:,100:105] = 0


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


plt.imshow(
    target,
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Target")
plt.colorbar()
plt.clim(0, 2)
plt.show()


source_0 = SMO.initial_source(
    sigma_large_inner, sigma_large_outer, NA, N_coherence, pixel, Mask_dim, lamda
)
h, g = SMO.amplitude_response_func(N_filter, NA, lamda, pixel, order)
aerial = SMO.calculate_aerial_image(
    source_0, target, N_filter, NA, lamda, pixel, order, h, g
)


################### Initialize optimization
######### Source
source = SMO.initial_source(
    sigma_large_inner, sigma_large_outer, NA, N_coherence, pixel, Mask_dim, lamda
)

#####mask
m = target

flag_mask = np.zeros(
    (Mask_dim, Mask_dim)
)  # Locations of the changable pixels on mask in the current iteration
# flag_source=np.zeros((N_coherence,N_coherence));   #Locations of the changable pixels on source in the current iteration
error = 100000  # Output pattern error in the current iteration

###########Calculate the output pattern error in the previous iteration###########
aerial_pre = SMO.calculate_aerial_image(
    source, m, N_filter, NA, lamda, pixel, order, h, g
)
error_pre = SMO.compute_error(
    source, m, N_filter, NA, lamda, pixel, order, h, g, target
)  # Output pattern error in the previous iteration

m = 1 * target

error_post = 0
repeats = 0

all_error = []

plt.imshow(m)

# learning_rate = 5 * 0.01
learning_rate_0 = 0.1
learning_rate = 1 * learning_rate_0
complexity_weight = 0.005 * 0
cost_tol = 0.001

for num_iterations in range(30000):

    m_iter = 1 * m  # Initialize optimization to output of previous loop

    # plt.imshow(m_iter,  extent=[
    #      -Mask_dim * pixel / 2,
    #      Mask_dim * pixel / 2,
    #      -Mask_dim * pixel / 2,
    #      Mask_dim * pixel / 2,
    #  ],)
    # plt.title("Best mask")
    # plt.colorbar()
    # plt.clim(-2, 2)
    # plt.show()

    # flag_mask=0*m_iter+1        #Consider all pixels on mask as candidates
    direction_mask = 1 * SMO.mask_gradient(
        source, m_iter, target, aerial, h, g, N_filter, pixel, lamda
    )  # Calculate new mask gradient
    # test_mask=abs(np.multiply(direction_mask,flag_mask))    #abs

    # Hard binarization of gradient
    direction_mask[direction_mask > 0] = 1
    direction_mask[direction_mask < 0] = -1

    # direction_mask=1*np.tanh(direction_mask*100)    # soft binarize gradient

    # plt.imshow(direction_mask, interpolation="none")
    # plt.title("Direction mask")
    # plt.colorbar()
    # plt.clim(-2, 2)
    # plt.show()

    # test_mask = abs(direction_mask)

    m_iter = (
        m_iter + learning_rate * direction_mask
    )  # Flip pixel sign at position of max gradient

    aerial = SMO.calculate_aerial_image(
        source, m_iter, N_filter, NA, lamda, pixel, order, h, g
    )  # calculate new AI
    error_post = SMO.compute_error(
        source, m_iter, N_filter, NA, lamda, pixel, order, h, g, target
    )  # Output pattern error

    # complexity_penalty = complexity_weight * np.sum(abs(m_iter))  # L1 regularization
    complexity_penalty = 0
    # print(complexity_penalty)

    error_post += complexity_penalty  # add additional cost for new complexity

    if error_post >= error_pre:
        repeats += 1
        print("Grad step failed to improve cost repeats = " + str(repeats))
        learning_rate = learning_rate_0 * 1 * random.uniform(0.01, 1)

    if error_post < error_pre:
        m = 1 * m_iter
        error_pre = 1 * error_post
        repeats = 0
        learning_rate = learning_rate_0

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

    print("Iteration = " + str(num_iterations) + "     Error = " + str(error_pre))
    all_error.append(error_pre)

    if repeats > 20:
        print("convereged")
        break


plt.imshow(
    target,
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Target")
plt.colorbar()
plt.clim(0, 2)
plt.show()

plt.imshow(
    aerial_pre,
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Inital AI")
plt.colorbar()
plt.clim(0, 2)
plt.show()


plt.imshow(
    np.real(aerial),
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Final AI")
plt.colorbar()
plt.clim(0, 2)
plt.show()


plt.imshow(
    np.real(m_iter),
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Final mask")
plt.colorbar()
plt.clim(-2, 2)
plt.show()


plt.imshow(
    abs(target - np.real(aerial_pre)),
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("Initial AI-Target")
plt.colorbar()
plt.clim(-0.5, 0.5)
plt.show()


plt.imshow(
    abs(target - np.real(aerial)),
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("final AI- Target")
plt.colorbar()
plt.clim(-0.5, 0.5)
plt.show()


plt.imshow(
    1 * (direction_mask),
    extent=[
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
        -Mask_dim * pixel / 2,
        Mask_dim * pixel / 2,
    ],
)
plt.title("MaskGradient")
plt.colorbar()
plt.show()


x = np.linspace(-Mask_dim / 2, Mask_dim / 2, Mask_dim) * pixel

plt.plot(x, np.real(aerial[round(Mask_dim / 2), :]), "r")
plt.plot(x, np.real(aerial_pre[round(Mask_dim / 2), :]), "k")
plt.plot(x, np.real(target[round(Mask_dim / 2), :]), "b")
plt.title("AI lineout")
plt.show()


plt.plot(np.log10(all_error), "b")
plt.show()

##

plt.plot(x, m_iter[round(Mask_dim / 2), :], "r")
plt.plot(x, m_iter[:, round(Mask_dim / 2)], "b")
plt.title("Mask lineout")
plt.show()


# plt.plot((all_error), "b")
# plt.show()
