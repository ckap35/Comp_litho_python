# Imaging model and gradient

The model follows scalar Abbe imaging: compute one coherent image field per
source point, square its magnitude, then sum with the source intensity weights.
All lengths are in nanometers and frequencies in cycles per nanometer.

## Sampling and boundaries

Arrays use `(y, x)` order. A centered axis of length `N` has coordinates
`arange(N) - (N - 1) / 2`. The source-frequency spacing is
`1 / (image_size * pixel_size_nm)`, and sigma is the source radius divided by
`NA / wavelength_nm`. An even source lattice has half-integer coordinates and
no on-axis point.

The sampled amplitude response is `J1(u) / u`, where

$$
u = \frac{2\pi\,\mathrm{NA}}{\lambda}\sqrt{x^2+y^2}.
$$

At the origin its value is 1/2. The response is set to zero outside a circle
of radius `(kernel_size // 2) * pixel_size_nm` and normalized to unit sum.
The default 41-pixel kernel and 2 nm spacing give a 40 nm support radius.

For source frequency `(fy, fx)`, the coherent kernel is

$$
K_s(x,y)=h(x,y)\exp[2\pi i(f_y y+f_x x)].
$$

The same frequency grid defines both the annulus and the phase ramp. Source
weights are nonnegative, normalized intensities. `source_weights` can replace
the default annulus on that grid.

Convolution is linear, zero padded, and cropped to the input shape. The mask
is opaque outside the array.

## Objective and derivative

For a real amplitude mask `m`, target intensity `T`, and source weights `w_s`,

$$
u_s=K_s*m,\qquad I=\sum_s w_s|u_s|^2,
\qquad L=\sum_{i,j}(I_{ij}-T_{ij})^2.
$$

The derivative with respect to the real mask values is

$$
\nabla_m L=4\sum_s w_s\operatorname{Re}
\left[K_s^H*((I-T)u_s)\right].
$$

`K_s^H` is the reversed, conjugated complex kernel, including the illumination
phase. `ImagingModel.loss_and_gradient` reuses the forward fields when computing
this derivative. Kernels and source arrays are copied and cached as read-only
arrays.

The optimization variable is a real amplitude with no bounds, so mask values
can be negative or exceed one. The source weights stay fixed, and the loss
uses aerial intensity directly.

## Stopping conditions

Armijo backtracking accepts a step only when its loss and gradient are finite
and the loss decreases by the required amount. If every trial fails, the
optimizer returns the last accepted mask with `line_search_failed`.

| Status | Stopping condition |
| --- | --- |
| `gradient_tolerance` | Gradient norm reaches the specified tolerance |
| `relative_loss_tolerance` | Relative change in accepted loss reaches the specified tolerance |
| `max_iterations` | Iteration budget exhausted; `converged` is false |
| `line_search_failed` | All trial steps rejected; the last accepted mask is returned |

The result stores accepted losses, gradient norms, step sizes, and the number
of rejected trials.
