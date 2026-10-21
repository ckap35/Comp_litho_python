# Computational lithography in Python

A small implementation of scalar, partially coherent Abbe imaging and
gradient-based continuous-tone mask optimization. The starting point is
*Computational Lithography* by Xu Ma and Gonzalo R. Arce.

The source is fixed. Optimization changes a real-valued mask amplitude to
minimize the sum of squared differences between aerial and target intensity.
Mask amplitudes are unconstrained and can be negative or exceed one. This is
a relaxed inverse problem, not a binary-mask synthesis or resist simulation.

## Run

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python run_mask_optimization.py --iterations 500
```

The installed `comp-litho` command runs the same example. The original
`python Run_MO_CTM_class_based.py` command remains available.
Imports do not start optimization or open figures.

The example starts from a 60 by 60 pixel square, equivalent to the original
four touching squares. Default parameters are 193 nm wavelength, NA 1.25,
2 nm pixels, a 231 by 231 image, a 41 by 41 kernel, and an 18 by 18 source
lattice with annular sigma from 0.89 to 0.99.

Results are saved in `results/mask_optimization/`:

- `summary.png`: target, initial/final aerial images, final mask, error, and loss.
- `arrays.npz`: input/output arrays and accepted iteration history.
- `parameters.json` and `metrics.json`: settings and numerical results.

Use `--output PATH` to save elsewhere or `--initial-step VALUE` to change the
starting line-search step. No randomness is used. Timing varies by machine.

## Model conventions

- All lengths are in nm. Array axes are `(y, x)`.
- Coordinates use the zero-based midpoint `(size - 1) / 2`.
- Source frequency spacing is `1 / (image_size * pixel_size_nm)` cycles/nm.
  Sigma is radial source frequency divided by `NA / wavelength_nm`. Phase
  ramps use `exp(2 pi i (fy y + fx x))`, consistent with that same spacing.
  An even source lattice has half-integer coordinates and no on-axis sample.
- The amplitude response is a sampled `J1(u)/u`, with its limit of 1/2 at
  the origin. It is truncated to a circular spatial support and normalized
  to have unit sum. Finite support approximates an ideal circular pupil;
  changing kernel size or source sampling can change the physical result.
- Source intensities are nonnegative and normalized to sum to one. A custom
  `source_weights` array can replace the annulus when constructing a model.
- Convolutions are linear, zero padded, and cropped to the input shape.
  The exterior of the simulated mask is opaque; images are not periodic.
- Optical kernels and their conjugate-reversed adjoints are cached. Models
  and cached arrays are immutable; construct a new model to change optics.

For source weights `w`, fields `u = K m`, and intensity `I = sum(w |u|^2)`,
the loss is `L = sum((I - target)^2)` and its real-mask derivative is
`4 sum(w Re(K^H ((I - target) u)))`. Here `K^H` uses the conjugate-reversed
full complex kernel, including its illumination phase.

## Optimization and tests

The optimizer uses deterministic Armijo backtracking. A trial mask, loss,
and gradient are accepted together. A failed line search retains the last
accepted iterate and reports failure. Other statuses distinguish gradient
tolerance, relative loss tolerance, and the iteration limit. Relative loss
tolerance means objective stability; it does not prove a stationary point.

Run the tests with Python's standard-library test runner:

```bash
python -m unittest discover -s tests -v
```

Tests compare FFT imaging against direct convolution, verify adjoint and
finite-difference derivatives (including asymmetric illumination and image
boundaries), and exercise rejected steps, failure reporting, repeatability,
and a short imaging optimization.

## Files

- `imaging.py`: immutable optical model, aerial image, and analytical gradient.
- `optimization.py`: backtracking gradient descent and optimization history.
- `run_mask_optimization.py`: example, command-line arguments, and saved results.
- `tests/`: numerical correctness and optimizer regression tests.

Compared with the earlier code, the array midpoints, response at zero, source
phase spacing, and adjoint gradient have been corrected. The source phase
spacing previously corresponded to half the sigma used to select the annulus.
Results therefore need not match the earlier implementation.


