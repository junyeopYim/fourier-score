"""Output parameterizations for a shared DSM loss; no training dependencies."""

GAUSSIAN_OBJECTIVES = {
    'scalar_gaussian': {'covariance': 'scalar', 'scale_residual': True},
    'fourier_gaussian_unscaled': {'covariance': 'fourier', 'scale_residual': False},
    'fourier_gaussian': {'covariance': 'fourier', 'scale_residual': True},
}
OBJECTIVES = ('score', 'diffusion', *GAUSSIAN_OBJECTIVES)
# Score and diffusion are sign conventions, so only one is in the main ablation.
COMPARISON_OBJECTIVES = ('score', *GAUSSIAN_OBJECTIVES)
