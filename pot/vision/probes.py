import numpy as np


def render_sine_grating(
    H: int,
    W: int,
    freq: float,
    theta: float,
    phase: float,
    contrast: float,
    seed: int | None = None,
) -> np.ndarray:
    """Render a sine grating with values in ``[0, 1]``.

    Args:
        H, W: Output height and width.
        freq: Spatial frequency of the grating.
        theta: Orientation in radians.
        phase: Phase offset in radians.
        contrast: Multiplicative contrast term (0-1).
        seed: Optional random seed for determinism.

    Returns:
        ``np.ndarray`` of shape ``(H, W)``.
    """

    # Explicitly initialize RNG for determinism (no randomness used).
    np.random.default_rng(seed)

    y, x = np.meshgrid(
        np.linspace(-0.5, 0.5, H, endpoint=False),
        np.linspace(-0.5, 0.5, W, endpoint=False),
        indexing="ij",
    )
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    grating = np.sin(2 * np.pi * freq * x_theta + phase)
    grating = 0.5 + 0.5 * contrast * grating
    return np.clip(grating, 0.0, 1.0)


def render_texture(
    H: int,
    W: int,
    octaves: int = 1,
    scale: float = 8.0,
    texture_type: str = "noise",
    freq: float = 8.0,
    theta: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate procedural textures.

    Args:
        H, W: Output height and width.
        octaves: Number of octaves for Perlin noise.
        scale: Base scale for Perlin noise.
        texture_type: One of ``{"noise", "checkerboard", "perlin"}``.
        freq: Frequency parameter for checkerboard patterns.
        theta: Rotation angle (radians) for checkerboard patterns.
        seed: Optional random seed for determinism.

    Returns:
        ``np.ndarray`` of shape ``(H, W)`` with values in ``[0, 1]``.
    """

    rng = np.random.default_rng(seed)

    if texture_type == "noise":
        tex = rng.random((H, W))
    elif texture_type == "checkerboard":
        y, x = np.meshgrid(
            np.linspace(-0.5, 0.5, H, endpoint=False),
            np.linspace(-0.5, 0.5, W, endpoint=False),
            indexing="ij",
        )
        x_r = x * np.cos(theta) + y * np.sin(theta)
        y_r = -x * np.sin(theta) + y * np.cos(theta)
        tex = ((np.floor(x_r * freq) + np.floor(y_r * freq)) % 2).astype(float)
    elif texture_type == "perlin":
        tex = _perlin_noise(H, W, rng, scale, octaves)
    else:
        raise ValueError(f"Unknown texture_type: {texture_type}")
    return np.clip(tex, 0.0, 1.0)


def _perlin_noise(
    H: int, W: int, rng: np.random.Generator, scale: float, octaves: int
) -> np.ndarray:
    """Simple Perlin-like noise implementation."""

    noise = np.zeros((H, W))
    amplitude = 1.0
    frequency = 1.0 / scale
    max_amp = 0.0

    for _ in range(octaves):
        nx = W * frequency
        ny = H * frequency
        grid = rng.random((int(np.ceil(ny)) + 2, int(np.ceil(nx)) + 2))
        x = np.linspace(0, nx, W, endpoint=False)
        y = np.linspace(0, ny, H, endpoint=False)
        xi = x.astype(int)
        yi = y.astype(int)
        xf = x - xi
        yf = y - yi

        c00 = grid[yi[:, None], xi[None, :]]
        c10 = grid[yi[:, None], xi[None, :] + 1]
        c01 = grid[yi[:, None] + 1, xi[None, :]]
        c11 = grid[yi[:, None] + 1, xi[None, :] + 1]

        u = xf[None, :]
        v = yf[:, None]
        interp = (
            c00 * (1 - u) * (1 - v)
            + c10 * u * (1 - v)
            + c01 * (1 - u) * v
            + c11 * u * v
        )

        noise += interp * amplitude
        max_amp += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    noise /= max_amp
    return noise

