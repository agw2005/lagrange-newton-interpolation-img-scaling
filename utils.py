import numpy as np
import cv2

def lagrange_interpolate(x_values, y_values, x):
    result = 0
    n = len(x_values)
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

def newton_divided_diff(x_values, y_values):
    n = len(x_values)
    coef = np.copy(y_values).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_values[i] - x_values[i - j])
    return coef

def newton_interpolate(x_values, coef, x):
    n = len(x_values)
    result = coef[0]
    term = 1.0
    for i in range(1, n):
        term *= (x - x_values[i - 1])
        result += coef[i] * term
    return result

def get_neighbors(x_values, y_values, x, k=4):
    distances = np.abs(np.array(x_values) - x)
    indices = np.argsort(distances)[:k]
    indices = sorted(indices)
    return [x_values[i] for i in indices], [y_values[i] for i in indices]

def lagrange_local_interpolate(x_values, y_values, x, k=4):
    xs, ys = get_neighbors(x_values, y_values, x, k)
    return lagrange_interpolate(xs, ys, x)

def newton_local_interpolate(x_values, y_values, x, k=4):
    xs, ys = get_neighbors(x_values, y_values, x, k)
    coef = newton_divided_diff(xs, ys)
    return newton_interpolate(xs, coef, x)

def upscale_image(image, method='lagrange', k=4):
    if image.ndim != 2:
        raise ValueError("Only grayscale images are supported.")
        
    orig_h, orig_w = image.shape
    new_h = 2 * (orig_h - 1) + 1
    new_w = 2 * (orig_w - 1) + 1
    upscaled = np.zeros((new_h, new_w))

    x_old = list(range(orig_w))
    y_old = list(range(orig_h))
    x_new = np.linspace(0, orig_w - 1, new_w)
    y_new = np.linspace(0, orig_h - 1, new_h)

    # Horizontal interpolation
    row_interpolated = np.zeros((orig_h, new_w))
    for i in range(orig_h):
        for j, x in enumerate(x_new):
            if method == 'lagrange':
                row_interpolated[i][j] = lagrange_local_interpolate(x_old, image[i], x, k)
            elif method == 'newton':
                row_interpolated[i][j] = newton_local_interpolate(x_old, image[i], x, k)

    # Vertical interpolation
    for j in range(new_w):
        col = row_interpolated[:, j]
        for i, y in enumerate(y_new):
            if method == 'lagrange':
                upscaled[i][j] = lagrange_local_interpolate(y_old, col, y, k)
            elif method == 'newton':
                upscaled[i][j] = newton_local_interpolate(y_old, col, y, k)

    return np.clip(upscaled, 0, 255).astype(np.uint8)
