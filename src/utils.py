import numpy as np

# array smoothing function
def smooting_kernel(kernel_size, std_dev=2):
    # type: (int, float) -> np.ndarray
    """
    :param kernel_size: size of the smoothing kernel
    :param std_dev: gaussian sigma value
    :return: a smooting kernel with the requested parameters
    """
    x = np.arange(-kernel_size, kernel_size)
    x_1 = -0.5 * ((x - 0) / std_dev) ** 2
    x_2 = (std_dev * np.sqrt(2 * np.pi))
    kernel = np.exp(x_1) / x_2
    kernel = np.exp(kernel - kernel.max())
    return kernel / kernel.sum()


def smooth(points, kernel_size, dimensions=2):
    # type: (np.ndarray, int, int) -> np.ndarray
    """
    :param points: array you want to smoot
    :param kernel_size: size of the smooting kernel; it must be an odd number
    :param dimensions: points dimensions (2 --> 2D point, 3 --> 3D point, ...)
    :return: smoothed version of the input array
    >> NOTE: this is NOT an inplace operation
    """

    assert kernel_size % 2 == 1, '`kernel_size` must be an odd number'
    assert kernel_size > 0, '`kernel_size` must be > 0'

    # copy input array
    t = points.copy()

    # generate smooting kernel
    kernel_size = kernel_size // 2 + 1
    kernel = smooting_kernel(kernel_size=kernel_size)

    k = []
    for i in range(dimensions):
        k.append(kernel)
    kernel = np.array(k).T

    # smoot the copy of the input array
    # NOTE: this is sliding window operation with non-optimized performance!
    for i in range(0, len(t)):
        # >> left edge of the sequence
        if i - kernel_size < 0:
            t[i] = t[:i + kernel_size].mean(0)
        # >> right edge of the sequence
        elif i + kernel_size > len(t):
            t[i] = t[i - kernel_size:].mean(0)
        # >> all the other sequence values
        else:
            t[i] = np.sum(t[i - kernel_size:i + kernel_size] * kernel, 0)  # [:,:,np.newaxis], 0)

    # return smoothed version of the input array
    return t