import numpy
import six


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of convolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_deconv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of convolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the convolution operation.

    """
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def im2col_cpu(
        img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    return col


def col2im_cpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]
