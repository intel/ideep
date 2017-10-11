def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def get_deconv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return s * (size - 1) + k - s + 1 - 2 * p
    else:
        return s * (size - 1) + k - 2 * p


class conv_geometry(object):
    def __init__(self, x_shape, W_shape, stride, pad, cover_all):
        assert isinstance(x_shape, tuple), 'X shape must be tuple'
        assert isinstance(W_shape, tuple), 'W shape must be tuple'

        sy, sx = _pair(stride)
        p_upper, p_left = _pair(pad)

        out_c, _, kh, kw = W_shape
        n, c, h, w = x_shape

        out_h = get_conv_outsize(h, kh, sy, p_upper, cover_all=cover_all)
        assert out_h > 0, 'Height in the output should be positive.'
        out_w = get_conv_outsize(w, kw, sx, p_left, cover_all=cover_all)
        assert out_w > 0, 'Width in the output should be positive.'

        p_down = sy * (out_h - 1) + kh - h - p_upper
        p_right = sx * (out_w - 1) + kw - w - p_left

        self.p_upper = p_upper
        self.p_left = p_left
        self.p_down = p_down
        self.p_right = p_right
        self.out_h = out_h
        self.out_w = out_w

        self._out_shape = (n, out_c, out_h, out_w)
        self._geometry = (_pair(stride), (p_upper, p_left), (p_down, p_right))

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def geometry(self):
        return self._geometry


class deconv_geometry(object):
    def __init__(
        self, gy_shape, W_shape, stride, pad, outsize, cover_all=False):

        assert isinstance(gy_shape, tuple), 'X shape must be tuple'
        assert isinstance(W_shape, tuple), 'W shape must be tuple'

        sy, sx = _pair(stride)
        p_upper, p_left = _pair(pad)

        _, c, kh, kw = W_shape
        n, out_c, out_h, out_w = gy_shape

        h, w = outsize
        if h is None:
            h = get_deconv_outsize(out_h, kh, sy, p_upper)
            assert w > 0, 'Height in the output should be positive.'
        if w is None:
            w = get_deconv_outsize(out_w, kw, sx, p_left)
            assert h > 0, 'Width in the output should be positive.'

        p_down = sy * (out_h - 1) + kh - h - p_upper
        p_right = sx * (out_w - 1) + kw - w - p_left

        self.p_upper = p_upper
        self.p_left = p_left
        self.p_down = p_down
        self.p_right = p_right
        self.out_h = out_h
        self.out_w = out_w

        self._in_shape = (n, c, h, w)
        self._geometry = (_pair(stride), (p_upper, p_left), (p_down, p_right))

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def geometry(self):
        return self._geometry
