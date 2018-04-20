/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


%rename (convolution2DParam) conv_param_t;
struct conv_param_t {
    std::vector<int> out_dims;
    int kh, kw; // kernel size
    int dilate_y = 0, dilate_x = 0; // in MKL-DNN, common conv is treated as 0 dilate
    int sy, sx; // stride
    int pad_lh, pad_lw, pad_rh, pad_rw; //padding
};

%rename (pooling2DParam) pooling_param_t;
struct pooling_param_t {
    std::vector<int> out_dims;
    int kh, kw; // kernel size
    int sy, sx; // stride
    int pad_lh, pad_lw, pad_rh, pad_rw; //padding

    enum algorithm {
        pooling_max,
        pooling_avg,
        pooling_avg_include_padding,
        pooling_avg_exclude_padding,
    } algo_kind;
};

%rename (localResponseNormalizationParam) lrn_param_t;
struct lrn_param_t {
    int n; // local size
    double k;
    double alpha;
    double beta;

    enum algorithm {
        lrn_across_channels,
        lrn_within_channel,
    } algo_kind;
};
