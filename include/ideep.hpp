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

#ifndef _IDEEP_HPP
#define _IDEEP_HPP

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "ideep/abstract_types.hpp"
#include "ideep/tensor.hpp"
#include "ideep/computations.hpp"

// The ideep version number has four segments
// The first three are the same as oneDNN version numbers
// The fourth is to indicate ideep API change
// So, ideep version = MAJOR.MINOR.PATCH.REVISION
// e.g., 3.1.0.0
#define IDEEP_VERSION_MAJOR    DNNL_VERSION_MAJOR
#define IDEEP_VERSION_MINOR    DNNL_VERSION_MINOR
#define IDEEP_VERSION_PATCH    DNNL_VERSION_PATCH
#define IDEEP_VERSION_REVISION 1

#endif
