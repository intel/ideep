/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

%module (package="mkldnn.api") c_api
%{
  #include <mkldnn.h>
%}

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stddef.h>
#include <stdint.h>
#endif

/** @addtogroup c_api C API
 *  @{
 *
 *  @addtogroup c_api_types Types
 *  @{
 *
 *  @addtogroup c_api_types_generic Generic
 *  @{ */

/** Status values returned by Intel(R) MKL-DNN functions. */
typedef enum {
    /** The operation was successful */
    mkldnn_success = 0,
    /** The operation failed due to an out-of-memory condition */
    mkldnn_out_of_memory = 1,
    /** The operation failed and should be retried */
    mkldnn_try_again = 2,
    /** The operation failed because of incorrect function arguments  */
    mkldnn_invalid_arguments = 3,
    /** The operation failed because a primitive was not ready for execution */
    mkldnn_not_ready = 4,
    /** The operation failed because requested functionality is not implemented
     */
    mkldnn_unimplemented = 5,
    /** Primitive iterator passed over last primitive descriptor */
    mkldnn_iterator_ends = 6,
    /** Primitive or engine failed on execution */
    mkldnn_runtime_error = 7,
    /** Queried element is not required for given primitive */
    mkldnn_not_required = 8,
} mkldnn_status_t;

/** Data type specification */
typedef enum {
    /** Undefined data type, used for empty memory descriptors. */
    mkldnn_data_type_undef = 0,
    /** 32-bit/single-precision floating point. */
    mkldnn_f32 = 1,
    /** 32-bit signed integer. */
    mkldnn_s32 = 2,
} mkldnn_data_type_t;

/** Memory format specification.
 *
 * Intel(R) MKL-DNN uses the following notation for memory format names:
 *  - @c 'n' denotes the mini-batch dimension
 *  - @c 'c' denotes a channels dimension
 *  - When there are multiple channel dimensions (for example, in convolution
 *    weights tensor), @c 'i' and @c 'o' denote dimensions of input and output
 *    channels
 *  - @c 'h' and @c 'w' denote spatial width and height
 *  - Upper-case letters indicate that the data is laid out in blocks
 *    for a particular dimension. In such cases, the format name contains both
 *    upper- and lower-case letters for that dimension with lower-case letter
 *    preceded by the block size. For example: @c 'mkldnn_nChw8c' describes a
 *    format where the outermost dimension is mini-batch, followed by the
 *    channel block number, followed by the spatial height and width, and
 *    finally followed by 8-element channel blocks.
 *
 * @note
 *    Channel designations can be different. For example: both the @c
 *    'mkldnn_nc' and @c 'mkldnn_io' formats can be used to describe a 2D
 *    tensor.
 */
typedef enum {
    /** Undefined memory format, used for empty memory descriptors. */
    mkldnn_format_undef = 0,
    /** Unspecified format. The primitive selects a format
     * automatically. */
    mkldnn_any,
    /** A tensor in a generic format described by the stride and blocking
     * values in each dimension. See #mkldnn_blocking_desc_t for more
     * information. */
    mkldnn_blocked,
    /** 1D data tensor. */
    mkldnn_x,
    /** 2D data tensor. */
    mkldnn_nc,
    /** 4D data tensor in the @c nchw format typically used in Caffe. */
    mkldnn_nchw,
    /** 4D data tensor in the @c nhwc format typically used in TensorFlow. */
    mkldnn_nhwc,
    /** 4D data tensor in the @c chwn format typically used in Neon. */
    mkldnn_chwn,
    /** 4D data tensor in the @c nchw format with channels data laid out in
     * memory in 8-element blocks. */
    mkldnn_nChw8c,
    /** 4D data tensor in the @c nchw format with channels data laid out in
     * memory in 16-element blocks. */
    mkldnn_nChw16c,
    /** 2D weights tensor in the format (input channels, output channels). */
    mkldnn_oi,
    /** 2D weights tensor in the format (input channels, output channels). */
    mkldnn_io,
    /** 4D weights tensor in the format (input channels, output channels,
     * width, height). */
    mkldnn_oihw,
    /** 4D weights tensor in the format (input channels, height, width,
     * output channels). */
    mkldnn_ihwo,
    /** 4D weights tensor in the format (height, width, input channels,
     * output channels). */
    mkldnn_hwio,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 8-element blocks. */
    mkldnn_OIhw8i8o,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 16-element blocks. */
    mkldnn_OIhw16i16o,
    /** 4D weights tensor in the @c oihw format with output channels data
     * laid out in memory in 16-element blocks and input channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    mkldnn_OIhw8i16o2i,
    /** 4D weights tensor in the @c oihw format with input channels data
     * laid out in memory in 16-element blocks and output channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    mkldnn_OIhw8o16i2o,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 8-element blocks. */
    mkldnn_OIhw8o8i,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 16-element blocks. */
    mkldnn_OIhw16o16i,
    /** 4D weights tensor in the format (output channels, input channels,
     * height, width) with output channels data laid out in memory in 8-element
     * blocks. */
    mkldnn_Oihw8o,
    /** 4D weights tensor in the format (output channels, input channels,
     * height, width) with output channels data laid out in memory in
     * 16-element blocks. */
    mkldnn_Oihw16o,
    /** 4D weights tensor in the format (output channels, width, height, input
     * channels) with output channels data laid out in memory in 8-element
     * blocks. */
    mkldnn_Ohwi8o,
    /** 4D weights tensor in the format (output channels, width, height, input
     * channels) with output channels data laid out in memory in 16-element
     * blocks. */
    mkldnn_Ohwi16o,
    /** 4D weights tensor in the @c oihw format with both input and output
     * channels data laid out in memory in 16-element and 4-element blocks. */
    mkldnn_OhIw16o4i,
    /** 5D weights tensor in the @c oihw format with extra outer dimension for
     * groups. */
    mkldnn_goihw,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 8-element blocks.
     */
    mkldnn_gOIhw8i8o,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 16-element blocks.
     */
    mkldnn_gOIhw16i16o,
    /** 5D weights tensor in the @c oihw format with output channels data
     * laid out in memory in 16-element blocks and input channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    mkldnn_gOIhw8i16o2i,
    /** 5D weights tensor in the @c oihw format with input channels data
     * laid out in memory in 16-element blocks and output channels data
     * laid out in memory in 8-element blocks blocked by pairs. */
    mkldnn_gOIhw8o16i2o,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 8-element blocks.
     */
    mkldnn_gOIhw8o8i,
    /** 5D weights tensor in the blocked version of @c goihw format with both
     * input and output channels data laid out in memory in 16-element blocks.
     */
    mkldnn_gOIhw16o16i,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 8-element blocks. */
    mkldnn_gOihw8o,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 16-element blocks. */
    mkldnn_gOihw16o,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 8-element blocks. */
    mkldnn_gOhwi8o,
    /** 5D weights tensor in the blocked version of @c goihw format with output
     * channels data laid out in memory in 16-element blocks. */
    mkldnn_gOhwi16o,
    /** 5D weights tensor in the @c goihw format with both input and output
     * channels data laid out in memory in 16-element and 4-element blocks. */
    mkldnn_gOhIw16o4i,
    /** 4D weights tensor in the oihw format with input channels data laid out
     * in memory in 8-element blocks. */
    mkldnn_oIhw8i = mkldnn_nChw8c,
    /** 4D weights tensor in the oihw format with input channels data laid out
     * in memory in 16-element blocks. */
    mkldnn_oIhw16i = mkldnn_nChw16c,
} mkldnn_memory_format_t;

/** Kinds of padding. Define how to interpret the data in padding regions. */
typedef enum {
    /** The data in padding regions is zero. */
    mkldnn_padding_zero,
} mkldnn_padding_kind_t;

/** Kinds of propagation. */
typedef enum {
    /* TODO: suggest renames */
    /** Undefined propagation type. */
    mkldnn_prop_kind_undef = 0,
    /** Forward data propagation (training mode). In this mode primitives
     * perform computations necessary for subsequent backward propagation. */
    mkldnn_forward_training = 64,
    /** Forward data propagation (inference mode). In this mode primitives only
     * perform computations that are necessary for inference and omit
     * computations that are only necessary for backward propagation. */
    mkldnn_forward_inference = 96,
    /** Forward data propagation (alias for @c mkldnn_forward_inference) */
    mkldnn_forward_scoring = mkldnn_forward_inference,
   /** Forward data propagation (alias for @c mkldnn_forward_training) */
    mkldnn_forward = mkldnn_forward_training,
    /** Backward propagation (with respect to all parameters */
    mkldnn_backward = 128,
    /** Backward data propagation */
    mkldnn_backward_data = 160,
    /** Backward weights propagation */
    mkldnn_backward_weights = 192,
    /** Backward bias propagation */
    mkldnn_backward_bias = 193,
} mkldnn_prop_kind_t;

/** Kinds of primitives. Used to implement a way to extend the library with new
 * primitives without changing the ABI. */
typedef enum {
    /** Undefined primitive (XXX: why do we have it?). */
    mkldnn_undefined_primitive,
    /** A memory primitive. */
    mkldnn_memory,
    /** A view primitive. */
    mkldnn_view,
    /** A reorder primitive.*/
    mkldnn_reorder,
    /** A (out-of-place) concat primitive. */
    mkldnn_concat,
    /** A (in-place) concat primitive. */
    mkldnn_concat_inplace,
    /** A sum primitive. */
    mkldnn_sum,
    /** A convolution primitive. */
    mkldnn_convolution,
    /** A ReLU primitive. */
    mkldnn_relu,
    /** A Softmax primitive. */
    mkldnn_softmax,
    /** A pooling primitive. */
    mkldnn_pooling,
    /** An LRN primitive. */
    mkldnn_lrn,
    /** An batch normalization primitive. */
    mkldnn_batch_normalization,
    /** An inner product primitive. */
    mkldnn_inner_product,
    /** A convolution primitive merged with relu */
    mkldnn_convolution_relu,
} mkldnn_primitive_kind_t;

/** Kinds of algorithms. */
typedef enum {
    /** Direct convolution */
    mkldnn_convolution_direct = 1,
    /** Winograd convolution */
    mkldnn_convolution_winograd = 2,
    /** Eltwise: ReLU */
    mkldnn_eltwise_relu = 8,
    /** Max pooling */
    mkldnn_pooling_max = 34,
    /** Average pooling include padding */
    mkldnn_pooling_avg_include_padding = 40,
    /** Average pooling exclude padding */
    mkldnn_pooling_avg_exclude_padding = 41,
    mkldnn_pooling_avg = mkldnn_pooling_avg_exclude_padding,
    /** Local response normalization (LRN) across multiple channels */
    mkldnn_lrn_across_channels = 65,
    /** LRN within a single channel */
    mkldnn_lrn_within_channel = 66,
} mkldnn_alg_kind_t;

/** Flags for batch-normalization primititve. */
typedef enum {
    /** Use global statistics
     *
     * If specified
     *  - on forward propagation use mean and variance provided by user (input)
     *  - on backward propagation reduces the amount of computations, since
     *    mean and variance are considered as constants
     *
     *  If not specified:
     *   - on forward propagation mean and variance are computed and stored in
     *     output
     *   - on backward propagation compute full derivative wrt to data
     */
    mkldnn_use_global_stats = 0x1U,
    /** Use scale and shift parameters
     *
     * If specified:
     *  - on forward propagation use scale and shift (aka scale and bias) for
     *    the batch normalization results
     *  - on backward propagation (for prop_kind == #mkldnn_backward) compute
     *    diff wrt to scale and shift (hence one extra output used)
     *
     * If no specified:
     *  - on backward propagation prop_kind == #mkldnn_backward_data has the
     *    same behavior as prop_kind == #mkldnn_backward
     */
    mkldnn_use_scaleshift = 0x2U,
    /** Omit statistics
     *
     * @warning: deprecated, use #mkldnn_use_global_stats instead
     *
     * For time being had an affect on backward propagation only which allowed
     * skipping some computations (the same semantics as
     * #mkldnn_use_global_stats)
     */
    mkldnn_omit_stats = mkldnn_use_global_stats,
} mkldnn_batch_normalization_flag_t;

/** @} */

/** @addtogroup c_api_types_memory Auxiliary types for memory description
 *  @{ */

/** Maximum number of dimensions a tensor can have. Only restricts the amount
 * of space used for the tensor description. Individual computational
 * primitives may support only tensors of certain dimensions. */
#define TENSOR_MAX_DIMS 12

/** A type to describe tensor dimensions. */
typedef int mkldnn_dims_t[TENSOR_MAX_DIMS];
/** A type to describe strides within a tensor. */
typedef ptrdiff_t mkldnn_strides_t[TENSOR_MAX_DIMS];

/** Generic description of blocked data layout for most memory formats. */
typedef struct {
    /** Block size for each of the dimensions. */
    mkldnn_dims_t block_dims;
    /** strides[0]: stride between the first elements of adjacent blocks.
     * @n strides[1]: strides between elements in the same block. */
    mkldnn_strides_t strides[2];
    /** Size of the data including padding in each dimension. */
    mkldnn_dims_t padding_dims;
    /** Per-dimension offset from the padding to actual data, the top-level
     * tensor with offsets applied must lie within the padding area. */
    mkldnn_dims_t offset_padding_to_data;
    /** Offset from memory origin to the current block, non-zero only in
     * a description of a memory sub-block. */
    ptrdiff_t offset_padding;
} mkldnn_blocking_desc_t;

/** @addtogroup c_api_types_op_descs Operation descriptors
 *  @{*/

/** A pointer to any of the operation descriptors. */
typedef void *mkldnn_op_desc_t;
/** A pointer to any of the operation descriptors (constant variant). */
typedef const void *const_mkldnn_op_desc_t;

/** Memory descriptor. The description is based on a number of dimensions,
 * dimensions themselves, plus information about elements type and memory
 * format. Additionally, contains format-specific descriptions of the data
 * layout. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_memory. */
    mkldnn_primitive_kind_t primitive_kind;
    /** Number of dimensions */
    int ndims;
    /** Dimensions in the following order: mini-batch, channel, spatial. For
     * example: <code>{N, C, H, W}</code>. */
    mkldnn_dims_t dims;
    /** Data type of the tensor elements. */
    mkldnn_data_type_t data_type;
    /** Memory format. */
    mkldnn_memory_format_t format;
    union {
        /** Description of the data layout for memory formats that use
         * blocking. */
        mkldnn_blocking_desc_t blocking;
        /* ... other descriptions possible */
    } layout_desc;
} mkldnn_memory_desc_t;

/** @} */

/** A descriptor of a convolution operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_convolution. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward_data,
     * #mkldnn_backward_weights, and #mkldnn_backward_bias. */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of the convolution algorithm. Possible values:
     * #mkldnn_convolution_direct. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Weights memory descriptor. */
    mkldnn_memory_desc_t weights_desc;
    /** Weights gradient memory descriptor. */
    mkldnn_memory_desc_t diff_weights_desc;
    /** Bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Bias gradient memory descriptor. */
    mkldnn_memory_desc_t diff_bias_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** Convolution strides in each spatial dimension. */
    mkldnn_dims_t strides;
    /** Padding in each spatial dimension. padding[0] is a padding in the
     * beginning (@p padding_l), padding[1] is a padding in the end (@p
     * padding_r). */
    mkldnn_dims_t padding[2];
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
} mkldnn_convolution_desc_t;

/** A descriptor of a element-wise operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_relu. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of eltwise algorithm. Possible values: #mkldnn_eltwise_relu */
    mkldnn_alg_kind_t alg_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_data_desc;
    /** Algorithm specific parameter.
     * Accordance table:
     *  - #mkldnn_eltwise_relu: @p alpha -- negative slope, @p beta ignored
     */
    double alpha, beta;
    /** Scaling factor for negative values. Stored as double-precision, but
     * interpreted in a way specific to the data type in each implementation.
     * @deprecated: for ReLU use alpha instead
     * @warning: read-only value */
    double negative_slope;
} mkldnn_eltwise_desc_t;

/* @depracated: use mkldnn_eltwise_desc_t */
typedef mkldnn_eltwise_desc_t mkldnn_relu_desc_t;

/** A descriptor of a Softmax operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
    * descriptor. Must be #mkldnn_softmax. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference. */
    mkldnn_prop_kind_t prop_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** The axis along which to perform the softmax. */
    int softmax_axis;
} mkldnn_softmax_desc_t;

/** A descriptor of a pooling operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_pooling. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** The kind of pooling algorithm. Possible values: #mkldnn_pooling_max,
     * #mkldnn_pooling_avg. */
    mkldnn_alg_kind_t alg_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** Pooling kernel strides for spatial dimensions. */
    mkldnn_dims_t strides;
    /** Pooling kernel spatial dimensions. */
    mkldnn_dims_t kernel;
    /** Padding in each spatial dimension. padding[0] is a padding in the
     * beginning (@p padding_l), padding[1] is a padding in the end (@p
     * padding_r). */
    mkldnn_dims_t padding[2];
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
} mkldnn_pooling_desc_t;

/** A descriptor of a Local Response Normalization (LRN) operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_lrn. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** LRN algorithm. Possible values #mkldnn_lrn_within_channel or
     * #mkldnn_lrn_across_channels. */
    mkldnn_alg_kind_t alg_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_data_desc;
    /** The number of channels to sum over (for cross-channel LRN) or the side
     * length of the square region to sum over (for within-channel LRN). */
    int local_size;
    /** LRN alpha parameter. */
    double lrn_alpha;
    /** LRN beta parameter. */
    double lrn_beta;
    /** LRN k parameter. */
    double lrn_k;
} mkldnn_lrn_desc_t;

/** A descriptor of a Batch Normalization operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_batch_normalization. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** Source and destination memory descriptor. */
    mkldnn_memory_desc_t data_desc;
    /** Source and destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_data_desc;
    /** Scale and shift data and gradient memory descriptors.
     *
     * Scaleshift memory descriptor uses 2D #mkldnn_nc format[2,Channels]. 1-st
     * dimension contains gamma parameter, 2-nd dimension contains beta
     * parameter. */
    mkldnn_memory_desc_t data_scaleshift_desc;
    mkldnn_memory_desc_t diff_data_scaleshift_desc;
    /** Mean and variance data memory descriptors.
     *
     * Mean and variance memory descriptors use 1D #mkldnn_x format[Channels].
     */
    mkldnn_memory_desc_t mean_desc;
    mkldnn_memory_desc_t variance_desc;
    /** Batch normalization epsilon parameter. */
    double batch_norm_epsilon;
    unsigned flags;
} mkldnn_batch_normalization_desc_t;

/** A descriptor of an inner product operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_inner_product. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward_data,
     * #mkldnn_backward_weights, and #mkldnn_backward_bias. */
    mkldnn_prop_kind_t prop_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Weights memory descriptor. */
    mkldnn_memory_desc_t weights_desc;
    /** Weights gradient memory descriptor. */
    mkldnn_memory_desc_t diff_weights_desc;
    /** Bias memory descriptor. */
    mkldnn_memory_desc_t bias_desc;
    /** Bias gradient memory descriptor. */
    mkldnn_memory_desc_t diff_bias_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
} mkldnn_inner_product_desc_t;

/** A descriptor of a convolution followed by relu operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_convolution_relu. */
    mkldnn_primitive_kind_t primitive_kind;
    /** A descriptor of a convolution operation. */
    mkldnn_convolution_desc_t convolution_desc;
    /** Scaling factor for negative values, stored as double-precision but
     * interpreted in a way specific to the data type in each implementation */
    double negative_slope;
} mkldnn_convolution_relu_desc_t;

/** @} */

/** @addtogroup c_api_engine_types Engine
 * @{ */

/** @brief Kinds of engines. */
typedef enum {
    /** An unspecified engine. */
    mkldnn_any_engine,
    /** CPU engine. */
    mkldnn_cpu,
} mkldnn_engine_kind_t;

/** @struct mkldnn_engine
 * @brief An opaque structure to describe an engine. */
struct mkldnn_engine;
/** @brief An engine handle. */
typedef struct mkldnn_engine *mkldnn_engine_t;
#if 0
/* FIXME: looks like this never happens */
/** @brief A constant engine handle. */
typedef const struct mkldnn_engine *const_mkldnn_engine_t;
#endif

/** @} */

/** @addtogroup c_api_primitive_desc_iterators Primitive descriptor iterators
 * @{ */

/** @struct mkldnn_primitive_desc_iterator
 * @brief An opaque structure to describe a primitive descriptor iterator . */
struct mkldnn_primitive_desc_iterator;

/** @brief A primitive descriptor iterator handle. */
typedef struct mkldnn_primitive_desc_iterator
    *mkldnn_primitive_desc_iterator_t;

/** @brief A constant primitive descriptor iterator handle. */
typedef const struct mkldnn_primitive_desc_iterator
    *const_mkldnn_primitive_desc_iterator_t;

/** @} */

/** @addtogroup c_api_primitive_descs Primitive descriptors
 * @{ */

/** @struct mkldnn_primitive_desc
 * @brief An opaque structure to describe a primitive descriptor . */
struct mkldnn_primitive_desc;

/** @brief A primitive descriptor handle. */
typedef struct mkldnn_primitive_desc *mkldnn_primitive_desc_t;

/** @brief A constant primitive descriptor handle. */
typedef const struct mkldnn_primitive_desc *const_mkldnn_primitive_desc_t;

/** @} */

/** @addtogroup c_api_types_primitive Primitive
 * @{ */

/** @struct mkldnn_primitive
 * An opaque structure to describe a primitive. */
struct mkldnn_primitive;
/** A primitive handle. */
typedef struct mkldnn_primitive *mkldnn_primitive_t;
/** A constant primitive handle. */
typedef const struct mkldnn_primitive *const_mkldnn_primitive_t;

/** A wrapper structure to specify a particular output of a primitive. */
typedef struct {
    /** Primitive to specify the output for. */
    const_mkldnn_primitive_t primitive;
    /** Desired output index. */
    size_t output_index;
} mkldnn_primitive_at_t;

/** @} */

/** @addtogroup c_api_types_query Queries
 * @{ */

/** Primitive descriptor query specification
 *
 * For generic function mkldnn_primitive_desc_query() the type of result must
 * be agreed with queried argument. The correspondence table:
 *      Query                        | type of result
 *      --------------------------------------------------------------
 *      #mkldnn_query_engine         | mkldnn_engine_t *
 *      #mkldnn_query_primitive_kind | mkldnn_primitive_kind_t *
 *      *_s32                        | int *
 *      *_s64                        | ptrdiff_t *
 *      *_f64                        | double *
 *      *_md                         | const mkldnn_memory_desc_t **
 *      *_${op}_d                    | const mkldnn_${op}_desc_t **
 *      *_pd                         | const_mkldnn_primitive_desc_t *
 *
 * @note
 *     Rule of thumb: all opaque types and structures are returned by
 *     reference. All numbers are returned by value.
 *
 * @warning
 *     All returned references point to constant objects and valid only during
 *     the lifetime of queried primitive descriptor. Returned objects must not
 *     be destroyed by user. If there is a need to keep the object longer than
 *     a lifetime of queried primitive descriptor use
 *     mkldnn_primitive_desc_clone() to make a copy. */
typedef enum {
    mkldnn_query_undef = 0,  /**< no query */

    mkldnn_query_engine, /**< execution engine */
    mkldnn_query_primitive_kind, /**< primitive kind */

    mkldnn_query_num_of_inputs_s32, /**< number of inputs expected */
    mkldnn_query_num_of_outputs_s32, /**< number of outputs expected */

    mkldnn_query_time_estimate_f64, /**< runtime estimation (seconds) */
    mkldnn_query_memory_consumption_s64, /**< memory consumption -- extra
                                           (scratch) memory, additional to all
                                           inputs and outputs memory (bytes) */

    /* memory and op descriptor section */
    mkldnn_query_some_d = 64, /**< stub */
    mkldnn_query_memory_d, /**< memory descriptor for memory and view */
    mkldnn_query_convolution_d, /**< convolution descriptor */
    mkldnn_query_eltwise_d, /**< eltwise descriptor */
    mkldnn_query_relu_d = mkldnn_query_eltwise_d, /**< @deprecated */
    mkldnn_query_softmax_d, /**< softmax descriptor */
    mkldnn_query_pooling_d, /**< pooling descriptor */
    mkldnn_query_lrn_d, /**< lrn descriptor */
    mkldnn_query_batch_normalization_d, /**< batch normalization descriptor */
    mkldnn_query_inner_product_d, /**< inner product descriptor */
    mkldnn_query_convolution_relu_d, /**< convolution-relu descriptor */

    /* (memory) primitive descriptor section */
    mkldnn_query_some_pd = 128, /**< stub */
    mkldnn_query_input_pd, /**< input memory primitive desc */
    mkldnn_query_output_pd, /**< output memory primitive desc */
    mkldnn_query_src_pd, /**< source memory primitive desc */
    mkldnn_query_diff_src_pd, /**< source gradient memory primitive desc */
    mkldnn_query_weights_pd, /**< weights memory primitive descriptor desc */
    mkldnn_query_diff_weights_pd, /**< weights grad. memory primitive desc */
    mkldnn_query_dst_pd, /**< destination memory primitive desc */
    mkldnn_query_diff_dst_pd, /**< destination grad. memory primitive desc */
    mkldnn_query_workspace_pd, /**< workspace memory primitive desc */
} mkldnn_query_t;

/** @} */

/** @addtogroup c_api_types_stream Execution stream
 * @{ */

/** @brief Kinds of streams. */
typedef enum {
    /** An unspecified engine. */
    mkldnn_any_stream,
    /** Eager stream. */
    mkldnn_eager,
    /** Lazy stream. */
    mkldnn_lazy,
} mkldnn_stream_kind_t;

/** @struct mkldnn_stream
 * An opaque structure to describe an execution stream. */
struct mkldnn_stream;
/** An execution stream handle. */
typedef struct mkldnn_stream *mkldnn_stream_t;
/** A constant execution stream handle. */
typedef const struct mkldnn_stream *const_mkldnn_stream_t;

#ifdef __cplusplus
}
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* All symbols shall be internal unless marked as MKLDNN_API */
#if defined _WIN32 || defined __CYGWIN__
#   define MKLDNN_HELPER_DLL_IMPORT __declspec(dllimport)
#   define MKLDNN_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#   if __GNUC__ >= 4
#       define MKLDNN_HELPER_DLL_IMPORT __attribute__ ((visibility ("default")))
#       define MKLDNN_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#   else
#       define MKLDNN_HELPER_DLL_IMPORT
#       define MKLDNN_HELPER_DLL_EXPORT
#   endif
#endif

#ifdef MKLDNN_DLL
#   ifdef MKLDNN_DLL_EXPORTS
#       define MKLDNN_API MKLDNN_HELPER_DLL_EXPORT
#   else
#       define MKLDNN_API MKLDNN_HELPER_DLL_IMPORT
#   endif
#else
#   define MKLDNN_API
#endif

#include "mkldnn_types.h"
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/** @addtogroup c_api C API
 *  @{ */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup c_api_primitive Primitive operations
 * @{ */

/** @addtogroup c_api_primitive_common Common primitive operations
 * @{ */

/** Creates a primitive descriptor @p iterator for given @p op_desc, @p engine,
 * and optionally a hint primitive descriptor from forward propagation
 * (required for backward propagation). Pass @c NULL for forward propagation.
 */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_iterator_create(
        mkldnn_primitive_desc_iterator_t *iterator,
        /*const_*/mkldnn_op_desc_t op_desc, mkldnn_engine_t engine,
        const_mkldnn_primitive_desc_t hint_forward_primitive_desc);

/** Iterates over primitive descriptors. Returns #mkldnn_iterator_ends if no
 * more primitive descriptors are available */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_iterator_next(
        mkldnn_primitive_desc_iterator_t iterator);

/** Fetches current primitive descriptor.
 *
 * @note
 *     fetched primitive descriptor should be deleted by user using
 *     mkldnn_primitive_desc_destroy() once becomes unneeded */
mkldnn_primitive_desc_t MKLDNN_API mkldnn_primitive_desc_iterator_fetch(
        const_mkldnn_primitive_desc_iterator_t iterator);

/** Deletes a primitive descriptor @p iterator */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_iterator_destroy(
        mkldnn_primitive_desc_iterator_t iterator);

/** Creates a @p primitive_desc using @p op_desc, @p engine, and optionally a
 * hint primitive descriptor from forward propagation. The call is equivalent
 * to create a primitive descriptor iterator, instantly fetch a primitive_desc
 * and destroy the iterator. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_create(
        mkldnn_primitive_desc_t *primitive_desc,
        /*const_*/mkldnn_op_desc_t op_desc, mkldnn_engine_t engine,
        const_mkldnn_primitive_desc_t hint_forward_primitive_desc);

/** Makes a copy of a @p primitive_desc. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_clone(
        mkldnn_primitive_desc_t *primitive_desc,
        const_mkldnn_primitive_desc_t existing_primitive_desc);

/** Deletes a @p primitive_desc. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_destroy(
        mkldnn_primitive_desc_t primitive_desc);

/** Queries primitive descriptor
 *
 * @sa mkldnn_query_t */
mkldnn_status_t MKLDNN_API mkldnn_primitive_desc_query(
        const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
        int index, void *result);

/** Queries primitive descriptor for memory descriptor
 *
 * @returns NULL in case of any error */
const mkldnn_memory_desc_t MKLDNN_API *mkldnn_primitive_desc_query_memory_d(
        const_mkldnn_primitive_desc_t primitive_desc);

/** Queries primitive descriptor for primitive descriptor
 *
 * @returns NULL in case of any error */
const_mkldnn_primitive_desc_t MKLDNN_API mkldnn_primitive_desc_query_pd(
        const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
        int index);

/** Queries primitive descriptor for signed 32bit int
 *
 * @returns 0 in case of any error */
int MKLDNN_API mkldnn_primitive_desc_query_s32(
        const_mkldnn_primitive_desc_t primitive_desc, mkldnn_query_t what,
        int index);

/** Creates a @p primitive using a @p primitive_desc descriptor and arrays of
 * @p inputs and @p outputs. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_create(
        mkldnn_primitive_t *primitive,
        const_mkldnn_primitive_desc_t primitive_desc,
        const mkldnn_primitive_at_t *inputs,
        const_mkldnn_primitive_t *outputs);

/** Retrieves a reference to the @p primitive_desc descriptor of given @p
 * primitive.
 *
 * @warning
 *     Returned object must not be destroyed by user. 'const' qualifier of the
 *     returned object prevents such attempts. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_primitive_desc(
        const_mkldnn_primitive_t primitive,
        const_mkldnn_primitive_desc_t *primitive_desc);

/** For a @p primitive, returns @p input at the @p index position. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_input_at(
        const_mkldnn_primitive_t primitive, size_t index,
        mkldnn_primitive_at_t *input);

/** For a @p primitive, returns @p output at the @p index position. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_get_output(
        const_mkldnn_primitive_t primitive, size_t index,
        const_mkldnn_primitive_t *output);

/** Deletes a @p primitive. */
mkldnn_status_t MKLDNN_API mkldnn_primitive_destroy(
        mkldnn_primitive_t primitive);

/** Creates an #mkldnn_primitive_at_t structure from a @p primitive and @p
 * output_index. This function only fills in the data structure
 * and does not check whether parameters are correct. The actual error checking
 * is done when the resulting #mkldnn_primitive_at structure is passed to a
 * primitive creation function. */
mkldnn_primitive_at_t MKLDNN_API mkldnn_primitive_at(
        const_mkldnn_primitive_t primitive, size_t output_index);

/** @} */

/** @addtogroup c_api_memory Memory
 * A primitive to describe data.
 * @{ */

/** Initializes a @p memory_desc memory descriptor using @p ndims, @p dims, @p
 * data_type, and data @p format. @p format can be #mkldnn_any, which means
 * that specific data layouts are not permitted. */
mkldnn_status_t MKLDNN_API mkldnn_memory_desc_init(
        mkldnn_memory_desc_t *memory_desc, int ndims, const mkldnn_dims_t dims,
        mkldnn_data_type_t data_type, mkldnn_memory_format_t format);

/** Creates a @p memory_primitive_desc memory primitive descriptor using @p
 * memory_desc and @p engine. @p memory_desc cannot be uncertain, that is,
 * initialized with #mkldnn_any. */
mkldnn_status_t MKLDNN_API mkldnn_memory_primitive_desc_create(
        mkldnn_primitive_desc_t *memory_primitive_desc,
        const mkldnn_memory_desc_t *memory_desc, mkldnn_engine_t engine);

/** Creates a @p view_primitive_desc for a given @p memory_primitive_desc, with
 * @p dims sizes and @p offset offsets. May fail if layout used does not allow
 * obtain desired view. In this case consider using extract primitive */
mkldnn_status_t MKLDNN_API mkldnn_view_primitive_desc_create(
        mkldnn_primitive_desc_t *view_primitive_desc,
        const_mkldnn_primitive_desc_t memory_primitive_desc,
        const mkldnn_dims_t dims, const mkldnn_dims_t offsets);

/** Compares two descriptors of memory primitives.
 * @return 1 if the descriptors are the same.
 * @return 0 if the descriptors are different.
 *
 * Use this function to identify whether a reorder is required for the memory
 * primitives. @p lhs and @p rhs must be either memory or view primitive
 * descriptors. */
int MKLDNN_API mkldnn_memory_primitive_desc_equal(
        const_mkldnn_primitive_desc_t lhs,
        const_mkldnn_primitive_desc_t rhs);

/** Returns the size (in bytes) that is required for given @p
 * memory_primitive_desc */
/* XXX: view? */
size_t MKLDNN_API mkldnn_memory_primitive_desc_get_size(
        const_mkldnn_primitive_desc_t memory_primitive_desc);

/** For a @p memory primitive, returns the data @p handle. For the CPU engine,
 * the data handle is a pointer to the actual data. */
/* XXX: view? */
mkldnn_status_t MKLDNN_API mkldnn_memory_get_data_handle(
        const_mkldnn_primitive_t memory, void **handle);

/** For a @p memory primitive, sets the data @p handle. */
mkldnn_status_t MKLDNN_API mkldnn_memory_set_data_handle(
        mkldnn_primitive_t memory, void *handle);

/** @} */

/** @addtogroup c_api_reorder Reorder
 * A primitive to copy data between memory formats.
 * @{ */

/** Initializes a @p reorder_primitive_desc using descriptors of @p input and
 * @p output memory primitives. */
mkldnn_status_t MKLDNN_API mkldnn_reorder_primitive_desc_create(
        mkldnn_primitive_desc_t *reorder_primitive_desc,
        const_mkldnn_primitive_desc_t input,
        const_mkldnn_primitive_desc_t output);

/** @} */

/** @addtogroup c_api_concat Concat
 * A primitive to concatenate data by arbitrary dimension
 * @{ */

/** Creates out-of-place @p concat_primitive_desc for concatenation of @p n
 * inputs by @p concat_dimension with resulting @p output_desc memory
 * descriptor. @p output_desc can be NULL or be specified with #mkldnn_any
 * format -- in this case appropriate memory format would be chosen
 * automatically. */
mkldnn_status_t MKLDNN_API mkldnn_concat_primitive_desc_create(
        mkldnn_primitive_desc_t *concat_primitive_desc,
        const mkldnn_memory_desc_t *output_desc, int n, int concat_dimension,
        const_mkldnn_primitive_desc_t *input_pds);

#if 0
/** Creates in-place @p concat_primitive_desc for given @p n @p inputs memory
 * primitive descriptors along @p concat_dimension. All inputs must have the
 * same memory format. Output memory format would be the same. Likewise
 * view_primitive_desc_create the call may fail, if memory format of inputs do
 * not allow inplace concatenation for given sizes.
 *
 * @note this primitive is more like a synchronization stub for concatenation,
 * since concat_inplace does no operation during execution.
 *
 * @note since not operation happens user must ensure that input */
mkldnn_status_t MKLDNN_API mkldnn_concat_inplace_by_input_primitive_desc_create(
        mkldnn_primitive_desc_t *concat_primitive_desc,
        int n, int concat_dimension, const_mkldnn_primitive_desc_t *inputs);

/** Creates in-place @p concat_primitive_desc for given @p output memory
 * descriptor and @n inputs with @p sizes sizes along @p concat_dimension. As
 * opposed to out-of-place concatenation @p output must be fully defined here.
 * Likewise view_primitive_desc_create the call may fail, because given memory
 * format does not allow inplace concatenation for given sizes.
 *
 * @note this primitive is more like a synchronization stub for concatenation,
 * since concat_inplace does no operation during execution. */
mkldnn_status_t MKLDNN_API mkldnn_concat_inplace_by_output_primitive_desc_create(
        mkldnn_primitive_desc_t *concat_primitive_desc,
        const mkldnn_primitive_desc_t output, int n, int concat_dimension,
        int *sizes);
#endif

/** @} */

/** @addtogroup c_api_sum Sum
 * A primitive to sum data
 * @{ */

/** Creates out-of-place @p sum_primitive_desc for sum of @p n
 * inputs multiplied by scale with resulting @p output_desc memory
 * descriptor. @p output_desc can be NULL or be specified with #mkldnn_any
 * format -- in this case appropriate memory format would be chosen
 * automatically. */
mkldnn_status_t MKLDNN_API mkldnn_sum_primitive_desc_create(
        mkldnn_primitive_desc_t *sum_primitive_desc,
        const mkldnn_memory_desc_t *output_desc, int n, double* scale,
        const_mkldnn_primitive_desc_t *input_pds);


/** @} */

/** @addtogroup c_api_convolution Convolution
 * A primitive to compute convolution using different algorithms.
 *
 * \f[dst[n][oc][oh][ow]  =
 *     \sum_{kw=0}^{KW}\sum_{kh=0}^{KH}\sum_{ic=0}^{IC}
 *     src[n][ic][oh \cdot s_h - p_l[0] + kh][ow \cdot s_w - p_r[1] + kw]
 *     \cdot weights[g][oc][ic][kh][kw]
 *     + bias[g][oc],\f]
 *
 * where size of output spatial domain is given by
 * \f$ OH = \left\lfloor{\frac{IH - KH + p_l[0] + p_r[0]}{s_h}}
 *          \right\rfloor + 1\f$,
 * \f$ OW = \left\lfloor{\frac{IW - KW + p_l[1] + p_r[1]}{s_w}}
 *          \right\rfloor + 1\f$,
 *
 * and summation is carried over input channels \f$ic\f$ in
 * group \f$g\f$, and \f$s_h, s_w\f$ are @p strides and
 * \f$p_l, p_r\f$ are @p padding_l and @p padding_r.
 * @{ */

/** Initializes a convolution descriptor @p conv_desc for forward propagation
 * using @p prop_kind (possible values are #mkldnn_forward_training or
 * #mkldnn_forward_inference), @p alg_kind, memory descriptors, @p strides, @p
 * padding_l, @p padding_r, and @p padding_kind. In order to create a
 * convolution without bias, @p bias_desc should be either @c NULL or point to
 * a descriptor with memory format equals to #mkldnn_format_undef.
 *
 * @note if @p padding_r is @c NULL, the padding is supposed to be symmetric
 *
 * @note memory descriptors are allowed to be initialized with #mkldnn_any
 * value of @p format_kind. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_forward_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a convolution descriptor @p conv_desc for backward propagation
 * with respect to data using @p alg_kind, memory descriptors, @p strides, @p
 * padding_l, @p padding_r, and @p padding_kind.
 *
 * @note memory descriptors are allowed to be initialized with #mkldnn_any
 * value of @p format_kind. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_backward_data_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** Initializes a convolution descriptor @p conv_desc for backward propagation
 * with respect to weights using @p alg_kind, memory descriptors, @p strides,
 * @p padding_l, @p padding_r, and @p padding_kind.
 *
 * @note memory descriptors are allowed to be initialized with #mkldnn_any
 * value of @p format_kind. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_backward_weights_desc_init(
        mkldnn_convolution_desc_t *conv_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t padding_l, const mkldnn_dims_t padding_r,
        mkldnn_padding_kind_t padding_kind);

/** @} */

/** @addtogroup c_api_relu ReLU
 * A primitive to compute a parametric rectifier linear unit (ReLU).
 * 
 * \f[dst[n][c][h][w] = \max(src[n][c][h][w], 0) +
 *                      \min(src[n][c][h][w], 0) \cdot negative\_slope\f]
 * @{ */

/** Initializes a @p relu_desc for forward propagation using @p prop_kind
 * (possible values are #mkldnn_forward_training or #mkldnn_forward_inference),
 * @p negative_slope and memory descriptor @p data_desc.
mkldnn_status_t MKLDNN_API mkldnn_relu_forward_desc_init(
        mkldnn_relu_desc_t *relu_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *data_desc, double negative_slope); */

/** Initializes a @p relu_desc for backward propagation using @p negative_slope
 * and memory descriptors @p diff_data_desc and @p data_desc.
mkldnn_status_t MKLDNN_API mkldnn_relu_backward_desc_init(
        mkldnn_relu_desc_t *relu_desc,
        const mkldnn_memory_desc_t *diff_data_desc,
        const mkldnn_memory_desc_t *data_desc, double negative_slope); */

/** @} */

/** @addtogroup c_api_softmax Softmax
 * A primitive to perform softmax.
 *
 * \f[dst[u][c][in] =
 *    \frac{\exp(src[ou][c][in]) - \max\limits_{c}(src[ou][c][in])}
 *    {\sum\limits_{c}\{\exp(src[ou][c][in]) 
 *    - \max\limits_{c}(src[ou][c][in])\}},\f]
 *
 * where \f$ou, iu\f$ are outer and inner sizes repectively, defined
 * by @p data_desc.dims and @p softmax_axis.
 * @{ */

/** Initializes a @p softmax_desc for forward propagation using @p prop_kind
 * (possible value are #mkldnn_forward_training or #mkldnn_forward_inference)
 * and memory descriptor @p data_desc. */
mkldnn_status_t MKLDNN_API mkldnn_softmax_forward_desc_init(
        mkldnn_softmax_desc_t *softmax_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *data_desc, int softmax_axis);

/** @} */

/** @addtogroup c_api_pooling Pooling
 * A primitive to perform max, min, or average pooling.
 * 
 * Max pooling:
 * \f[dst[n][oc][oh][ow] =
 *     \max\limits_{kw,kh}
 *     (src[n][ic][oh \cdot s_h - p_l[0] + kh][ow \cdot s_w - p_r[1] + kw]),\f]
 *
 * Average pooling:
 * \f[dst[n][oc][oh][ow] =
 *     \frac{1}{KW \cdot KH}\sum\limits_{kw,kh}
 *     src[n][ic][oh \cdot s_h - p_l[0] + kh][ow \cdot s_w - p_r[1] + kw],\f]
 *
 * where \f$p_l, p_r\f$ are @p padding_l and @p padding_r
 * respectively and output spatial dimensions are calculated
 * similarly as done in convolution.
 * @{ */

/** Initializes a pooling descriptor @p pool_desc for forward propagation using
 * @p prop_kind (possible values are #mkldnn_forward_training or
 * #mkldnn_forward_inference), @p alg_kind, memory descriptors, and pooling
 * parameters in spatial domain: @p strides, @p kernel sizes, @p padding_l, @p
 * padding_r, and @p padding_kind.
 *
 * @note if @p padding_r is @c NULL, the padding is supposed to be symmetric
 *
 * @todo clarify! */
mkldnn_status_t MKLDNN_API mkldnn_pooling_forward_desc_init(
        mkldnn_pooling_desc_t *pool_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t kernel, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** Initializes a pooling descriptor @p pool_desc for backward propagation
 * using @p alg_kind, memory descriptors, and pooling parameters in spatial
 * domain: @p strides, @p kernel sizes, @p padding_l, @p padding_r, and @p
 * padding_kind.
 *
 * @todo clarify! */
mkldnn_status_t MKLDNN_API mkldnn_pooling_backward_desc_init(
        mkldnn_pooling_desc_t *pool_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *diff_dst_desc, const mkldnn_dims_t strides,
        const mkldnn_dims_t kernel, const mkldnn_dims_t padding_l,
        const mkldnn_dims_t padding_r, mkldnn_padding_kind_t padding_kind);

/** @} */

/** @addtogroup c_api_lrn LRN
 * A primitive to perform local response normalization (LRN) across or within
 * channels. 
 * 
 * LRN accross channels:
 * \f[dst[n][c][h][w] = \left\{k + \frac{\alpha}{n_{l}}
 *                      \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2}
 *                      (src[n][c+i][h][w])^2\right\}^{-\beta}
 *                      src[n][c][h][w],\f]
 *
 * LRN within channels:
 * \f[dst[n][c][h][w] = \left\{k + \frac{\alpha}{n_{l}}
 *                      \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2}
 *                      (src[n][c][h+i][w+i])^2\right\}^{-\beta}
 *                      src[n][c][h][w],\f]
 *
 * where \f$n_{l}\f$ is the @p local_size.
 * @{ */

/** Initializes an @p lrn_desc for forward propagation using @p prop_kind
 * (possible values are #mkldnn_forward_training or #mkldnn_forward_inference),
 * @p alg_kind, memory descriptor @p data_desc, and regularization
 * parameters @p local_size, @p alpha, @p beta, and @p k. */
mkldnn_status_t MKLDNN_API mkldnn_lrn_forward_desc_init(
        mkldnn_lrn_desc_t *lrn_desc, mkldnn_prop_kind_t prop_kind,
        mkldnn_alg_kind_t alg_kind, const mkldnn_memory_desc_t *data_desc,
        int local_size, double alpha, double beta, double k);

/** Initializes an @p lrn_desc for backward propagation using @p alg_kind,
 * memory descriptors @p data_desc, and @p diff_data_desc, and regularization
 * parameters @p local_size, @p alpha, @p beta, and @p k. */
mkldnn_status_t MKLDNN_API mkldnn_lrn_backward_desc_init(
        mkldnn_lrn_desc_t *lrn_desc, mkldnn_alg_kind_t alg_kind,
        const mkldnn_memory_desc_t *diff_data_desc,
        const mkldnn_memory_desc_t *data_desc, int local_size, double alpha,
        double beta, double k);

/** @} */

/** @addtogroup c_api_batch_normalization Batch Normalization
 * A primitive to perform batch normalization
 * \f[dst[n][c][h][w] = \gamma[c] \frac{src[n][c][h][w] - \mu[c]}
 *                      {\sqrt{\sigma[c] + eps}} + \beta[c],\f]
 *
 * where \f$\gamma[c], \beta[c]\f$ are weights and bias for a channel and,
 *
 * \f$\mu[c] = \frac{1}{NHW} \sum\limits_{whn} src[n][c][h][w]\f$,
 * \f$\sigma[c] = \frac{1}{NHW} \sum\limits_{whn} 
 *                              (src[n][c][h][w] - \mu[c])^2\f$,
 *
 * and eps is a constant to improve numerical stability.
 * @{ */

/** Initializes a batch normalization descriptor @p bnrm_desc for forward
 * propagation using @p prop_kind, (possible values are
 * #mkldnn_forward_training or #mkldnn_forward_inference), memory descriptor
 * @p data_desc, normalization parameter @p epsilon and flags (possible values
 * are #mkldnn_use_global_stats and #mkldnn_use_scaleshift).
 *
 * @sa mkldnn_batch_normalization_desc_t */
mkldnn_status_t MKLDNN_API mkldnn_batch_normalization_forward_desc_init(
        mkldnn_batch_normalization_desc_t *bnrm_desc,
        mkldnn_prop_kind_t prop_kind, const mkldnn_memory_desc_t *data_desc,
        double epsilon, unsigned flags);

/** Initializes a batch normalization descriptor @p bnrm_desc for backward
 * propagation with respect to data and scale-shift parameters using memory
 * descriptors @p data_desc and @p diff_data_desc, and normalization parameter
 * @p epsilon and flags (possible values are #mkldnn_use_global_stats and
 * #mkldnn_use_scaleshift).
 *
 * @sa mkldnn_batch_normalization_desc_t */
mkldnn_status_t MKLDNN_API mkldnn_batch_normalization_backward_desc_init(
        mkldnn_batch_normalization_desc_t *bnrm_desc,
        mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *diff_data_desc,
        const mkldnn_memory_desc_t *data_desc,
        double epsilon, unsigned flags);

/** @} */

/** @addtogroup c_api_inner_product Inner product
 * A primitive to compute an inner product.
 * Inner product layer is also known as fully connected layer.
 * with spatial dimension:
 * 
 * \f[dst[n][oc] = \sum\limits_{ic, kh, kw}
 *                 src[n][ic][kh][kw] \cdot weights[oc][ic][kh][kw] 
 *                 + bias[oc]\f]
 * @{ */

/** Initializes an inner product descriptor @p ip_desc for forward propagation
 * using @p prop_kind (possible values are #mkldnn_forward_training or
 * #mkldnn_forward_inference) and memory descriptors. In order to create an
 * inner product without bias, @p bias_desc should be either @c NULL or a
 * pointer to descriptor with memory format equals to #mkldnn_format_undef.
 *
 * @note
 *     memory descriptors are allowed to be initialized with #mkldnn_any value
 *     of @p format_kind. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_forward_desc_init(
        mkldnn_inner_product_desc_t *ip_desc, mkldnn_prop_kind_t prop_kind,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *bias_desc,
        const mkldnn_memory_desc_t *dst_desc);

/** Initializes an inner product descriptor @p ip_desc for backward propagation
 * with respect to data using memory descriptors.
 *
 * @note
 *     memory descriptors are allowed to be initialized with #mkldnn_any value
 *     of @p format_kind. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_backward_data_desc_init(
        mkldnn_inner_product_desc_t *ip_desc,
        const mkldnn_memory_desc_t *diff_src_desc,
        const mkldnn_memory_desc_t *weights_desc,
        const mkldnn_memory_desc_t *diff_dst_desc);

/** Initializes an inner product descriptor @p ip_desc for backward propagation
 * with respect to weights using memory descriptors.
 *
 * @note
 *     memory descriptors are allowed to be initialized with #mkldnn_any value
 *     of @p format_kind. */
mkldnn_status_t MKLDNN_API mkldnn_inner_product_backward_weights_desc_init(
        mkldnn_inner_product_desc_t *ip_desc,
        const mkldnn_memory_desc_t *src_desc,
        const mkldnn_memory_desc_t *diff_weights_desc,
        const mkldnn_memory_desc_t *diff_bias_desc,
        const mkldnn_memory_desc_t *diff_dst_desc);

/** @} */

/** @addtogroup c_api_convolution_relu Convolution followed by ReLU
 * A merged primitive to compute a convolution followed by relu.
 * @{ */

/** Initializes a merged convolution-relu descriptor @p conv_relu_desc for
 * forward propagation (supported inference mode only) using convolution
 * descriptor @p conv_desc and ReLU parameter @p negative slope. */
mkldnn_status_t MKLDNN_API mkldnn_convolution_relu_desc_init(
        mkldnn_convolution_relu_desc_t *conv_relu_desc,
        const mkldnn_convolution_desc_t *conv_desc, double negative_slope);

/** @} */

/** @} */

/** @addtogroup c_api_engine Engine operations
 * @{ */

/** Returns the number of engines of a particular @p kind. */
size_t MKLDNN_API mkldnn_engine_get_count(mkldnn_engine_kind_t kind);

/** Creates an @p engine of particular @p kind and @p index. */
mkldnn_status_t MKLDNN_API mkldnn_engine_create(mkldnn_engine_t *engine,
        mkldnn_engine_kind_t kind, size_t index);

/** Returns the kind of an @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_engine_get_kind(mkldnn_engine_t engine,
        mkldnn_engine_kind_t *kind);

/** Destroys an @p engine. */
mkldnn_status_t MKLDNN_API mkldnn_engine_destroy(mkldnn_engine_t engine);

/** @} */

/** @addtogroup c_api_stream Execution stream operations
 * @{ */

/** Creates an execution @p stream of @p stream_kind. */
mkldnn_status_t MKLDNN_API mkldnn_stream_create(mkldnn_stream_t *stream,
        mkldnn_stream_kind_t stream_kind);

/** Submits @p primitives to an execution @p stream. The number of primitives
 * is @p n.  All or none of the primitives can be lazy. In case of an error,
 * returns the offending @p error_primitive if it is not @c NULL. */
mkldnn_status_t MKLDNN_API mkldnn_stream_submit(mkldnn_stream_t stream,
        size_t n, mkldnn_primitive_t primitives[],
        mkldnn_primitive_t *error_primitive);

/** Waits for all primitives in the execution @p stream to finish. Returns
 * immediately if @p block is zero. In case of an error, returns
 * the offending @p error_primitive if it is not @c NULL. */
mkldnn_status_t MKLDNN_API mkldnn_stream_wait(mkldnn_stream_t stream,
        int block, mkldnn_primitive_t *error_primitive);

/** Reruns all the primitives within the @p stream. In case of an error,
 * returns the offending @p error_primitive if it is not @c NULL. */
mkldnn_status_t MKLDNN_API mkldnn_stream_rerun(mkldnn_stream_t stream,
        mkldnn_primitive_t *error_primitive);

/** Destroys an execution @p stream. */
mkldnn_status_t MKLDNN_API mkldnn_stream_destroy(mkldnn_stream_t stream);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif
