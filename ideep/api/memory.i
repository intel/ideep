%module (package="mkldnn.api") memory
%{
  #define SWIG_FILE_WITH_INIT
  #include <cstddef>
  #include <mkldnn.hpp>
  #include "utils.hpp"
  #include "mkldnn_ex.hpp"
  using mkldnn::handle_traits;
%}

%include stl.i
%include exception.i

%feature("flatnested");
%feature("nodefaultctor");

%import support.i
%import seq_typemap.i

%int_sequence_typemap(mkldnn::memory::dims);

namespace mkldnn {

namespace c_api {
  %import c_api.i
}

struct memory: public primitive {
private:
    std::shared_ptr<char> _handle;

public:
    typedef std::vector<int> dims; /*manual scrip*/

    template <typename T> static void validate_dims(std::vector<T> v);

    enum data_type {
        data_undef = mkldnn_data_type_undef,
        f32 = mkldnn_f32,
        s32 = mkldnn_s32,
    };

    enum format {
        format_undef = mkldnn_format_undef,
        any = mkldnn_any,
        blocked = mkldnn_blocked,
        x = mkldnn_x,
        nc = mkldnn_nc,
        nchw = mkldnn_nchw,
        nhwc = mkldnn_nhwc,
        chwn = mkldnn_chwn,
        nChw8c = mkldnn_nChw8c,
        nChw16c = mkldnn_nChw16c,
        oi = mkldnn_oi,
        io = mkldnn_io,
        oihw = mkldnn_oihw,
        ihwo = mkldnn_ihwo,
        hwio = mkldnn_hwio,
        oIhw8i = mkldnn_oIhw8i,
        oIhw16i = mkldnn_oIhw16i,
        OIhw8i8o = mkldnn_OIhw8i8o,
        OIhw16i16o = mkldnn_OIhw16i16o,
        OIhw8o8i = mkldnn_OIhw8o8i,
        OIhw16o16i = mkldnn_OIhw16o16i,
        OIhw8i16o2i = mkldnn_OIhw8i16o2i,
        OIhw8o16i2o = mkldnn_OIhw8o16i2o,
        Oihw8o = mkldnn_Oihw8o,
        Oihw16o = mkldnn_Oihw16o,
        Ohwi8o = mkldnn_Ohwi8o,
        Ohwi16o = mkldnn_Ohwi16o,
        OhIw16o4i = mkldnn_OhIw16o4i,
        goihw = mkldnn_goihw,
        gOIhw8i8o = mkldnn_gOIhw8i8o,
        gOIhw16i16o = mkldnn_gOIhw16i16o,
        gOIhw8i16o2i = mkldnn_gOIhw8i16o2i,
        gOIhw8o16i2o = mkldnn_gOIhw8o16i2o,
        gOihw8o = mkldnn_gOihw8o,
        gOihw16o = mkldnn_gOihw16o,
        gOhwi8o = mkldnn_gOhwi8o,
        gOhwi16o = mkldnn_gOhwi16o,
        gOIhw8o8i = mkldnn_gOIhw8o8i,
        gOIhw16o16i = mkldnn_gOIhw16o16i,
        gOhIw16o4i = mkldnn_gOhIw16o4i,
    };

    struct desc {
        mkldnn_memory_desc_t data;
        desc(dims adims, data_type adata_type,
                format aformat);
        desc(const mkldnn_memory_desc_t &adata);
    };

    struct primitive_desc {
        primitive_desc() {}
        primitive_desc(const desc &adesc, const engine &aengine);
        memory::desc desc();
        size_t get_size() const;
        bool operator==(const primitive_desc &other) const;
        bool operator!=(const primitive_desc &other) const;
    };

    memory(const primitive &aprimitive);
    // XXX: This is not what we want
    memory(const primitive_desc &adesc);
    memory(const primitive_desc &adesc, void *ahandle);

    primitive_desc get_primitive_desc() const;
    inline void *get_data_handle() const;
    inline void set_data_handle(void *handle) const;

};

}

mkldnn::memory::format get_fmt(mkldnn::memory::primitive_desc mpd);
mkldnn::memory::format get_desired_format(int channel);


%template (dims) std::vector<int>;
%template (vectord) std::vector<double>;
%template (mpd_list) std::vector<mkldnn::memory::primitive_desc>;
