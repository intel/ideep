#!/usr/bin/env python

from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools import setup
from setuptools.extension import Extension
from numpy import get_include
from platform import system
import sys
import external

setup_requires = []
install_requires = ['numpy>=1.9.0']

mkldnn_root = external.mkldnn.root()
mkldnn_version = 'b01e3a55a07be62172e713bcd2644c5176360212'


def prepare_mkldnn():
    external.mkldnn.prepare(mkldnn_version)


class _build_py(build_py):
    def run(self):
        prepare_mkldnn()
        self.run_command('build_ext')
        build_py.run(self)


class _install(install):
    def run(self):
        prepare_mkldnn()
        self.run_command('build_ext')
        install.run(self)


modules = {
    'ideep.api._c_api':
    ['ideep/api/c_api.i'],

    'ideep.api._support':
    ['ideep/api/support.i'],

    'ideep.api._memory':
    ['ideep/api/memory.i', 'ideep/utils.cc'],

    'ideep.api._inner_product_forward':
    ['ideep/api/inner_product_forward.i'],

    'ideep.api._inner_product_backward_data':
    ['ideep/api/inner_product_backward_data.i'],

    'ideep.api._inner_product_backward_weights':
    ['ideep/api/inner_product_backward_weights.i'],

    'ideep.api._convolution_forward':
    ['ideep/api/convolution_forward.i'],

    'ideep.api._convolution_backward_data':
    ['ideep/api/convolution_backward_data.i'],

    'ideep.api._convolution_backward_weights':
    ['ideep/api/convolution_backward_weights.i'],

    'ideep.api._eltwise_forward':
    ['ideep/api/eltwise_forward.i'],

    'ideep.api._eltwise_backward':
    ['ideep/api/eltwise_backward.i'],

    'ideep.api._pooling_forward':
    ['ideep/api/pooling_forward.i'],

    'ideep.api._pooling_backward':
    ['ideep/api/pooling_backward.i'],

    'ideep.api._lrn_forward':
    ['ideep/api/lrn_forward.i'],

    'ideep.api._lrn_backward':
    ['ideep/api/lrn_backward.i'],

    'ideep.api._sum':
    ['ideep/api/sum.i'],

    'ideep.api._reorder':
    ['ideep/api/reorder.i'],

    'ideep.api._concat':
    ['ideep/api/concat.i'],

    'ideep.api._view':
    ['ideep/api/view.i'],

    'ideep.api._bn_forward':
    ['ideep/api/bn_forward.i'],

    'ideep.api._bn_backward':
    ['ideep/api/bn_backward.i'],

    'ideep.api._dropout':
    ['ideep/api/dropout.i'],

    'ideep.api._cosim_dump':
    ['ideep/api/cosim_dump.i', 'ideep/api/cosim_dump.cc'],
}

swig_opts = [
    '-c++',
    '-builtin',
    '-modern',
    '-modernargs',
    '-Iideep/api',
    '-Iideep',
    '-Iideep/swig_utils'
]

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11']

link_opts = [
    '-Wl,-z,now',
    '-Wl,-z,noexecstack',
    '-Wl,-rpath,' + mkldnn_root + '/lib',
    '-L' + mkldnn_root + '/lib'
]

includes = [
    get_include(),
    'ideep',
    'ideep/swig_utils',
    mkldnn_root + '/include'
]

libraries = ['mkldnn', 'mklml_intel']

if system() == 'Linux':
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['boost_system', 'glog', 'm']
    mdarray_src = ['ideep/mdarray.i', 'ideep/mdarray.cc', 'ideep/cpu_info.cc']
else:
    mdarray_src = ['ideep/mdarray.i', 'ideep/mdarray.cc']

ext_modules = []
for m, s in modules.items():
    ext = Extension(
        m,
        sources=s,
        swig_opts=swig_opts,
        extra_compile_args=ccxx_opts,
        extra_link_args=link_opts,
        include_dirs=includes,
        libraries=libraries
    )
    ext_modules.append(ext)

ext = Extension(
    'ideep/_mdarray',
    sources=mdarray_src,
    swig_opts=swig_opts,
    extra_compile_args=ccxx_opts,
    extra_link_args=link_opts,
    include_dirs=includes,
    libraries=libraries
)
ext_modules.append(ext)

packages = ['ideep', 'ideep.api', 'ideep.chainer']

setup(
    name='ideep',
    version='0.0.0',
    description='Intel deeplearning optimization python pkg',
    author='intel',
    author_email='',
    url='',
    license='MIT License',
    packages=packages,
    ext_modules=ext_modules,
    cmdclass={'install': _install, 'build_py': _build_py},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
