from setuptools import Command, distutils, Extension, setup
from platform import system

import external
import sys

import setuptools.command.install
import setuptools.command.build_ext
import distutils.command.build
import distutils.command.clean


###############################################################################
# Custom build commands
###############################################################################

class build_deps(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        external.prepare()


class build(distutils.command.build.build):
    sub_commands = [
        ('build_deps', lambda self: True),
    ] + distutils.command.build.build.sub_commands


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        setuptools.command.build_ext.build_ext.run(self)


class install(setuptools.command.install.install):
    def run(self):
        if not self.skip_build:
            self.run_command('build_deps')
        setuptools.command.install.install.run(self)


class clean(distutils.command.clean.clean):
    def run(self):
        external.clean()
        distutils.command.clean.clean.run(self)


cmdclass = {
    'build': build,
    'build_ext': build_ext,
    'build_deps': build_deps,
    'install': install,
    'clean': clean,
}


###############################################################################
# Configure compile flags
###############################################################################

swig_opts = ['-c++', '-builtin', '-modern', '-modernargs',
             '-Iideep4py/py/mm',
             '-Iideep4py/py/primitives',
             '-Iideep4py/py/swig_utils',
             # '-Iideep4py/py/dlcp',
             '-Iideep4py/include/primitives/',
             '-Iideep4py/include/mm/']

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11', '-Wno-unknown-pragmas']
link_opts = ['-Wl,-z,now', '-Wl,-z,noexecstack',
             '-Wl,-rpath,' + '$ORIGIN/lib', '-L' + './external/lib']

includes = ['ideep4py/include',
            'ideep4py/include/mkl',
            'ideep4py/common',
            'ideep4py/include/mm',
            'ideep4py/py/mm',
            'ideep4py/py/primitives',
            # 'ideep4py/py/dlcp',
            'ideep4py/include/primitives',
            'ideep4py/include/blas',
            'ideep4py/include/primitives/ops',
            'ideep4py/include/primitives/prim_mgr',
            'external/include']

libraries = ['mkldnn', 'mklml_intel']  # , 'dlcomp']

if system() == 'Linux':
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['glog', 'm']
    src = ['ideep4py/py/ideep4py.i',
           # 'ideep4py/py/dlcp/dlcp_py.cc',
           'ideep4py/mm/mem.cc',
           'ideep4py/mm/tensor.cc',
           'ideep4py/py/mm/mdarray.cc',
           'ideep4py/common/cpu_info.cc',
           'ideep4py/common/utils.cc',
           'ideep4py/common/common.cc',
           'ideep4py/blas/sum.cc',
           'ideep4py/py/mm/basic.cc',
           'ideep4py/primitives/ops/eltwise_fwd.cc',
           'ideep4py/primitives/ops/eltwise_bwd.cc',
           'ideep4py/primitives/eltwise.cc',
           'ideep4py/primitives/ops/conv_fwd.cc',
           'ideep4py/primitives/ops/conv_bwd_weights.cc',
           'ideep4py/primitives/ops/conv_bwd_data.cc',
           'ideep4py/primitives/ops/reorder_op.cc',
           'ideep4py/primitives/conv.cc',
           'ideep4py/primitives/ops/pooling_fwd.cc',
           'ideep4py/primitives/ops/pooling_bwd.cc',
           'ideep4py/primitives/pooling.cc',
           'ideep4py/primitives/ops/linear_fwd.cc',
           'ideep4py/primitives/ops/linear_bwd_weights.cc',
           'ideep4py/primitives/ops/linear_bwd_data.cc',
           'ideep4py/primitives/linear.cc',
           'ideep4py/primitives/bn.cc',
           'ideep4py/primitives/ops/bn_fwd.cc',
           'ideep4py/primitives/ops/bn_bwd.cc',
           'ideep4py/primitives/ops/concat_fwd.cc',
           'ideep4py/primitives/ops/concat_bwd.cc',
           'ideep4py/primitives/concat.cc',
           'ideep4py/primitives/ops/lrn_fwd.cc',
           'ideep4py/primitives/ops/lrn_bwd.cc',
           'ideep4py/primitives/lrn.cc',
           'ideep4py/primitives/dropout.cc',
           ]
else:
    # TODO
    src = ['mkldnn/mdarray.i', 'mkldnn/mdarray.cc']


###############################################################################
# Declare extensions and package
###############################################################################

install_requires = [
    'numpy>=1.9,<=1.13',
]

tests_require = [
    'mock',
    'pytest',
]

ext_modules = []

ext = Extension(
    'ideep4py._ideep4py', sources=src,
    swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, extra_link_args=link_opts,
    include_dirs=includes, libraries=libraries)

ext_modules.append(ext)

packages = ['ideep4py', 'ideep4py.cosim']

setup(
    name='ideep4py',
    version='1.0.0',
    description='ideep4py is a wrapper for iDeep library.',
    author='Intel',
    author_email='',
    url='https://github.com/intel/ideep',
    license='MIT License',
    packages=packages,
    package_dir={'ideep4py': 'ideep4py/'},
    package_data={'ideep4py': ['lib/*', ]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
)
