from setuptools import Command, distutils, Extension, setup
from platform import system

import sys
import os

import setuptools.command.install
import setuptools.command.build_ext
import distutils.command.build
import distutils.command.clean


###############################################################################
# mkl-dnn preparation
###############################################################################

MODULE_DESC = 'Intel mkl-dnn'
PYTHON_ROOT = os.path.split(os.path.realpath(__file__))[0]

MKLDNN_WORK_PATH = PYTHON_ROOT + '/../mkl-dnn'
MKLDNN_ROOT = PYTHON_ROOT
MKLDNN_LIB_PATH = MKLDNN_ROOT + '/lib'
MKLDNN_INCLUDE_PATH = MKLDNN_ROOT + '/include'
MKLDNN_SOURCE_PATH = MKLDNN_WORK_PATH
MKLDNN_BUILD_PATH = MKLDNN_WORK_PATH + '/build'
MKLML_PKG_PATH = MKLDNN_WORK_PATH + '/external'

lib_targets = ['libmkldnn.so',
               'libmkldnn.so.0',
               'libmklml_gnu.so',
               'libmklml_intel.so',
               'libiomp5.so']


def get_mklml_path():
    mklml_pkg_path_leafs = os.listdir(MKLML_PKG_PATH)
    mklml_origin_path = None
    for leaf in mklml_pkg_path_leafs:
        if os.path.isdir('%s/%s' % (MKLML_PKG_PATH, leaf)) and \
           'mklml' in leaf:
            mklml_origin_path = '%s/%s' % (MKLML_PKG_PATH, leaf)
            break
    return mklml_origin_path


def install_mkldnn():
    print('Installing ...')

    os.chdir(MKLDNN_SOURCE_PATH)
    os.system(
      'cd build && cmake -DCMAKE_INSTALL_PREFIX=%s .. && \
       make -j && make install' % MKLDNN_ROOT)

    # install mklml
    mklml_origin_path = get_mklml_path()
    if mklml_origin_path:
        os.system('cp %s/lib/* %s' % (mklml_origin_path, MKLDNN_LIB_PATH))
        os.system('cp %s/include/* %s' %
                  (mklml_origin_path, MKLDNN_INCLUDE_PATH))
    else:
        sys.exit('%s build error... No Intel mklml pkg.' % MODULE_DESC)


def prepare_mkldnn():
    print('Intel mkl-dnn preparing ...')
    mkldnn_installed = True

    if not os.path.exists(MKLDNN_SOURCE_PATH + '/src'):
        sys.exit('%s prepare error... Please init mkl-dnn submodule first.' % MODULE_DESC)
    else:
        mklml_origin_path = get_mklml_path()
        if not mklml_origin_path:
            sys.exit('%s prepare error... No Intel mklml pkg.' % MODULE_DESC)
        include_targets = []
        include_targets += os.listdir(mklml_origin_path + '/include')
        include_targets += os.listdir(MKLDNN_SOURCE_PATH + '/include')

        if not os.path.exists(MKLDNN_BUILD_PATH):
            sys.exit('%s prepare error... Please build for mkl-dnn first.' % MODULE_DESC)
        elif not all(os.path.exists(MKLDNN_LIB_PATH + '/' + lib)
                     for lib in lib_targets) or \
            not all(os.path.exists(MKLDNN_INCLUDE_PATH + '/' + include)
                    for include in include_targets):
            mkldnn_installed = False

    if not mkldnn_installed:
        install_mkldnn()

    os.chdir(PYTHON_ROOT)
    print('Intel mkl-dnn prepared !')


###############################################################################
# External preparation
###############################################################################

EXT_LIB_PATH = PYTHON_ROOT + '/lib'
EXT_INCLUDE_PATH = PYTHON_ROOT + '/include'
EXT_SHARE_PATH = PYTHON_ROOT + '/share'
TARGET_LIB_PATH = PYTHON_ROOT + '/ideep4py/lib'

target_libs = [
    # 'libdlcomp.so',
    'libiomp5.so',
    'libmkldnn.so*',
]


def prepare_ext():
    if not os.path.exists(EXT_LIB_PATH):
        os.system('mkdir %s' % EXT_LIB_PATH)
    if not os.path.exists(EXT_INCLUDE_PATH):
        os.system('mkdir %s' % EXT_INCLUDE_PATH)

    prepare_mkldnn()
    # dlcp.prepare()

    if os.path.exists(TARGET_LIB_PATH):
        os.system('rm -rf %s' % TARGET_LIB_PATH)
    os.system('mkdir %s' % TARGET_LIB_PATH)
    libmklml = os.popen(
        'ldd lib/libmkldnn.so |\
        grep libmklml | awk \'{print $1}\'').read()
    global target_libs
    target_libs += [libmklml[:-1]]
    for lib in target_libs:
        os.system('cp %s/%s %s' % (EXT_LIB_PATH, lib, TARGET_LIB_PATH))


def clean_ext():
    if os.path.exists(TARGET_LIB_PATH):
        os.system('rm -rf %s' % TARGET_LIB_PATH)
    if os.path.exists(EXT_LIB_PATH):
        os.system('rm -rf %s' % EXT_LIB_PATH)
    if os.path.exists(EXT_INCLUDE_PATH):
        os.system('rm -rf %s' % EXT_INCLUDE_PATH)
    if os.path.exists(EXT_SHARE_PATH):
        os.system('rm -rf %s' % EXT_SHARE_PATH)


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
        prepare_ext()


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
        clean_ext()
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
             '-Iideep4py/include/primitives',
             '-Iideep4py/include/mm',
             '-I../include',
             '-I../include/ideep']

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11', '-Wno-unknown-pragmas', '-mavx']
link_opts = ['-Wl,-z,now', '-Wl,-z,noexecstack',
             '-Wl,-rpath,' + '$ORIGIN/lib', '-L' + './lib']

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
            'include',
            '../include',
            '../include/ideep']

libraries = ['mkldnn', 'mklml_intel']  # , 'dlcomp']

if system() == 'Linux':
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['m']
    src = ['ideep4py/py/ideep4py.i',
           # 'ideep4py/py/dlcp/dlcp_py.cc',
           # 'ideep4py/mm/mem.cc',
           # 'ideep4py/mm/tensor.cc',
           'ideep4py/py/mm/mdarray.cc',
           'ideep4py/common/cpu_info.cc',
           'ideep4py/common/utils.cc',
           # 'ideep4py/common/common.cc',
           # 'ideep4py/blas/sum.cc',
           # 'ideep4py/py/mm/basic.cc',
           # 'ideep4py/primitives/ops/reorder_op.cc',
           # 'ideep4py/primitives/bn.cc',
           # 'ideep4py/primitives/ops/bn_fwd.cc',
           # 'ideep4py/primitives/ops/bn_bwd.cc',
           # 'ideep4py/primitives/ops/concat_fwd.cc',
           # 'ideep4py/primitives/ops/concat_bwd.cc',
           # 'ideep4py/primitives/concat.cc',
           # 'ideep4py/primitives/ops/lrn_fwd.cc',
           # 'ideep4py/primitives/ops/lrn_bwd.cc',
           # 'ideep4py/primitives/lrn.cc',
           # 'ideep4py/primitives/dropout.cc',
           ]
else:
    # TODO
    src = ['mkldnn/mdarray.i', 'mkldnn/mdarray.cc']


###############################################################################
# Declare extensions and package
###############################################################################

install_requires = [
    'numpy==1.13',
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
    version='1.0.3',
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
