from setuptools import Command, distutils, Extension, setup
from platform import system

import sys
import os
import shutil

import setuptools.command.install
import setuptools.command.build_ext
import distutils.command.build
import distutils.command.clean


###############################################################################
# mkl-dnn preparation
###############################################################################
os_name = system()

MODULE_DESC = 'Intel mkl-dnn'
cwd = os.path.split(os.path.realpath(__file__))[0]

ideep4py_dir = cwd + '/ideep4py'
ideep_build_dir = cwd + '/../build'

def install_mkldnn():
    print('Installing ...')
    os.system(
      'cmake -DCMAKE_INSTALL_PREFIX=%s --build %s \
              && cmake --build %s --target install' \
              % (ideep4py_dir, ideep_build_dir, ideep_build_dir))

###############################################################################
# External preparation
###############################################################################

libdir = ideep4py_dir + '/lib'
includedir = ideep4py_dir + '/include'
sharedir = ideep4py_dir + '/share'

def prepare_ext():
    install_mkldnn()
    # dlcp.prepare()


def clean_ext():
    if os.path.exists(libdir):
        shutil.rmtree(libdir)
#    if os.path.exists(includedir):
#        shutil.rmtree(includedir)
    if os.path.exists(sharedir):
        shutil.rmtree(sharedir)

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
             '-Iideep4py/include/mm',
             '-Iinclude']

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11', '-Wno-unknown-pragmas', '-march=native', '-mtune=native']

if os_name == 'Darwin':
    link_opts = ['-Wl,-rpath,@loader_path/lib', '-Lideep4py/lib']
else:
    link_opts = ['-Wl,-rpath,$ORIGIN/lib', '-Lideep4py/lib']

includes = ['ideep4py/include',
            'ideep4py/include/mklml',
            'ideep4py/common',
            'ideep4py/include/mm',
            'ideep4py/py/mm',
            'ideep4py/py/primitives',
            # 'ideep4py/py/dlcp',
            'ideep4py/include/blas',
            'include', 'include/mklml', 'include/ideep']

if os_name == 'Linux':
    libraries = ['mkldnn', 'mklml_intel']  # , 'dlcomp']
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['m']
    link_opts += ['-Wl,-z,now', '-Wl,-z,noexecstack']
else:
    libraries = ['mkldnn', 'mklml']

src = ['ideep4py/py/ideep4py.i',
       # 'ideep4py/py/dlcp/dlcp_py.cc',
       # 'ideep4py/mm/mem.cc',
       # 'ideep4py/mm/tensor.cc',
       'ideep4py/py/mm/mdarray.cc',
       # 'ideep4py/blas/sum.cc',
       # 'ideep4py/py/mm/basic.cc',
       ]

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
    package_data={'ideep4py' : ['lib/*']},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
)
