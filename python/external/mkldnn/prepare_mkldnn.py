import os
import sys
import multiprocessing

MODULE_DESC = 'Intel mkl-dnn'

MKLDNN_WORK_PATH = os.path.split(os.path.realpath(__file__))[0]
MKLDNN_ROOT = MKLDNN_WORK_PATH + '/..'
MKLDNN_LIB_PATH = MKLDNN_ROOT + '/lib'
MKLDNN_INCLUDE_PATH = MKLDNN_ROOT + '/include'
MKLDNN_SOURCE_PATH = MKLDNN_WORK_PATH + '/source'
MKLDNN_BUILD_PATH = MKLDNN_WORK_PATH + '/source/build'
MKLML_PKG_PATH = MKLDNN_SOURCE_PATH + '/external'
NUM_CPUS = multiprocessing.cpu_count()

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


def download(mkldnn_version):
    print('Downloading ...')

    os.chdir(MKLDNN_WORK_PATH)
    os.system(
        'git clone -b master \
            --single-branch https://github.com/01org/mkl-dnn.git source')

    os.chdir(MKLDNN_SOURCE_PATH)
    os.system('git reset --hard %s' % mkldnn_version)

    if not os.path.exists(MKLML_PKG_PATH):
        os.system('cd scripts && ./prepare_mkl.sh && cd ..')


def build():
    print('Building ...')

    os.system(
        'mkdir -p build && cd build \
            && cmake -DCMAKE_INSTALL_PREFIX=%s .. \
            && make -j %d' % (MKLDNN_ROOT, NUM_CPUS))


def install(refresh_build):
    print('Installing ...')

    os.chdir(MKLDNN_SOURCE_PATH)

    # install mkldnn
    if refresh_build:
        os.system('cd build && make -j %d && make install' % NUM_CPUS)
    else:
        os.system('cd build && make install')

    # install mklml
    mklml_origin_path = get_mklml_path()
    if mklml_origin_path:
        os.system('cp %s/lib/* %s' % (mklml_origin_path, MKLDNN_LIB_PATH))
        os.system('cp %s/include/* %s' %
                  (mklml_origin_path, MKLDNN_INCLUDE_PATH))
    else:
        sys.exit('%s build error... No Intel mklml pkg.' % MODULE_DESC)


def build_install():
    build()
    install(False)


def download_build_install(mkldnn_version):
    download(mkldnn_version)
    build_install()


def prepare(mkldnn_version):
    print('Intel mkl-dnn preparing ...')
    mkldnn_prepared = True
    mkldnn_built = True
    mkldnn_installed = True

    if os.path.exists(MKLDNN_SOURCE_PATH):
        os.chdir(MKLDNN_SOURCE_PATH)
        res = os.popen('git log | sed -n \'1p\'', 'r')
        commit_head = res.read()
        if mkldnn_version not in commit_head:
            os.chdir(MKLDNN_WORK_PATH)
            mkldnn_prepared = False
        else:
            mklml_origin_path = get_mklml_path()
            if not mklml_origin_path:
                sys.exit('%s build error... No Intel mklml pkg.' % MODULE_DESC)
            include_targets = []
            include_targets += os.listdir(mklml_origin_path + '/include')
            include_targets += os.listdir(MKLDNN_SOURCE_PATH + '/include')

            if not os.path.exists(MKLDNN_BUILD_PATH):
                mkldnn_built = False
            elif not all(os.path.exists(MKLDNN_ROOT + '/lib/' + lib)
                         for lib in lib_targets) or \
                not all(os.path.exists(MKLDNN_ROOT + '/include/' + include)
                        for include in include_targets):
                mkldnn_installed = False
    else:
        mkldnn_prepared = False

    if not mkldnn_prepared:
        download_build_install(mkldnn_version)
    elif not mkldnn_built:
        build_install()
    elif not mkldnn_installed:
        install(True)

    os.chdir(sys.path[0])
    print('Intel mkl-dnn prepared !')
