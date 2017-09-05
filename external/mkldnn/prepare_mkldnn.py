import os
import sys

MKLDNN_ROOT = os.environ['HOME'] + '/.chainer'
MKLDNN_WORK_PATH = os.path.split(os.path.realpath(__file__))[0]
MKLDNN_LIB_PATH = MKLDNN_ROOT + '/lib'
MKLDNN_INCLUDE_PATH = MKLDNN_ROOT + '/include'
MKLDNN_SOURCE_PATH = MKLDNN_WORK_PATH + '/source'
MKLDNN_BUILD_PATH = MKLDNN_WORK_PATH + '/source/build'
MKLML_PKG_PATH = MKLDNN_SOURCE_PATH + '/external'


def exception(msg):
    try:
        sys.exit(0)
    finally:
        print('Warning: %s' % msg)
        print('Cleanup ...')
        os.chdir(MKLDNN_WORK_PATH)
        os.system('rm %s -rf' % MKLDNN_SOURCE_PATH)


def download(mkldnn_version):
    print('Downloading ...')

    os.chdir(MKLDNN_WORK_PATH)
    if os.system('git clone -b master --single-branch https://github.com/01org/mkl-dnn.git source') != 0:
        exception('Could not fork mkldnn source')

    os.chdir(MKLDNN_SOURCE_PATH)
    if os.system('git reset --hard %s' % mkldnn_version) != 0:
        exception('Could not reset to %s' % mkldnn_version)

    if not os.path.exists(MKLML_PKG_PATH):
        if os.system('cd scripts && ./prepare_mkl.sh && cd ..') != 0:
            exception('Fail to prepare mkl')


def build():
    print('Building ...')

    if os.system('mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=%s .. && make -j' % MKLDNN_ROOT) != 0:
        exception('Fail to build mkldnn')


def install(refresh_build):
    print('Installing ...')

    os.chdir(MKLDNN_SOURCE_PATH)

    # install mkldnn
    if refresh_build:
        ret = os.system('cd build && make -j && make install')
    else:
        ret = os.system('cd build && make install')
    if ret != 0:
        exception('Fail to instal mkldnn')

    # install mklml
    mklml_pkg_path_leafs = os.listdir(MKLML_PKG_PATH)
    mklml_origin_path = None
    for leaf in mklml_pkg_path_leafs:
        if os.path.isdir('%s/%s' % (MKLML_PKG_PATH, leaf)) and \
           'mklml' in leaf:
            mklml_origin_path = '%s/%s' % (MKLML_PKG_PATH, leaf)
            break

    if mklml_origin_path:
        os.system('cp %s/lib/* %s' % (mklml_origin_path, MKLDNN_LIB_PATH))
        os.system('cp %s/include/* %s' % (mklml_origin_path, MKLDNN_INCLUDE_PATH))


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
            os.system('rm -rf %s' % MKLDNN_SOURCE_PATH)
            os.system('rm -rf %s' % MKLDNN_LIB_PATH)
            os.system('rm -rf %s' % MKLDNN_INCLUDE_PATH)
            mkldnn_prepared = False
        else:
            if not os.path.exists(MKLDNN_BUILD_PATH):
                os.system('rm -rf %s' % MKLDNN_LIB_PATH)
                os.system('rm -rf %s' % MKLDNN_INCLUDE_PATH)
                mkldnn_built = False
            elif (not os.path.exists(MKLDNN_LIB_PATH)) or \
                 (not os.path.exists(MKLDNN_INCLUDE_PATH)):
                os.system('rm -rf %s' % MKLDNN_LIB_PATH)
                os.system('rm -rf %s' % MKLDNN_INCLUDE_PATH)
                mkldnn_installed = False
    else:
        os.system('rm -rf %s' % MKLDNN_LIB_PATH)
        os.system('rm -rf %s' % MKLDNN_INCLUDE_PATH)
        mkldnn_prepared = False

    if not mkldnn_prepared:
        download_build_install(mkldnn_version)
    elif not mkldnn_built:
        build_install()
    elif not mkldnn_installed:
        install(True)

    os.chdir(sys.path[0])
    print('Intel mkl-dnn prepared !')


def root():
    return MKLDNN_ROOT


def lib_path():
    return MKLDNN_LIB_PATH


def include_path():
    return MKLDNN_INCLUDE_PATH


def source_path():
    return MKLDNN_SOURCE_PATH
