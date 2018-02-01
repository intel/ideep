import external.mkldnn as mkldnn  # NOQA
import external.dlcp as dlcp  # NOQA

import os

EXT_LIB_PATH =\
    os.path.split(os.path.realpath(__file__))[0] + '/lib'
EXT_INCLUDE_PATH =\
    os.path.split(os.path.realpath(__file__))[0] + '/include'
EXT_SHARE_PATH =\
    os.path.split(os.path.realpath(__file__))[0] + '/share'
TARGET_LIB_PATH =\
    os.path.split(os.path.realpath(__file__))[0] + '/../ideep4py/lib'

target_libs = [
    # 'libdlcomp.so',
    'libiomp5.so',
    'libmkldnn.so*',
]

mkldnn_version = 'ae00102be506ed0fe2099c6557df2aa88ad57ec1'


def prepare():
    if not os.path.exists(EXT_LIB_PATH):
        os.system('mkdir %s' % EXT_LIB_PATH)
    if not os.path.exists(EXT_INCLUDE_PATH):
        os.system('mkdir %s' % EXT_INCLUDE_PATH)

    mkldnn.prepare(mkldnn_version)
    # dlcp.prepare()

    if os.path.exists(TARGET_LIB_PATH):
        os.system('rm -rf %s' % TARGET_LIB_PATH)
    os.system('mkdir %s' % TARGET_LIB_PATH)
    libmklml = os.popen(
        'ldd external/lib/libmkldnn.so |\
        grep libmklml | awk \'{print $1}\'').read()
    global target_libs
    target_libs += [libmklml[:-1]]
    for lib in target_libs:
        os.system('cp %s/%s %s' % (EXT_LIB_PATH, lib, TARGET_LIB_PATH))


def clean():
    if os.path.exists(TARGET_LIB_PATH):
        os.system('rm -rf %s' % TARGET_LIB_PATH)
    if os.path.exists(EXT_LIB_PATH):
        os.system('rm -rf %s' % EXT_LIB_PATH)
    if os.path.exists(EXT_INCLUDE_PATH):
        os.system('rm -rf %s' % EXT_INCLUDE_PATH)
    if os.path.exists(EXT_SHARE_PATH):
        os.system('rm -rf %s' % EXT_SHARE_PATH)
