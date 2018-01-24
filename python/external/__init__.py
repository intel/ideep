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

mkldnn_version = 'ae00102be506ed0fe2099c6557df2aa88ad57ec1'


def prepare():
    if not os.path.exists(EXT_LIB_PATH):
        os.system('mkdir %s' % EXT_LIB_PATH)
    if not os.path.exists(EXT_INCLUDE_PATH):
        os.system('mkdir %s' % EXT_INCLUDE_PATH)

    mkldnn.prepare(mkldnn_version)
    dlcp.prepare()

    if os.path.exists(TARGET_LIB_PATH):
        os.system('rm -rf %s' % TARGET_LIB_PATH)
    os.system('mkdir %s' % TARGET_LIB_PATH)
    os.system('cp %s/*.so* %s' % (EXT_LIB_PATH, TARGET_LIB_PATH))


def clean():
    if os.path.exists(TARGET_LIB_PATH):
        os.system('rm -rf %s' % TARGET_LIB_PATH)
    if os.path.exists(EXT_LIB_PATH):
        os.system('rm -rf %s' % EXT_LIB_PATH)
    if os.path.exists(EXT_INCLUDE_PATH):
        os.system('rm -rf %s' % EXT_INCLUDE_PATH)
    if os.path.exists(EXT_SHARE_PATH):
        os.system('rm -rf %s' % EXT_SHARE_PATH)
