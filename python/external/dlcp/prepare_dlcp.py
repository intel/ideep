import os

DLCP_WORK_PATH = os.path.split(os.path.realpath(__file__))[0]
DLCP_ROOT = DLCP_WORK_PATH + '/..'
DLCP_LIB_PATH = DLCP_ROOT + '/lib'
DLCP_INCLUDE_PATH = DLCP_ROOT + '/include'


def prepare():
    os.system('cp %s/lib/* %s' % (DLCP_WORK_PATH, DLCP_LIB_PATH))
    os.system('cp %s/include/* %s' % (DLCP_WORK_PATH, DLCP_INCLUDE_PATH))
