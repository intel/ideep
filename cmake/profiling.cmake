#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if(profiling_ideep_included)
    return()
endif()
set(profiling_ideep_included true)

if("${VTUNEROOT}" STREQUAL "")
    message(STATUS "VTune profiling environment is unset")
else()
  #     MKL-DNN will contain vtune_lib
    if (WIN32)
        set(jitlibname "jitprofiling.lib")
    else()
        set(jitlibname "libjitprofiling.a")
    endif()

    if (APPLE)
        link_directories("${VTUNEROOT}/Frameworks")
    else()
        link_directories("${VTUNEROOT}/lib64")
    endif()

    set(itt_libname "ittnotify")
    list(APPEND vtune_include "${VTUNEROOT}/include")
    add_definitions(-DPROFILE_ENABLE)
    message(STATUS "VTune profiling environment is set")
endif()
