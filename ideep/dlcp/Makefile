##################################################################################
 # # Copyright 2016-2017 Intel Corporation
 # #
 # # Licensed under the Apache License, Version 2.0 (the "License");
 # # you may not use this file except in compliance with the License.
 # # You may obtain a copy of the License at
 # #
 # #     http://www.apache.org/licenses/LICENSE-2.0
 # #
 # # Unless required by applicable law or agreed to in writing, software
 # # distributed under the License is distributed on an "AS IS" BASIS,
 # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # # See the License for the specific language governing permissions and
 # # limitations under the License.
##################################################################################


LOCAL_DIR = $(shell pwd)

#COMPILER ?= gnu
COMPILER ?= intel

FLAG_DEBUG       ?= 0

AR       	= ar
CXXFLAGS    += -fPIC

ifeq ($(COMPILER), intel)
    CC        = icc
    CXX       = icpc
    CXXFLAGS += -std=c++11
    LDFLAGS  += -static-intel
else ifeq ($(COMPILER), gnu)
    CC        = gcc
    CXX       = g++
    CXXFLAGS += -std=c++0x
else
    $(error Unsupported compiler $(COMPILER))
endif

ifeq ($(FLAG_DEBUG), 1)
    CXXFLAGS += -O0 -g
else
    CXXFLAGS += -O2
endif

ifeq ($(CXX), icpc)
    LDFLAGS += -static-intel 
endif

ifeq ($(CXX), icpc)
    CXXFLAGS += -qopenmp
endif

COMPRESSION_LIB		= lib/libdlcomp.so
COMPRESSION_LIBNAME = libdlcomp.so
SRC_DIR				= $(LOCAL_DIR)/src
INCL_DIR			= $(LOCAL_DIR)/include $(LOCAL_DIR)/src

TARGET              = libdlcomp.so
INCS        		= -I$(INCLUDE_DIR) -I$(SRC_DIR)
LDFLAGS             += -ldl -lrt -lpthread -liomp5
CXXFLAGS 			+= $(addprefix -I,$(INCL_DIR)) 


SRCS += src/dl_compression_impl.cpp
SRCS += src/dl_compression_util.cpp
SRCS += src/dl_compression.cpp

OBJS := $(SRCS:.cpp=.o)


all: $(TARGET)

$(TARGET): $(COMPRESSION_LIB)

$(COMPRESSION_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -Wl,-soname,$(COMPRESSION_LIBNAME) -o $(COMPRESSION_LIB) $(OBJS) $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(COMPRESSION_LIB) 

cleanall: clean

