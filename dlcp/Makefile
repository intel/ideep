##################################################################################
 # # Copyright (c) 2018 Intel Corporation.
 # #
 # # Permission is hereby granted, free of charge, to any person obtaining a copy
 # # of this software and associated documentation files (the "Software"), to deal
 # # in the Software without restriction, including without limitation the rights
 # # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # # copies of the Software, and to permit persons to whom the Software is
 # # furnished to do so, subject to the following conditions:
 # #
 # # The above copyright notice and this permission notice shall be included in
 # # all copies or substantial portions of the Software.
 # #
 # # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # # THE SOFTWARE.
 # #
##################################################################################


LOCAL_DIR = $(shell pwd)

DLCP_CXX ?= icpc

FLAG_DEBUG ?= 0

AR       	= ar
CXXFLAGS    += -fPIC

ifneq (,$(findstring icpc, $(DLCP_CXX)))
    CXX       = $(DLCP_CXX)
    CXXFLAGS += -std=c++11
    LDFLAGS  += -static-intel
else
    $(error Unsupported compiler $(DLCP_CXX))
endif

ifeq ($(FLAG_DEBUG), 1)
    CXXFLAGS += -O0 -g
else
    CXXFLAGS += -O2
endif

ifneq (,$(findstring icpc,$(CXX)))
    LDFLAGS += -static-intel 
endif

ifneq (,$(findstring icpc,$(CXX)))
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

