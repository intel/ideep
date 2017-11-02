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
    LDFLAGS += -static-intel -qopenmp
endif

ifeq ($(CXX), icpc)
    CXXFLAGS += 
endif

COMPRESSION_LIB		= libdlcomp.so
SRC_DIR				= $(LOCAL_DIR)
INCL_DIR			= $(LOCAL_DIR)/../include 

TARGET              = test_compress_lib
EXECUTE				= test
INCS        		= -I$(INCLUDE_DIR) -I$(SRC_DIR)
LDFLAGS             += -L$(LOCAL_DIR)/../lib -ldlcomp 
CXXFLAGS 			+= $(addprefix -I,$(INCL_DIR)) 


SRCS += main.cpp

OBJS := $(SRCS:.cpp=.o)


all: $(TARGET)

$(TARGET): $(EXECUTE)

$(EXECUTE): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXECUTE) $(OBJS) $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(EXECUTE) 

cleanall: clean

