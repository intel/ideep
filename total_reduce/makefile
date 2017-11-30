TEST1=reduce_avg_intel
TEST2=workload
TEST3=reduce_int

MPI_INCLUDE=/usr/include/mpich

OBJECTS=pal.o total_reduce.o time.o ring.o pending_message.o compute_request.o payload.o TR_interface.o

CC=gcc-5 -mavx2
LINK=gcc-5 -mavx2 -O3
CC_FLAG=-O3 -Wall -Wextra -Werror \
            -Wno-unused-parameter \
            -Wno-unused-result    \
            -Wno-missing-field-initializers    \
            -c -g -I${MPI_INCLUDE}

LDFLAGS=-g -pthread -ldl

CC_FLAG_MLSL=${CC_FLAG}
LDFLAGS_MLSL=${LDFLAGS}
CC_FLAG_MLSL+= -DUSE_MLSL=1 -Iexternal/mlsl/l_mlsl_2018.0.003/intel64/include
LDFLAGS_MLSL+= -Lexternal/mlsl/l_mlsl_2018.0.003/intel64/lib -lmlsl
WORKLOAD_OBJECTS_MLSL=time.o

LDFLAGS_MPI=${LDFLAGS}
LDFLAGS_MPI+= -lmpi
WORKLOAD_OBJECTS=${OBJECTS}

ifdef DEBUG
    CC_FLAG+= -DDEBUG
endif
ifdef PROFILE
    CC_FLAG+= -DPROFILE=${PROFILE}
endif

HEADERS=compute_request.h  knobs.h  payload.h  pending_message.h  ring.h  time.h  total_reduce.h  TR_interface.h  pal.h

all: ${TEST1} ${TEST2}_inception_v3 \
              ${TEST2}_res50        \
              ${TEST2}_vgg16        \
              ${TEST3}

${TEST1}: test/reduce_avg.o ${OBJECTS}
	${LINK} -o ${TEST1} test/reduce_avg.o ${OBJECTS} ${LDFLAGS_MPI}

${TEST3}: test/reduce_int.o ${OBJECTS}
	${LINK} -o ${TEST3} test/reduce_int.o ${OBJECTS} ${LDFLAGS_MPI}

${TEST2}_inception_v3: test/workload_inception_v3.o ${WORKLOAD_OBJECTS}
	${LINK} -o ${TEST2}_inception_v3 \
                ${WORKLOAD_OBJECTS} ${LDFLAGS_MPI} \
                test/workload_inception_v3.o

${TEST2}_mlsl_inception_v3: test/workload_mlsl_inception_v3.o ${WORKLOAD_OBJECTS_MLSL}
	${LINK} -o ${TEST2}_mlsl_inception_v3 \
                ${WORKLOAD_OBJECTS_MLSL} ${LDFLAGS_MLSL} \
                test/workload_mlsl_inception_v3.o

${TEST2}_vgg16: test/workload_vgg16.o ${WORKLOAD_OBJECTS}
	${LINK} -o ${TEST2}_vgg16 test/workload_vgg16.o ${WORKLOAD_OBJECTS} ${LDFLAGS_MPI}

${TEST2}_mlsl_vgg16: test/workload_mlsl_vgg16.o ${WORKLOAD_OBJECTS_MLSL}
	${LINK} -o ${TEST2}_mlsl_vgg16 test/workload_mlsl_vgg16.o ${WORKLOAD_OBJECTS_MLSL} ${LDFLAGS_MLSL}

${TEST2}_res50: test/workload_res50.o ${WORKLOAD_OBJECTS}
	${LINK} -o ${TEST2}_res50 test/workload_res50.o ${WORKLOAD_OBJECTS} ${LDFLAGS_MPI}

${TEST2}_mlsl_res50: test/workload_mlsl_res50.o ${WORKLOAD_OBJECTS_MLSL}
	${LINK} -o ${TEST2}_mlsl_res50 test/workload_mlsl_res50.o ${WORKLOAD_OBJECTS_MLSL} ${LDFLAGS_MLSL}

test/reduce_avg.o: test/reduce_avg.c ${HEADERS}
	${CC} ${CC_FLAG} test/reduce_avg.c -o test/reduce_avg.o

test/reduce_int.o: test/reduce_int.c ${HEADERS}
	${CC} ${CC_FLAG} test/reduce_int.c -o test/reduce_int.o

test/workload_inception_v3.o: test/workload.cpp ${HEADERS}
	${CC} ${CC_FLAG} test/workload.cpp -DEVENT_NAME=inception_v3 -o test/workload_inception_v3.o

test/workload_vgg16.o: test/workload.cpp ${HEADERS}
	${CC} ${CC_FLAG} test/workload.cpp -DEVENT_NAME=vgg16 -o test/workload_vgg16.o

test/workload_res50.o: test/workload.cpp ${HEADERS}
	${CC} ${CC_FLAG} test/workload.cpp -DEVENT_NAME=res50 -o test/workload_res50.o

test/workload_mlsl_inception_v3.o: test/workload.cpp ${HEADERS}
	${CC} ${CC_FLAG_MLSL} test/workload.cpp -DEVENT_NAME=inception_v3 -o test/workload_mlsl_inception_v3.o

test/workload_mlsl_vgg16.o: test/workload.cpp ${HEADERS}
	${CC} ${CC_FLAG_MLSL} test/workload.cpp -DEVENT_NAME=vgg16 -o test/workload_mlsl_vgg16.o

test/workload_mlsl_res50.o: test/workload.cpp ${HEADERS}
	${CC} ${CC_FLAG_MLSL} test/workload.cpp -DEVENT_NAME=res50 -o test/workload_mlsl_res50.o

total_reduce.o: total_reduce.c ${HEADERS}
	${CC} ${CC_FLAG} total_reduce.c -o total_reduce.o

pal.o: pal.c ${HEADERS}
	${CC} ${CC_FLAG} pal.c -o pal.o

time.o: time.c ${HEADERS}
	${CC} ${CC_FLAG} time.c -o time.o

ring.o: ring.c ${HEADERS}
	${CC} ${CC_FLAG} ring.c -o ring.o

pending_message.o: pending_message.c ${HEADERS}
	${CC} ${CC_FLAG} pending_message.c -o pending_message.o

compute_request.o: compute_request.c ${HEADERS}
	${CC} ${CC_FLAG} compute_request.c -o compute_request.o

payload.o: payload.c ${HEADERS}
	${CC} ${CC_FLAG} payload.c -o payload.o

TR_interface.o: TR_interface.c ${HEADERS}
	${CC} ${CC_FLAG} TR_interface.c -o TR_interface.o

clean:
	rm -f ${TEST1} ${OBJECTS} test/*.o
