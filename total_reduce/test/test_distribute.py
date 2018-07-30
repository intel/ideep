import time
import numpy
import ideep4py
from ideep4py import distribute

if not distribute.available():
    print ("Distribute feature not built into iDeep,",
           "please use 'cmake -Dmultinode=ON ..' to build ideep")
    exit()

size = 999999
shape = [size]
buf = numpy.zeros(shape, numpy.float32)
buf_expect = numpy.zeros(shape, numpy.float32)

print ("Initialize distributed computation")
distribute.init()

world_size = distribute.get_world_size()
print ("world size = ", world_size)

rank = distribute.get_rank()
print ("rank = ", rank)

for i in range(shape[0]):
    buf[i] = i/(size+1) + rank

for r in range(world_size):
    for i in range(shape[0]):
        buf_expect[i] += i/(size+1) + r

buf = ideep4py.mdarray(buf)

start = time.time()
distribute.allreduce(0, 0, buf)
end = time.time()
print ("[", rank, "] Allreduce done in", end-start, "seconds")

numpy.testing.assert_allclose(buf, buf_expect, rtol=1e-06)

distribute.finalize()
