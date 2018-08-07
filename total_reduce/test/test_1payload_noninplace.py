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
src_buf = numpy.zeros(shape, numpy.float32)
dst_buf = numpy.zeros(shape, numpy.float32)
buf_expect = numpy.zeros(shape, numpy.float32)

print ("Initialize distributed computation")
distribute.init()

world_size = distribute.get_world_size()
print ("world size = %d" % (world_size))

rank = distribute.get_rank()
print ("rank = %d" % (rank))

for i in range(shape[0]):
    src_buf[i] = (i+0.0)/(size+1) + rank

src_buf = ideep4py.mdarray(src_buf)
dst_buf = ideep4py.mdarray(dst_buf)

iter_num = 1
start = time.time()

# non-inplace
for i in range(iter_num):
    distribute.allreduce(0, src_buf, dst_buf)
    distribute.barrier()

end = time.time()
print ("[%d] Allreduce done in %f seconds" % (rank, (end-start)/(iter_num+1)))
distribute.finalize()

if rank == 0:
    print ("Generate expected result...")
for r in range(world_size):
    for i in range(shape[0]):
        buf_expect[i] += (i+0.0)/(shape[0]+1) + r

if rank == 0:
    print ("[%d] Validate non-inplace result:" % (rank))
    numpy.testing.assert_allclose(dst_buf, buf_expect, rtol=1e-06)

if rank == 0:
    print ("pass!")
