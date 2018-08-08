import time
import numpy
import ideep4py
from ideep4py import distribute

if not distribute.available():
    print ("Distribute feature not built into iDeep,",
           "please use 'cmake -Dmultinode=ON ..' to build ideep")
    exit()

size = 9999999
shape = [size]
src_buf = numpy.zeros(shape, numpy.float32)
src_backup = numpy.zeros(shape, numpy.float32)
buf_expect = numpy.zeros(shape, numpy.float32)

print ("Initialize distributed computation")
distribute.init()

world_size = distribute.get_world_size()
print ("world size = %d" % (world_size))

rank = distribute.get_rank()
print ("rank = %d" % (rank))

for i in range(shape[0]):
    src_buf[i] = float(i)/(size+1) + rank

src_buf = ideep4py.mdarray(src_buf)
src_backup = ideep4py.mdarray(src_backup)
ideep4py.basic_copyto(src_backup, src_buf)

iter_num = 10
start = time.time()


def cb(id):
    print ("callback from payload %d in rank %d" % (id, distribute.get_rank()))


# inplace
for i in range(iter_num):
    ideep4py.basic_copyto(src_buf, src_backup)
    distribute.iallreduce(1, src_buf, cb)
    distribute.barrier()

end = time.time()

avg_time = (end-start)/iter_num
eff_bw = 2.0*(world_size-1)/world_size * shape[0] * 32 / avg_time/1000000000
print ("[%d] Allreduce done in %f seconds, bw=%fGbps"
       % (rank, avg_time, eff_bw))

distribute.finalize()

if rank == 0:
    print ("Generate expected result...")
for r in range(world_size):
    for i in range(shape[0]):
        buf_expect[i] += (i+0.0)/(shape[0]+1) + r

if rank == 0:
    print ("[%d] Validate inplace result:" % (rank))

numpy.testing.assert_allclose(src_buf, buf_expect, rtol=1e-06)

if rank == 0:
    print ("pass!")
