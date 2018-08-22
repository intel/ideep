import time
import numpy
import ideep4py
from ideep4py import distribute
import os

os.system("cat /etc/hostname")
if not distribute.available():
    print ("Distribute feature not built into iDeep,",
           "please use 'cmake -Dmultinode=ON ..' to build ideep")
    exit()

size = 9999999
total_size = 0
nlayer = 10
shape = [None]*nlayer

for layer in range(nlayer):
    shape[layer] = [size/(layer+1)]
    total_size += shape[layer][0]

src_bufs = [None]*nlayer
src_backups = [None]*nlayer
bufs_expect = [None]*nlayer
for layer in range(nlayer):
    src_bufs[layer] = numpy.zeros(shape[layer], numpy.float32)
    src_backups[layer] = numpy.zeros(shape[layer], numpy.float32)
    bufs_expect[layer] = numpy.zeros(shape[layer], numpy.float32)

distribute.init(6)

world_size = distribute.get_world_size()

rank = distribute.get_rank()

for layer in range(nlayer):
    src_bufs[layer] = (
        numpy.full(shape[layer], rank+layer*10, numpy.float32)
        + numpy.linspace(0.0,
                         (shape[layer][0]+0.0)/(shape[layer][0]+1.0),
                         num=shape[layer][0], endpoint=False,
                         dtype=numpy.float32))

for layer in range(nlayer):
    src_bufs[layer] = ideep4py.mdarray(src_bufs[layer])
    src_backups[layer] = ideep4py.mdarray(src_backups[layer])
    ideep4py.basic_copyto(src_backups[layer], src_bufs[layer])

iter_num = 50

distribute.barrier()

# inplace
total = 0.0
for i in range(iter_num):
    for layer in range(nlayer):
        ideep4py.basic_copyto(src_bufs[layer], src_backups[layer])
    start = time.time()
    for layer in range(nlayer):
        distribute.iallreduce(layer, src_bufs[layer])
    for layer in range(nlayer):
        distribute.wait(nlayer-1-layer)
    distribute.barrier()
    end = time.time()
    total = total + end - start


avg_time = total/iter_num
eff_bw = 2.0*(world_size-1)/world_size * total_size * 32 / avg_time/1000000000
print ("[%d] Allreduce done in %f seconds, bw=%fGbps"
       % (rank, avg_time, eff_bw))
distribute.finalize()

if rank == 0:
    print ("Generate expected result...")
for layer in range(nlayer):
    bufs_expect[layer] = (
        numpy.full(shape[layer],
                   (world_size-1)*world_size/2.0 + layer*10*world_size)
        + numpy.linspace(0, shape[layer][0]/(shape[layer][0]+1.0)*world_size,
                         num=shape[layer][0], endpoint=False))

if rank == 0:
    print ("Validate result:")
for layer in range(nlayer):
    numpy.testing.assert_allclose(src_bufs[layer],
                                  bufs_expect[layer],
                                  rtol=1e-06)
if rank == 0:
    print ("pass!")
