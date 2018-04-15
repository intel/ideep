import numpy
from ideep4py import mdarray

n = numpy.ndarray((2, 2, 2, 2), dtype=numpy.float32)
# create memory, entity array
print('\nCreate a')
a = mdarray(n)
print('\nset a[0][0][0][0] 2222.22')
a[0][0][0][0] = 2222.22
# share memory, view array
print('\nCreate b')
b = mdarray(a)
# share memory from view array
print('\nCreate c')
c = mdarray(b)
print('\nDel b')
del b
print('Del a')
del a

# dirty the memory
print('\nCreate d')
d = numpy.ndarray((2, 2, 2, 2), dtype=numpy.float32)
d = mdarray(d)
print('\nbefore dirty d[0][0][0][0] %f' % d[0][0][0][0])
d[0][0][0][0] = 3333.33

print('\nset d[0][0][0][0] %f' % d[0][0][0][0])
print('\nc[0][0][0][0] is dirty %f' % c[0][0][0][0])
# Clear context
print('\nDel c')
del c
print('\nDel d')
del d

# Test free array
print('\n\nTest free array')
n = numpy.ndarray((2, 2, 2, 2), dtype=numpy.float32)
# create memory, entity array
print('\nCreate a')
a = mdarray(n)
# share memory, view array
print('\nCreate b')
b = mdarray(a)
# share memory from view array
print('\nCreate c')
c = mdarray(b)
print('\nDel b')
del b
print('\nCreate d')
d = mdarray(c)
print('Del a')
del a
print('Del d')
del d
print('Del c')
del c
