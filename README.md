# wea.py

The Wrapped Exchange Array is a convenient way in order to exchange array data easily via processes or remote nodes. Inspired and adopted partly from Julia’s [InterProcessCommunication](https://github.com/emmt/InterProcessCommunication.jl) WrappedArray.

The wrapped exchange array can be accessed like a numpy array because under the hood, numpy is applied.

## Getting started

Install the package from Pypi

```bash
pip install wea
```

## Shared memory

```python
import wea

...
wa = wea.shared_memory.create_wrapped_array('/awesome-1', np.dtype('float64'), (10, 2))
wa[:] = my_new_data[:]
...
```

### Creating a shared memory segement

In order to create a new shared memory segment, use the following snippet

```python
import wea
import numpy

type = np.dtype('float64')
dims = (10, 2)
wa = wea.shared_memory.create_wrapped_array('/awesome-1', type, dims)
wa[:] = np.random.randn(dims[0], dims[1])
```

If creating was not possible because the segment already exists , a `FileExistsError` exception will be thrown.

### Attaching to an existing shared memory array

If a wrapped exchange array was already created, you can attach to it simply by

```python
import wea
import numpy

wa = wea.shared_memory.attach_wrapped_array('/awesome-1')
wa[:] = np.random.randn(dims[0], dims[1])
```

The metadata of the array are stored in the shared memory header segment and will be retrieved for the numpy array creation.

If attaching was not possible because the segment does not exist so far, a `FileNotFoundError` exception will be thrown.
