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
import numpy as np

...
wa = wea.shared_memory.create_shared_array('/awesome-1', np.dtype('float64'), (10, 2))
wa[:] = my_new_data[:]
...
```

### Creating a shared memory segment

In order to create a new shared memory segment, use the following snippet

```python
import wea
import numpy as np

type = np.dtype('float64')
dims = (10, 2)
wa = wea.shared_memory.create_shared_array('/awesome-1', type, dims)
wa[:] = np.random.randn(dims[0], dims[1])
```

If creating was not possible because the segment already exists , a `FileExistsError` exception will be thrown.

### Attaching to an existing shared memory array

If a wrapped exchange array was already created, you can attach to it simply by

```python
import wea
import numpy as np

wa = wea.shared_memory.attach_shared_array('/awesome-1')
wa[:] = np.random.randn(dims[0], dims[1])
```

The metadata of the array are stored in the shared memory header segment and will be retrieved for the numpy array creation.

If attaching was not possible because the segment does not exist so far, a `FileNotFoundError` exception will be thrown.

## Bytearray buffer memory

```python
import wea
import numpy as np

...
wa = wea.buffered_memory.create_buffered_array(np.dtype('float64'), (10, 2))
wa[:] = my_new_data[:]
buf = wa.exchange buffer
share(buf) # where share calls your prefered communication protocol
...
```

### Creating a buffered memory segment

In order to create a new buffered memory segment, use the following snippet

```python
import wea
import numpy as np

type = np.dtype('float64')
dims = (10, 2)
wa = wea.buffered_memory.create_buffered_array(type, dims)
wa[:] = np.random.randn(dims[0], dims[1])
buf: bytearray = wa.exchange_buffer
```

Actually it copies the content from the numpy array into the buffer. Thus, the current behavior is like a deep copy.

### Loading from an existing buffered memory segment

If a wrapped exchange array was already created, you can load from it simply by

```python
import wea
import numpy as np

buf: bytearray = receive() # where receive via your prefered communication protocol
wa = wea.buffered_memory.load_buffered_array(buf)
```

The metadata of the array are stored in the buffered memory header segment and will be retrieved for the numpy array creation.
