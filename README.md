# pyramid_sampler

A small utility for taking a 3D zarr image at a single resolution and
downsampling to create an image pyramid.

## installation

```
python -m pip install pyramid-sampler
```

## usage

### create a test base image

```python
import zarr
from pyramid_sampler import initialize_test_image

# create an on-disk zarr store
zarr_file = "test_image.zarr"
zarr_store = zarr.group(zarr_file)

# write base level 0 image to the specified store and field name
new_field = "field1"
base_res = (1024, 1024, 1024)
chunks = (64, 64, 64)
initialize_test_image(zarr_store, new_field, base_res, chunks=chunks)
```

`initialize_test_image` will utilize a `dask.delayed` workflow, so you can
configure a dask client prior to calling `initilize_test_iamge` if you wish. The
resulting base image will reside in `zarr_store[new_field][0]`.

### downsampling an image

First initialize a `Downsampler` instance with the path to the zarr store, the
refinement factor to use between image levels, the base image resolution and
chunksizes:

```python
from pyramid_sampler import Downsampler

zarr_file = "test_image.zarr"
refinement_factor = (2, 2, 2)
base_res = (1024, 1024, 1024)
chunks = (64, 64, 64)
ds = Downsampler(zarr_file, (2, 2, 2), base_res, chunks)
```

For now, this assumes your base image will reside in `zarr_file/field_name/0`.
To run the downsampling,

```python
field_to_downsample = "field1"
max_levels = 10
ds.downsample(max_levels, field_to_downsample)
```

Downsampling will only proceed until a layer is created with a single chunk of
size set by the `Downsampler` and image chunksize, i.e., until
`base_resolution / refinement_factor**current_level / chunks` has a value of 1
in any dimension, or
`max_levels = log(base_resolution/chunks) / log(refinement_factor)` (giving a
max level id of `max_levels - 1` to account for 0-indexing).

### assumptions

Some assumptions in the current algorithm:

- exact chunks only! The base image resoultion and `chunks` must perfectly
  subdivide and downsampling must result in an even number of chunks.
- the base field exists and is stored at `zarr_store[field][0]`
- Only tested with on-disk filestores, but should work for any zarr store.

## method

at present, the downsampling simply averages overlapping array elements: for a
given image level, `L1`, the pixels of the higher resoluiton image level
`L1 - 1` covered by each pixel in `L1` are found and averaged. Levels are built
up sequentially (i.e., `L1` is built from `L1 - 1`, not the base resolution).

Calculations and chunk-processing are accelerated with dask delayed objects
using `numba.jit` compilation for the pixel-averaging.

## developing & contributing

At present, this package is a small utility used for experimentations with zarr
files. But contributions are welcome! Open up an issue at
https://github.com/data-exp-lab/pyramid_sampler/issues to discuss ideas.

### cutting a new release

Notes for maintainers on cutting a new release:

1. create and push a new tag

```commandline
git tag v0.1.0
git push upstream v0.1.0
```

2. create new a release on github via the release interface, using the tag you
   just pushed.
3. on publishing the new github release, a github action
   `.github/workflows/cd.yml` runs. This action builds the distribution and
   pushes to pypi.

Note that the publication to pypi in step 3 uses a
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/) configured by
@chris.havlin
