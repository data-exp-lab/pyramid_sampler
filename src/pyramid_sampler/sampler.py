import numpy as np
import zarr
from dask import array as da, delayed, compute
import numba
import numpy.typing as npt


@numba.jit
def coarsen(output_shape, level_coarse, level_fine, refine_factor, covered_vals, offset):
    d_level = level_coarse - level_fine

    lev0_Npixels_0 = refine_factor[0] ** d_level
    lev0_Npixels_1 = refine_factor[1] ** d_level
    lev0_Npixels_2 = refine_factor[2] ** d_level

    output_array = np.zeros(output_shape, dtype=np.float64)

    for i0_coarse in range(0, output_shape[0]):
        i0_fine_0 = i0_coarse * refine_factor[0] ** d_level
        i0_fine_1 = i0_fine_0 + lev0_Npixels_0

        for i1_coarse in range(0, output_shape[1]):
            i1_fine_0 = i1_coarse * refine_factor[1] ** d_level
            i1_fine_1 = i1_fine_0 + lev0_Npixels_1
            for i2_coarse in range(0, output_shape[2]):
                i2_fine_0 = i2_coarse * refine_factor[2] ** d_level
                i2_fine_1 = i2_fine_0 + lev0_Npixels_2
                val = 0.0
                nvals = 0.0
                for i0 in range(i0_fine_0, i0_fine_1):
                    for i1 in range(i1_fine_0, i1_fine_1):
                        for i2 in range(i2_fine_0, i2_fine_1):
                            val += covered_vals[i0, i1, i2]
                            nvals += 1.0
                val = float(val / nvals)
                output_array[i0_coarse, i1_coarse, i2_coarse] = val

    return output_array


class Downsampler:
    def __init__(self,
                 zarr_store_path: str,
                 refine_factor: npt.ArrayLike,
                 level_0_res: npt.ArrayLike,
                 chunks: npt.ArrayLike | None = None):

        self.refine_factor = np.as_array(refine_factor).astype(int)
        self.finest_resolution = np.as_array(level_0_res).astype(int)
        assert len(self.refine_factor) == len(self.finest_resolution)
        self.zarr_store_path = zarr_store_path
        self.ndim = len(self.refine_factor)
        if chunks is None:
            chunks = (64,) * self.ndim
        self.chunks = np.as_array(chunks).astype(int)

    def get_fine_ijk(self,
                     ijk_coarse: npt.ArrayLike,
                     level_coarse: int,
                     level_fine:int,) -> npt.ndarray[npt.int64]:

        ijk_coarse = np.as_array(ijk_coarse).astype(int)
        d_level = level_coarse - level_fine
        ijk_0 = ijk_coarse * self.refine_factor**d_level
        return ijk_0.astype(int)

    def get_level_shape(self,
                        level_coarse: int,
                        ) -> npt.ndarray[npt.int64]:
        d_level = level_coarse - 0
        return self.finest_resolution // self.refine_factor**d_level


    def get_global_start_index(self, chunk_linear_index):
        chunks = self.chunks
        n_chunks_by_dim = [len(ch) for ch in chunks]
        chunk_index = np.unravel_index(chunk_linear_index, n_chunks_by_dim)
        ndims = len(chunks)
        si = []
        ei = []
        for idim in range(ndims):
            dim_chunks = np.array(chunks[idim], dtype=int)

            covered_chunks = dim_chunks[0:chunk_index[idim]]
            si.append(np.sum(covered_chunks).astype(int))
            ei.append(si[-1] + chunks[idim][chunk_index[idim]])

        si = np.array(si, dtype=int)
        ei = np.array(ei, dtype=int)
        return si, ei

    def get_level_nchunks(self, level_shape: npt.ArrayLike) -> npt.ndarray[npt.int64]:
        level_shape = np.as_array(level_shape).astype(int)
        return np.array(level_shape) // np.array(self.chunks)

    def get_chunks_by_dim(self, level_shape: npt.ArrayLike):
        chunks = self.chunks
        nchunks = self.get_level_nchunks(level_shape)
        chunksizes = []
        for dim in range(len(chunks)):
            dim_chunks = []
            for ichunk in range(nchunks[dim]):
                dim_chunks.append(chunks[dim])
            chunksizes.append(dim_chunks)
        return np.array(chunksizes, dtype=int)

    def _downsample_by_one_level(self,
                                 coarse_level: int,
                                 zarr_file: str,
                                 zarr_field: str,):

        level = coarse_level
        fine_level = level - 1
        refine_factor = self.refine_factor
        fine_res = self.get_level_shape(fine_level)
        lev_shape = self.get_level_shape(level, fine_res, refine_factor)
        nchunks_by_dim = self.get_level_nchunks(lev_shape)
        if np.any(nchunks_by_dim == 0):
            return None
        chunks_by_dim,  = self.get_chunks_by_dim(lev_shape, self.chunks)

        field1 = zarr.open(zarr_file)[zarr_field]
        field1.empty(level, shape=lev_shape, chunks=self.chunks)

        numchunks = field1[str(level)].nchunks

        chunk_writes = []
        for ichunk in range(0, numchunks):
            chunk_writes.append(
                delayed(_write_chunk_values)(ichunk, chunks_by_dim, zarr_file, zarr_field, level, fine_level))

        _ = compute(*chunk_writes)

def _write_chunk_values(downsampler: Downsampler,
                       ichunk: int,
                       level: int,
                       fine_level: int,
                       zarr_field:str):

    refine_factor = downsampler.refine_factor
    chunks = downsampler.chunks
    zarr_file = downsampler.zarr_store_path

    si, ei = downsampler.get_global_start_index(ichunk, chunks)

    # read in the level 0 range covered by this chunk
    si0 = downsampler.get_fine_ijk(si, level, 0, refine_factor)
    ei0 = downsampler.get_fine_ijk(ei, level, 0, refine_factor)

    fine_zarr = zarr.open(zarr_file)[zarr_field][str(fine_level)]
    covered_vals = fine_zarr[si0[0]:ei0[0], si0[1]:ei0[1], si0[2]:ei0[2]]

    outvals = coarsen(tuple(ei - si), level, fine_level, tuple(refine_factor), covered_vals, tuple(si))

    coarse_zarr = zarr.open(zarr_file)[zarr_field][str(level)]
    coarse_zarr[si[0]:ei[0], si[1]:ei[1]:, si[2]:ei[2]] = outvals

    return 1