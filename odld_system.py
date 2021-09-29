from __future__ import print_function, division
from os.path import isfile

import numpy as np
from numpy.random import normal as random_normal

from westpa.core.binning import RectilinearBinMapper
from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem

from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem
from westpa.core.binning import RectilinearBinMapper
from westpa.core.binning import FuncBinMapper
from westpa.core.binning import RecursiveBinMapper

import logging

PI = np.pi
log = logging.getLogger("westpa.rc")

pcoord_len = 21
pcoord_dtype = np.float32
# THESE ARE THE FOUR THINGS YOU SHOULD CHANGE
bintargetcount = 10  # number of walkers per bin
numberofdim = 1  # number of dimensions
binsperdim = [10]  # You will have prod(binsperdim)+numberofdim*(2+2*splitIsolated)+activetarget bins total
pcoordlength = 2  # length of the pcoord
do_pca = False  # choose to do principal component analysis
targetstate = [1.9]  # enter boundaries for target state or None if there is no target state in that dimension
targetstatedirection = -1  # if your target state is meant to be greater that the starting pcoor use 1 or else use -1
activetarget = 1  # if no target state make this zero
splitIsolated = True  # True if you want to split the most isolated walker (this will add an extra bin)


#########
def map_mab(coords, mask, output, *args, **kwargs):
    if not np.any(mask):
        return output

    allcoords = np.copy(coords)
    allmask = np.copy(mask)

    weights = None
    isfinal = None
    splitting = False
    if coords.shape[1] > numberofdim:
        isfinal = allcoords[:, numberofdim + 1].astype(np.bool_)
        coords = coords[isfinal, :numberofdim]
        weights = allcoords[isfinal, numberofdim + 0]
        mask = mask[isfinal]
        splitting = True

    n_segments = coords.shape[0]

    varcoords = np.copy(coords)
    originalcoords = np.copy(coords)
    if do_pca and len(output) > 1:
        colavg = np.mean(coords, axis=0)
        for i in range(len(coords)):
            for j in range(len(coords[i])):
                varcoords[i][j] = coords[i][j] - colavg[j]
        covcoords = np.cov(np.transpose(varcoords))
        eigval, eigvec = np.linalg.eigh(covcoords)
        eigvec = eigvec[:, np.argmax(np.absolute(eigvec), axis=1)]
        for i in range(len(eigvec)):
            if eigvec[i, i] < 0:
                eigvec[:, i] = -1 * eigvec[:, i]
        for i in range(numberofdim):
            for j in range(len(output)):
                coords[j][i] = np.dot(varcoords[j], eigvec[:, i])

    maxlist = []
    minlist = []
    difflist = []
    flipdifflist = []
    for n in range(numberofdim):
        # identify the boundary segments
        currentmax = np.amax(coords[mask, n])
        currentmin = np.amin(coords[mask, n])
        maxlist.append(currentmax)
        minlist.append(currentmin)

        # detect the bottleneck segments
        if splitting:
            temp = np.column_stack((originalcoords[mask, n], weights[mask]))
            I = temp[:, 0].argsort()
            temp = temp[I]
            seg_indices = np.arange(n_segments)[mask][I]
            for p in range(len(temp)):
                if temp[p][1] == 0:
                    temp[p][1] = 10 ** -39
            fliptemp = np.flipud(temp)
            flip_seg_indices = np.flipud(seg_indices)

            difflist.append(None)
            flipdifflist.append(None)
            maxdiff = 0
            flipmaxdiff = 0
            for i in range(1, len(temp) - 1):
                comprob = 0
                flipcomprob = 0
                j = i + 1
                while j < len(temp):
                    comprob = comprob + temp[j][1]
                    flipcomprob = flipcomprob + fliptemp[j][1]
                    j = j + 1
                diff = -np.log(comprob) + np.log(temp[i][1])
                if diff > maxdiff:
                    difflist[n] = seg_indices[i]
                    maxdiff = diff
                flipdiff = -np.log(flipcomprob) + np.log(fliptemp[i][1])
                if flipdiff > flipmaxdiff:
                    flipdifflist[n] = flip_seg_indices[i]
                    flipmaxdiff = flipdiff

    # assign segments to bins
    for i in range(len(output)):
        if not allmask[i]:
            continue

        holder = 2 * numberofdim
        for n in range(numberofdim):
            if (activetarget == 1) and targetstate[n] is not None:
                if (allcoords[i, n] * targetstatedirection) >= (
                    targetstate[n] * targetstatedirection
                ):
                    holder = np.prod(binsperdim) + numberofdim * 2
            if holder == np.prod(binsperdim) + numberofdim * 2:
                break
            elif splitting and splitIsolated:
                if i == difflist[n]:
                    holder = np.prod(binsperdim) + numberofdim * 2 + 2 * n + activetarget
                    break
                elif i == flipdifflist[n]:
                    holder = np.prod(binsperdim) + numberofdim * 2 + 2 * n + activetarget + 1
                    break

        if holder == 2 * numberofdim:
            for j in range(numberofdim):
                holder = holder + (
                    np.digitize(
                        allcoords[i][j],
                        np.linspace(minlist[j], maxlist[j], binsperdim[j] + 1),
                    )
                    - 1
                ) * np.prod(binsperdim[0:j])
        output[i] = holder
    return output


class ODLDPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super().__init__(rc)

        self.coord_len = pcoord_len
        self.coord_dtype = pcoord_dtype
        self.coord_ndim = 1

        self.initial_pcoord = np.array([9.5], dtype=self.coord_dtype)

        self.sigma = 0.001 ** (0.5)  # friction coefficient

        self.A = 2
        self.B = 10
        self.C = 0.5
        self.x0 = 1

        # Implement a reflecting boundary at this x value
        # (or None, for no reflection)
        self.reflect_at = 10.0

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""
        state.pcoord = self.initial_pcoord.copy()

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = self.initial_pcoord.copy()
        initial_state.istate_status = initial_state.ISTATE_STATUS_PREPARED
        return initial_state

    def propagate(self, segments):

        A, B, C, x0 = self.A, self.B, self.C, self.x0

        n_segs = len(segments)

        coords = np.empty(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )

        for iseg, segment in enumerate(segments):
            coords[iseg, 0] = segment.pcoord[0]

        twopi_by_A = 2 * PI / A
        half_B = B / 2
        sigma = self.sigma
        gradfactor = self.sigma * self.sigma / 2
        coord_len = self.coord_len
        reflect_at = self.reflect_at
        all_displacements = np.zeros(
            (n_segs, self.coord_len, self.coord_ndim), dtype=self.coord_dtype
        )
        for istep in range(1, coord_len):
            x = coords[:, istep - 1, 0]

            xarg = twopi_by_A * (x - x0)

            eCx = np.exp(C * x)
            eCx_less_one = eCx - 1.0

            all_displacements[:, istep, 0] = displacements = random_normal(
                scale=sigma, size=(n_segs,)
            )
            grad = (
                half_B
                / (eCx_less_one * eCx_less_one)
                * (twopi_by_A * eCx_less_one * np.sin(xarg) + C * eCx * np.cos(xarg))
            )

            newx = x - gradfactor * grad + displacements
            if reflect_at is not None:
                # Anything that has moved beyond reflect_at must move back that much

                # boolean array of what to reflect
                to_reflect = newx > reflect_at

                # how far the things to reflect are beyond our boundary
                reflect_by = newx[to_reflect] - reflect_at

                # subtract twice how far they exceed the boundary by
                # puts them the same distance from the boundary, on the other side
                newx[to_reflect] -= 2 * reflect_by
            coords[:, istep, 0] = newx

        for iseg, segment in enumerate(segments):
            segment.pcoord[...] = coords[iseg, :]
            segment.data["displacement"] = all_displacements[iseg]
            segment.status = segment.SEG_STATUS_COMPLETE

        return segments


class ODLDSystem(WESTSystem):
    def initialize(self):
        self.pcoord_ndim = 1
        self.pcoord_dtype = pcoord_dtype
        self.pcoord_len = pcoord_len

        outer_mapper = RectilinearBinMapper([[0, 2, 6, 8, 10]])

        adaptive_mapper = FuncBinMapper(
            map_mab,
            np.prod(binsperdim) + numberofdim * (2 + 2 * splitIsolated) + activetarget,
        )

        self.bin_mapper = RecursiveBinMapper(outer_mapper)
        self.bin_mapper.add_mapper(adaptive_mapper, [7])

        # self.bin_mapper = RectilinearBinMapper([[0,1.3] + list(np.arange(1.4, 10.1, 0.1)) + [float('inf')]])
        # self.bin_mapper = RectilinearBinMapper([list(np.arange(0.0, 10.1, 0.1))])
        self.bin_target_counts = np.empty((self.bin_mapper.nbins,), np.int_)
        self.bin_target_counts[...] = 10
