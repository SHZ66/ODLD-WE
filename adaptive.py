import numpy as np
from westpa.core.binning import FuncBinMapper


def map_mab(coords, mask, output, *args, **kwargs):
    pca = kwargs.pop("pca", False)
    bottleneck = kwargs.pop("bottleneck", True)
    nbins_per_dim = kwargs.get("nbins_per_dim")
    ndim = len(nbins_per_dim)

    if not np.any(mask):
        return output

    allcoords = np.copy(coords)
    allmask = np.copy(mask)

    weights = None
    isfinal = None
    splitting = False
    if coords.shape[1] > ndim:
        if coords.shape[1] > ndim + 1:
            isfinal = allcoords[:, ndim + 1].astype(np.bool_)
        else:
            isfinal = np.ones(coords.shape[0], dtype=np.bool_)
        coords = coords[isfinal, :ndim]
        weights = allcoords[isfinal, ndim + 0]
        mask = mask[isfinal]
        splitting = True

    # in case where there is no final segments but initial ones in range
    if not np.any(mask):
        coords = allcoords[:, :ndim]
        mask = allmask
        weights = None
        splitting = False

    n_segments = coords.shape[0]

    varcoords = np.copy(coords)
    originalcoords = np.copy(coords)
    if pca and len(output) > 1:
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
        for i in range(ndim):
            for j in range(len(output)):
                coords[j][i] = np.dot(varcoords[j], eigvec[:, i])

    maxlist = []
    minlist = []
    difflist = []
    flipdifflist = []
    for n in range(ndim):
        # identify the boundary segments
        seg_indices = np.arange(n_segments)[mask]
        maxi = np.argmax(coords[mask, n])
        mini = np.argmin(coords[mask, n])
        maxlist.append(seg_indices[maxi])
        minlist.append(seg_indices[mini])

        # detect the bottleneck segments
        if splitting:
            temp = np.column_stack((originalcoords[mask, n], weights[mask]))
            sorted_indices = temp[:, 0].argsort()
            temp = temp[sorted_indices]
            sorted_indices = seg_indices[sorted_indices]
            for p in range(len(temp)):
                if temp[p][1] == 0:
                    temp[p][1] = 10 ** -39
            fliptemp = np.flipud(temp)
            flip_seg_indices = np.flipud(sorted_indices)

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
                    difflist[n] = sorted_indices[i]
                    maxdiff = diff
                flipdiff = -np.log(flipcomprob) + np.log(fliptemp[i][1])
                if flipdiff > flipmaxdiff:
                    flipdifflist[n] = flip_seg_indices[i]
                    flipmaxdiff = flipdiff

    # assign segments to bins
    # the total number of linear bins + 2 boundary bins each dim
    bottleneck_base = np.prod(nbins_per_dim) + 2 * ndim
    for i in range(len(output)):
        if not allmask[i]:
            continue

        special = False
        for n in range(ndim):
            if splitting and bottleneck:
                if i == difflist[n]:
                    holder = bottleneck_base + 2 * n
                    special = True
                    break
                elif i == flipdifflist[n]:
                    holder = bottleneck_base + 2 * n + 1
                    special = True
                    break
            if i == minlist[n]:
                holder = 0
                special = True
                break
            elif i == maxlist[n]:
                holder = nbins_per_dim[n] + 1
                special = True
                break

        if not special:
            holder = 0
            for n in range(ndim):
                nbins = nbins_per_dim[n]
                minidx = minlist[n]
                maxidx = maxlist[n]
                coord = allcoords[i][n]

                minp = allcoords[minidx, n]
                maxp = allcoords[maxidx, n]
                bins = np.linspace(minp, maxp, nbins + 1)
                bin_number = np.digitize(coord, bins)

                # unlikely to happen unless two max pcoords are exactly the same
                if bin_number >= nbins:
                    bin_number = nbins - 1
                if bin_number < 0:
                    bin_number = 0

                holder += bin_number * np.prod(nbins_per_dim[:n])
        output[i] = holder

    for n in range(ndim):
        mini = minlist[n]
        maxi = maxlist[n]
        print(f"[MAB] Boundaries for dim{n}: {allcoords[mini, n]} - {allcoords[maxi, n]}")
        if splitting and bottleneck:
            fi = difflist[n]
            bi = flipdifflist[n]

            print(f"[MAB] Bottlenecks for dim{n} (forward):")
            if fi is not None:
                print(f"{allcoords[fi, n]}")
            else:
                print("None detected")

            print(f"[MAB] Bottlenecks for dim{n} (backward):")
            if bi is not None:
                print(f"{allcoords[bi, n]}")
            else:
                print("None detected")

    return output


class MABBinMapper(FuncBinMapper):
    def __init__(self, nbins, bottleneck=True, pca=False):
        kwargs = dict(nbins_per_dim=nbins,
                      bottleneck=bottleneck,
                      pca=pca)
        ndim = len(nbins)
        n_total_bins = np.prod(nbins) + ndim * (2 + 2 * bottleneck)
        super().__init__(map_mab, n_total_bins, kwargs=kwargs)
