import logging

import numpy as np
from westpa.core.we_driver import WEDriver

log = logging.getLogger(__name__)


class MABDriver(WEDriver):
    def assign(self, segments, initializing=False, endprop=False, recycling=False):
        """Assign segments to initial and final bins, and update the (internal) lists of used and available
        initial states. If ``initializing`` is True, then the "final" bin assignments will
        be identical to the initial bin assignments, a condition required for seeding a new iteration from
        pre-existing segments."""

        # collect initial and final coordinates into one place
        all_pcoords = np.empty(
            (2, len(segments), self.system.pcoord_ndim + 1),
            dtype=self.system.pcoord_dtype,
        )
        if recycling:
            for iseg, segment in enumerate(segments):
                all_pcoords[0, iseg] = np.append(segment.pcoord[0, :], -1)
                all_pcoords[1, iseg] = np.append(segment.pcoord[-1, :], -1)
        else:
            for iseg, segment in enumerate(segments):
                all_pcoords[0, iseg] = np.append(segment.pcoord[0, :], segment.weight)
                all_pcoords[1, iseg] = np.append(segment.pcoord[-1, :], segment.weight)
        if endprop:
            np.savetxt("binbounds.txt", all_pcoords[1, :, :])
        # assign based on initial and final progress coordinates
        initial_assignments = self.bin_mapper.assign(all_pcoords[0, :, :])
        if initializing:
            final_assignments = initial_assignments
        else:
            final_assignments = self.bin_mapper.assign(all_pcoords[1, :, :])
        initial_binning = self.initial_binning
        final_binning = self.final_binning
        flux_matrix = self.flux_matrix
        transition_matrix = self.transition_matrix
        for (segment, iidx, fidx) in zip(segments, initial_assignments, final_assignments):
            initial_binning[iidx].add(segment)
            final_binning[fidx].add(segment)
            flux_matrix[iidx, fidx] += segment.weight
            transition_matrix[iidx, fidx] += 1

        n_recycled_total = self.n_recycled_segs
        n_new_states = n_recycled_total - len(self.avail_initial_states)

        log.debug(
            "{} walkers scheduled for recycling, {} initial states available".format(
                n_recycled_total, len(self.avail_initial_states)
            )
        )

        if n_new_states > 0:
            return n_new_states
        else:
            return 0
