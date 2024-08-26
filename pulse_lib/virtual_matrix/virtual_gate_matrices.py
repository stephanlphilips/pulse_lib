import numpy as np
import logging

from .virtual_gate_matrix import VirtualGateMatrix

logger = logging.getLogger(__name__)

class VirtualGateMatrices:
    '''
    Collection of virtual gate matrices.
    '''
    def __init__(self):
        self._virtual_matrices = {}
        self._projection_cache_physical_channels = None
        self._projection_cache_matrices = []
        self._projection_cache_projection = None
        self._virtual_gate_names = []

    @property
    def virtual_gate_names(self):
        '''
        Returns names of all virtual gates in all virtual gate matrices.
        '''
        return self._virtual_gate_names

    def get_virtual_gate_projection(self, physical_channels):
        '''
        Returns a dictionary with per virtual gate name a dictionary
        with real gate names and multipliers.
        Example:
             'vP1': {'P1': 1.0, 'P2': -0.12},
             'vP2': {'P1': -0.10, 'P2': 1.0},
        '''
        # cache physical channels and matrices. Do not recompute if nothing changed.
        if (self._projection_cache_physical_channels == list(physical_channels)
            and len(self._virtual_matrices) == len(self._projection_cache_matrices)):
                for i, vm in enumerate(self._virtual_matrices.values()):
                    if not np.array_equal(vm._matrix, self._projection_cache_matrices[i]):
                        break
                else:
                    # nothing has changed.
                    return self._projection_cache_projection

        gates = list(physical_channels)
        projection_matrix = np.eye(len(gates))

        for vm in self._virtual_matrices.values():

            real_gates = vm.real_gates
            v2r = vm.v2r_matrix
            # select real gate columns from projection matrix
            col_indices = [gates.index(gate) for gate in real_gates]
            m = projection_matrix[:, col_indices]
            # multiply and concatenate
            p_new = m @ v2r

            projection_matrix = np.concatenate([projection_matrix, p_new], axis=-1)
            # add virtual gates to gate list
            gates += vm.virtual_gates

        # return map
        result = {}
        for i, gate in enumerate(gates):
            gate_values = {}
            result[gate] = gate_values
            for j, real_gate in enumerate(physical_channels):
                value = projection_matrix[j, i]
                if np.abs(value) > 1e-4:
                    gate_values[real_gate] = value

        self._projection_cache_physical_channels = list(physical_channels)
        self._projection_cache_matrices = []
        for vm in self._virtual_matrices.values():
            self._projection_cache_matrices.append(vm._matrix.copy())
        self._projection_cache_projection = result

        return result

    def add(self, name, real_gate_names, virtual_gate_names, matrix,
            real2virtual=False,
            filter_undefined=False,
            keep_squared=False,
            awg_channels=[]):
        '''
        Adds a virtual gate matrix.
        A real gate name must either be AWG channel or an already defined
        virtual gate name of another matrix.

        Args:
            name (str): name of the virtual gate matrix.
            real_gate_names (list[str]): names of real gates
            virtual_gate_names (list[str]): names of virtual gates
            matrix (2D array-like): matrix to convert voltages of virtual gates to voltages of real gates.
            real2virtual (bool): If True v_real = M^-1 @ v_virtual else v_real = M @ v_virtual
            filter_undefined (bool): If True removes rows with unknown real gates.
            keep_squared (bool): matrix is square and should be kept square when valid_indices is used.
            awg_channels (list[str]): names of the AWG channels.
        '''
        if name in self._virtual_matrices:
            del self._virtual_matrices[name]

        np_matrix = np.array(matrix)
        shape = np_matrix.shape
        n_real, n_virtual = (shape[1],shape[0]) if real2virtual else (shape[0],shape[1])

        if n_real != len(real_gate_names):
            raise ValueError(f"size virtual gate matrix ({n_real}) doesn't match "
                             f"the number of real gates names({len(real_gate_names)})")
        if n_virtual != len(virtual_gate_names):
            raise ValueError("size virtual gate matrix ({n_virtual}) doesn't match "
                             f"the number of virtual gates names({len(virtual_gate_names)})")

        if keep_squared and n_real != n_virtual:
            raise Exception(f'Matrix is not square {shape}')

        defined_virtual = self.virtual_gate_names
        defined_channels = defined_virtual + awg_channels

        if filter_undefined:
            # select gates that are defined in pulselib
            valid_indices = []
            not_defined_gates = []
            for i,gate in enumerate(real_gate_names):
                if gate in defined_channels:
                    valid_indices.append(i)
                else:
                    not_defined_gates.append(gate)

            if len(valid_indices) == 0:
                logger.warning(f"No valid gates found of the AWG for the virtual gate matrix {name}."
                                "This virtual gate matrix will be ignored.")
                return

            if len(not_defined_gates):
                logger.warning(f"Gates {not_defined_gates} of virtual gate matrix {name} "
                                "are not defined in pulselib and will be ignored.")

            real_gate_names = [real_gate_names[i] for i in valid_indices]
            if keep_squared:
                virtual_gate_names = [virtual_gate_names[i] for i in valid_indices]
        else:
            valid_indices = None
            for gate in real_gate_names:
                if gate not in defined_channels:
                    raise Exception(f'Gate {gate} of virtual gate matrix {name} is not defined')

        for gate in virtual_gate_names:
            if gate in defined_virtual:
                raise Exception(f'Gate {gate} of virtual gate matrix {name} already defined')

        vgm = VirtualGateMatrix(real_gate_names, virtual_gate_names, matrix,
                                real2virtual=real2virtual,
                                valid_indices=valid_indices,
                                square=keep_squared,
                                )
        self._virtual_matrices[name] = vgm

        # precompute list with virtual gate names
        v_gates = []
        for vm in self._virtual_matrices.values():
            v_gates += vm.virtual_gates
        self._virtual_gate_names = v_gates


