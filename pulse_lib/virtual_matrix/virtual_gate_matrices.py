import numpy as np
import logging

from .virtual_gate_matrix import VirtualGateMatrix

class VirtualGateMatrices:
    '''
    Collection of virtual gate matrices.
    '''
    def __init__(self):
        self._virtual_matrices = {}

    @property
    def virtual_gate_names(self):
        '''
        Returns names of all virtual gates in all virtual gate matrices.
        '''
        v_gates = []
        for vm in self._virtual_matrices.values():
            v_gates += vm.virtual_gates
        return v_gates

    @property
    def virtual_gate_projection(self):
        '''
        Returns a dictionary with per virtual gate name a dictionary
        with real gate names and multipliers.
        Example:
             'vP1': {'P1': 1.0, 'P2': -0.12},
             'vP2': {'P1': -0.10, 'P2': 1.0},
        '''
        # first create dictionary with all v_gates and their matrix
        v_gates = {}
        for vm in self._virtual_matrices.values():
            for gate in vm.virtual_gates:
                v_gates[gate] = vm

        vg_map = {}
        for v_gate in v_gates:
            combination = self._get_combination(v_gate, v_gates)
            vg_map[v_gate] = combination.gate_multipliers

        return vg_map

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
                logging.warn(f"No valid gates found of the AWG for the virtual gate matrix {name}."
                             "This virtual gate matrix will be ignored.")
                return

            if len(not_defined_gates):
                logging.warn(f"Gates {not_defined_gates} of virtual gate matrix {name} "
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


    def _get_combination(self, gate, v_gates):
        '''
        Resolves virtual gate to real gate projection.
        '''
        vm = v_gates[gate]
        iv = vm.virtual_gates.index(gate)
        multipliers = vm.v2r_matrix[:,iv]
        combination = GateCombination()
        for ir,r_gate in enumerate(vm.real_gates):
            if r_gate in v_gates:
                combination.add_virtual(
                        self._get_combination(r_gate, v_gates),
                        multipliers[ir]
                        )
            else:
                combination.add_real(r_gate, multipliers[ir])
        return combination



class GateCombination:
    def __init__(self):
        self.gate_multipliers = {}

    def add_real(self, gate, multiplier):
        self.gate_multipliers.setdefault(gate, 0)
        self.gate_multipliers[gate] += multiplier

    def add_virtual(self, gate_combination, multiplier):
        for gate,m in gate_combination.gate_multipliers.items():
            self.gate_multipliers.setdefault(gate, 0)
            self.gate_multipliers[gate] += m*multiplier
