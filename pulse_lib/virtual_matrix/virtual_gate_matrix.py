import numpy as np

class VirtualGateMatrix:
    '''
    Virtual gate matrix object.

    Args:
        name (str): name of the matrix
        real_gates (list[str]): names of real gates
        virtual_gates (list[str]): names of virtual gates
        matrix (2D array-like): matrix to convert voltages of virtual gates to voltages of real gates.
        real2virtual (bool): If True v_real = M^-1 @ v_virtual else v_real = M @ v_virtual
        valid_indices (list[int]):
        square (bool): matrix is square and should be kept square when valid_indices is used.
    '''
    def __init__(self, real_gates, virtual_gates, matrix, real2virtual=False, valid_indices=None, square=False):
        self._real_gates = real_gates
        self._virtual_gates = virtual_gates
        self._matrix = matrix
        self._invert = real2virtual
        self._valid_indices = valid_indices
        self._square = square

    @property
    def real_gates(self):
        '''
        Names of real gates
        '''
        return self._real_gates

    @property
    def virtual_gates(self):
        '''
        Names of virtual gates
        '''
        return self._virtual_gates

    @property
    def v2r_matrix(self):
        '''
        Returns numpy 2D matrix to convert virtual gates ro real gates.
        v_real = v2r_matrix @ v_virtual
        '''
        # NOTE: _matrix is an 2D-array like object that could be owned by
        #       another entity. It can be modified externally.
        matrix = np.array(self._matrix)
        if self._invert:
            matrix = np.linalg.inv(matrix)
        if self._valid_indices:
            matrix = matrix[self._valid_indices]
            if self._square:
                matrix = matrix[:, self._valid_indices]
        return matrix

