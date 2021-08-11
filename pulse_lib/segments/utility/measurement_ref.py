import operator
from abc import ABC
import numpy as np

oper2str = {
        operator.__and__: '&',
        operator.__or__: '|',
        operator.__xor__: '^',
        operator.__invert__: '~',
        }


class MeasurementExpressionBase(ABC):
    def __init__(self, keys, matrix):
        self._keys = keys
        self._matrix = matrix

    def __and__(self, other):
        return MeasurementBinaryExpression(self, other, operator.__and__)

    def __or__(self, other):
        return MeasurementBinaryExpression(self, other, operator.__or__)

    def __xor__(self, other):
        return MeasurementBinaryExpression(self, other, operator.__xor__)

    def __invert__(self):
        return MeasurementUnaryExpression(self, operator.__invert__)

    @property
    def keys(self):
        return self._keys

    def evaluate(self, results):
        values = tuple(results[key] for key in self.keys)
        return self._matrix[values].astype(int)

    @property
    def matrix(self):
        return self._matrix

    @property
    def ndim(self):
        return self._matrix.ndim


class MeasurementRef(MeasurementExpressionBase):
    def __init__(self, name, invert=False):
        values = np.array([False, True])
        if invert:
            values = ~values
        super().__init__([name], values)
        self._name = name

    def inverted(self):
        self._matrix = np.array([True, False])

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

class MeasurementUnaryExpression(MeasurementExpressionBase):
    def __init__(self, a, op):
        super().__init__(a.keys, op(a.matrix))
        self._a = a
        self._op = op

    def __str__(self):
        return f'{oper2str[self._op]}({self._a})'

    def __repr__(self):
        return f'{oper2str[self._op]}({self._a})'

class MeasurementBinaryExpression(MeasurementExpressionBase):
    def __init__(self, lhs, rhs, op):
        new_axis = tuple(lhs.matrix.ndim + i for i in range(rhs.matrix.ndim))
        super().__init__(
                lhs.keys + rhs.keys,
                op(np.expand_dims(lhs.matrix, axis=new_axis), rhs.matrix)
                )
        self._lhs = lhs
        self._rhs = rhs
        self._op = op

    def __str__(self):
        return f'({self._lhs}) {oper2str[self._op]} ({self._rhs})'

    def __repr__(self):
        return f'({self._lhs}) {oper2str[self._op]} ({self._rhs})'


if __name__ == '__main__':
    def show(text, expr):
        print(f'{text}: {expr}')
        print(expr.matrix.astype(int))
        for i in range(2**expr.ndim):
            results = {f'm{j+1}':(i>>j)&1 for j in range(expr.ndim)}
            print(f'{i:0{expr.ndim}b}: {expr.evaluate(results)} {results}')


    m1 = MeasurementRef('m1')
    m2 = MeasurementRef('m2')
    m3 = MeasurementRef('m3')

    show('m1 ^ m2', m1 ^ m2)
    show('m1 & m2', m1 & m2)
    show('~(m1 & m2)', ~(m1 & m2))

    show('~(m1 & m2) | m3', ~(m1 & m2) | m3)

    mm = m1 & m2
    res = {
        'm1': np.array([0,0,1,1]),
        'm2': np.array([0,1,0,1]),
            }
    print(mm.evaluate(res))

    read12 = MeasurementRef('read12')
    read12_cnot1 = MeasurementRef('read12_cnot1')
    read1 = read12 ^ read12_cnot1 # q1 = 1 if parity changed
    read2 = read12 & read12_cnot1 # q2 = 1 if odd parity in both measurements

    res = {
        'read12': np.array([0,0,1,1]),
        'read12_cnot1': np.array([0,1,0,1]),
            }
    print(read1.evaluate(res))
    print(read2.evaluate(res))

    print(2*read1.evaluate(res) + read2.evaluate(res))

    print()
    res = {
        'm0': np.array([0,0,0,0,1,1,1,1]),
        'm1': np.array([0,0,1,1,0,0,1,1]),
        'm2': np.array([0,1,0,1,0,1,0,1])
        }

    m0 = MeasurementRef('m0')
    m1 = MeasurementRef('m1')
    m2 = MeasurementRef('m2')
    r_0 = m0 & m1 & m2
    r_1 = ~m0 & m1 & m2
    r_2 = m0 & ~m1 & m2
    r_3 = m0 & m1 & ~m2

    print('r0')
    print(r_0.evaluate(res))
    print('r1')
    print(r_1.evaluate(res))
    print('r2')
    print(r_2.evaluate(res))
    print('r3')
    print(r_3.evaluate(res))

    print('qnd')
    qnd = r_0 | r_1 | r_2 | r_3
    print(qnd)
    print(qnd.evaluate(res))


# np.array2string(read1.matrix.astype(int)).replace('\n','')
# str(read1.matrix.astype(int)).replace('\n','')

'''
seq.add_cond(_read3, q3.I, q3.X180)

q3 == 1 -> q3.X180 else q3.I

m1,m2
0,0 -> q3.I
1,0 -> q3.X180
1,0
1,1 -> q3.I

'''
