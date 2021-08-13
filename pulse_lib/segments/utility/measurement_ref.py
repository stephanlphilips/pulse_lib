import operator
from abc import ABC, abstractmethod
import numpy as np

oper2str = {
        operator.__and__: '&',
        operator.__or__: '|',
        operator.__xor__: '^',
        operator.__invert__: '~',
        }


class MeasurementExpressionBase(ABC):

    def __init__(self, keys):
        self._keys = set(keys)

    @property
    def keys(self):
        return self._keys

    def __and__(self, other):
        return MeasurementBinaryExpression(self, other, operator.__and__)

    def __or__(self, other):
        return MeasurementBinaryExpression(self, other, operator.__or__)

    def __xor__(self, other):
        return MeasurementBinaryExpression(self, other, operator.__xor__)

    def __invert__(self):
        return MeasurementUnaryExpression(self, operator.__invert__)

    @abstractmethod
    def evaluate(self, results):
        raise NotImplementedError()


class MeasurementRef(MeasurementExpressionBase):
    def __init__(self, name, invert=False):
        super().__init__([name])
        self._name = name
        self._inverted = invert

    def inverted(self):
        self._inverted = True

    @property
    def name(self):
        return self._name

    def evaluate(self, results):
        key = self._name
        values = results[key].astype(bool)
        if self._inverted:
            values = ~values
        return values.astype(int)

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

class MeasurementUnaryExpression(MeasurementExpressionBase):
    def __init__(self, a, op):
        super().__init__(a.keys)
        self._a = a
        self._op = op

    def evaluate(self, results):
        values = self._a.evaluate(results).astype(bool)
        values = self._op(values)
        return values.astype(int)

    def __str__(self):
        return f'{oper2str[self._op]}({self._a})'

    def __repr__(self):
        return f'{oper2str[self._op]}({self._a})'

class MeasurementBinaryExpression(MeasurementExpressionBase):
    def __init__(self, lhs, rhs, op):
        super().__init__(lhs.keys | rhs.keys)
        self._lhs = lhs
        self._rhs = rhs
        self._op = op

    def evaluate(self, results):
        values_lhs = self._lhs.evaluate(results).astype(bool)
        values_rhs = self._rhs.evaluate(results).astype(bool)
        values = self._op(values_lhs, values_rhs)
        return values.astype(int)

    def __str__(self):
        return f'({self._lhs}) {oper2str[self._op]} ({self._rhs})'

    def __repr__(self):
        return f'({self._lhs}) {oper2str[self._op]} ({self._rhs})'


if __name__ == '__main__':
    def show(text, expr, nmeasurements):
        print(f'{text}: {expr} ({expr.keys})')
        for i in range(2**nmeasurements):
            results = {f'm{j+1}':np.array([(i>>j)&1]) for j in range(nmeasurements)}
            print(f'{i:0{nmeasurements}b}: {expr.evaluate(results)}')


    m1 = MeasurementRef('m1')
    m2 = MeasurementRef('m2')
    m3 = MeasurementRef('m3')
    m4 = MeasurementRef('m4')

    show('m1 ^ m2', m1 ^ m2, 2)
    show('m1 & m2', m1 & m2, 2)
    show('~(m1 & m2)', ~(m1 & m2), 2)
    show('~(m1 & m2) | m3', ~(m1 & m2) | m3, 3)
    show('(m1 ^ m2) & m2 ^ m3', (m1 ^ m2) & m2 ^ m3, 3)

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
