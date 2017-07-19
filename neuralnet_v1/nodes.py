import numpy as np
import abc
from neuralnet_v1 import get_default_graph


DEFAULTS = {'dtype': np.float32}
__all__ = ['Input', 'Variable', 'Matmul', 'Add', 'Subtract', 'Multiply', 'Divide', 'Pow', 'ReduceMean']


class Node(abc.ABC):
    """Base Class"""

    def __init__(self, ins=None, name=None, updatable=False):
        """
        init method
        :param ins: a list of Node instances (dependent nodes).
        :param name: the name of the node
        :param updatable: whether the node is updatable
        """
        self._ins = ins if ins else []
        self._g = get_default_graph()
        self._name = name
        self._updatable = updatable
        self._reset()
        # Make sure everything is okay, then add node to the graph.
        self._g.add(self._ins, self)

    def _reset(self):
        """Clear cache and reset upstream gradient."""
        self._cache = None
        self._dout = np.zeros(self.shape, dtype=DEFAULTS['dtype'])

    def __repr__(self):
        s = '{nodetype}(name="{name}", graph={graph!r})'.format(
            nodetype=self.__class__.__name__, name=self.name, graph=self.graph)
        return s

    def __str__(self):
        return self.name

    @property
    def updatable(self):
        return self._updatable

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        if self._name is None:
            self._name = self._g.get_node_name(self)
        return self._name

    @property
    def graph(self):
        return self._g

    @property
    def gradient(self):
        return self._dout

    def eval(self):
        """Evaluate the current node (cached)."""
        if self._cache is None:
            self._cache = self._eval()
        return self._cache

    @abc.abstractmethod
    def _eval(self):
        """Evaluate the current node."""

    def send(self, dout):
        """Receive upstream gradients."""
        self._dout += dout

    @abc.abstractmethod
    def propagate(self):
        """Evaluate and propagate the gradient."""

    def __matmul__(self, b):
        return Matmul(self, b)

    def __add__(self, b):
        return Add(self, b)

    def __sub__(self, b):
        return Subtract(self, b)

    def __rsub__(self, a):
        return Subtract(a, self)

    def __mul__(self, b):
        return Multiply(self, b)

    def __truediv__(self, b):
        return Divide(self, b)

    def __rtruediv__(self, a):
        return Divide(a, self)

    def __pow__(self, power, modulo=None):
        return Pow(self, power)


class Input(Node):
    """Input Node"""

    def __init__(self, shape, name=None):
        """
        init method
        :param shape: a tuple representing the shape of the input
        :param name:  the name of the node
        """
        self._shape = shape
        super().__init__(name=name, updatable=False)

    def _eval(self):
        if self.graph.feed_dict is None:
            raise ValueError('Please provide the value of "{!r}"'.format(self))
        return np.array(self.graph.feed_dict[self], dtype=DEFAULTS['dtype'])

    def propagate(self):
        """Do nothing."""


class Variable(Node):
    """Variable Node"""

    def __init__(self, init_val, name=None, updatable=True):
        """
        init method
        :param init_val: initial value
        :param name: the name of the node
        :param updatable: whether the node represents a constant
        """
        self._value = np.array(init_val, dtype=DEFAULTS['dtype'])
        self._shape = self._value.shape
        super().__init__(name=name, updatable=updatable)

    @property
    def value(self):
        return self._value

    def _eval(self):
        return self._value

    def propagate(self):
        """Do nothing."""


class Op(Node):
    """Operator"""

    def __init__(self, ins, name=None):
        ins = [n if isinstance(n, Node) else Variable(n, updatable=False) for n in ins]
        self._shape = self._infer_shape(*ins)
        super().__init__(ins=ins, name=name, updatable=False)

    @abc.abstractmethod
    def _infer_shape(self, *ins):
        """Infer node shape from inputs."""


class Matmul(Op):
    """Matrix Multiplication"""

    def __init__(self, a, b, name=None):
        """
        init method
        :param a: a Node or np.ndarray with shape=(m, n)
        :param b: a Node or np.ndarray with shape=(n, p)
        :param name: the name of the node
        """
        super().__init__(ins=[a, b], name=name)

    def _infer_shape(self, a, b):
        shape_a = a.shape
        shape_b = b.shape
        if len(shape_a) != 2 or len(shape_b) != 2:
            raise ValueError('Inputs of "Matmul" should be matrices.')
        if shape_a[1] != shape_b[0]:
            raise ValueError('shapes of "{}" and "{}" not aligned: {} x {}'.format(
                a, b, shape_a, shape_b))
        return (shape_a[0], shape_b[1])

    def _eval(self):
        a, b = self._ins
        return a.eval() @ b.eval()

    def propagate(self):
        a, b = self._ins
        a.send(self._dout @ b.eval().T)
        b.send(a.eval().T @ self._dout)


class BroadcastMixin:
    """a mixin class providing '_infer_shape' method"""

    def _infer_shape(self, a, b):
        """Infer shape according to numpy broadcasting rule."""
        shape_a = a.shape
        shape_b = b.shape
        rank_a = len(shape_a)
        rank_b = len(shape_b)
        self._sum_over_a = []  # axes along which a sum is performed
        self._sum_over_b = []
        if rank_a > rank_b:
            ret_shape = list(shape_a)
            self._squeeze_over_a = ()  # axes that will be removed
            self._squeeze_over_b = tuple(range(rank_a - rank_b))
        else:
            ret_shape = list(shape_b)
            self._squeeze_over_a = tuple(range(rank_b - rank_a))
            self._squeeze_over_b = ()
        for d_a, d_b, k in zip(reversed(shape_a),
                               reversed(shape_b),
                               range(max(rank_a, rank_b)-1, -1, -1)):
            if d_a == d_b:  # != 1
                continue
            elif d_a == 1:
                ret_shape[k] = d_b
                self._sum_over_a.append(k)
            elif d_b == 1:
                ret_shape[k] = d_a
                self._sum_over_b.append(k)
            else:
                raise ValueError('operands could not be '
                                 'broadcast together with '
                                 'shapes {} {}'.format(shape_a, shape_b))
        self._sum_over_a = self._squeeze_over_a + tuple(reversed(self._sum_over_a))
        self._sum_over_b = self._squeeze_over_b + tuple(reversed(self._sum_over_b))
        return tuple(ret_shape)

    def _sum_gradient_and_send(self, da, db):
        """Sum the gradients and then send them."""
        a, b = self._ins
        if self._sum_over_a:
            summed_da = da.sum(axis=self._sum_over_a, keepdims=True)
            summed_da = np.squeeze(summed_da, axis=self._squeeze_over_a)
        else:
            summed_da = da
        if self._sum_over_b:
            summed_db = db.sum(axis=self._sum_over_b, keepdims=True)
            summed_db = np.squeeze(summed_db, axis=self._squeeze_over_b)
        else:
            summed_db = db
        a.send(summed_da)
        b.send(summed_db)


class Add(BroadcastMixin, Op):

    def __init__(self, a, b, name=None):
        super().__init__(ins=[a, b], name=name)

    def _eval(self):
        a, b = self._ins
        return a.eval() + b.eval()

    def propagate(self):
        a, b = self._ins
        self._sum_gradient_and_send(self._dout, self._dout)


class Subtract(BroadcastMixin, Op):

    def __init__(self, a, b, name=None):
        super().__init__(ins=[a, b], name=name)

    def _eval(self):
        a, b = self._ins
        return a.eval() - b.eval()

    def propagate(self):
        a, b = self._ins
        self._sum_gradient_and_send(self._dout, -self._dout)


class Multiply(BroadcastMixin, Op):

    def __init__(self, a, b, name=None):
        super().__init__(ins=[a, b], name=name)

    def _eval(self):
        a, b = self._ins
        return a.eval() * b.eval()

    def propagate(self):
        a, b = self._ins
        da = self._dout * b.eval()
        db = self._dout * a.eval()
        self._sum_gradient_and_send(da, db)


class Divide(BroadcastMixin, Op):

    def __init__(self, a, b, name=None):
        super().__init__(ins=[a, b], name=name)

    def _eval(self):
        a, b = self._ins
        return a.eval() / b.eval()

    def propagate(self):
        a, b = self._ins
        da = self._dout / b.eval()
        db = -self._dout * a.eval() / b.eval()**2
        self._sum_gradient_and_send(da, db)


class Pow(Op):

    def __init__(self, a, p, name=None):
        """
        Calculate a ** p.
        :param a: a Node instance
        :param p: a real number
        :param name: the name of Node
        """
        super().__init__(ins=[a], name=name)
        self._p = p

    def _infer_shape(self, a):
        return a.shape

    def _eval(self):
        a = self._ins[0]
        return a.eval() ** self._p

    def propagate(self):
        a = self._ins[0]
        a.send(self._p * self._dout * a.eval() ** (self._p - 1))


class ReduceMean(Op):

    def __init__(self, a, name=None):
        super().__init__(ins=[a], name=name)

    def _infer_shape(self, a):
        return ()

    def _eval(self):
        a = self._ins[0]
        return a.eval().mean()

    def propagate(self):
        a = self._ins[0]
        n = np.product(a.shape)
        a.send(np.full(a.shape, fill_value=self._dout / n, dtype=DEFAULTS['dtype']))
