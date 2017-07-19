import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter
import contextlib


DEFAULTS = {'graph': None}
__all__ = ['Graph', 'get_default_graph']



class Graph:
    """Computation Graph"""

    def __init__(self):
        self._g = nx.DiGraph()
        self._name_counter = Counter()
        self.feed_dict = None

    def __contains__(self, item):
        return item in self._g

    @property
    def nodes(self):
        """all nodes in the graph"""
        return self._g.nodes()

    @property
    def updatable_nodes(self):
        """all updatable nodes"""
        return [n for n in self.nodes if n.updatable]

    def clear(self):
        """Clear the graph."""
        self._g.clear()
        self._name_counter.clear()

    def reset_nodes(self):
        for node in self.nodes:
            node._reset()

    @contextlib.contextmanager
    def as_default(self):
        """Replace the default graph with the current graph."""
        backup_g = DEFAULTS['graph']
        DEFAULTS['graph'] = self
        try:
            yield self
        except Exception:
            raise
        finally:
            DEFAULTS['graph'] = backup_g

    @contextlib.contextmanager
    def one_pass(self, feed_dict=None):
        """
        context manager
        :param feed_dict: a dict whose keys are Nodes.
        """
        self.feed_dict = feed_dict
        try:
            yield
        except Exception:
            raise
        finally:
            self.feed_dict = None
            self.reset_nodes()

    def topological(self):
        """topological sort"""
        return nx.topological_sort(self._g)

    def get_node_name(self, node):
        """Get the default name for a new node."""
        node_type = node.__class__.__name__
        idx = self._name_counter[node_type]
        self._name_counter[node_type] += 1
        return '{}:{}'.format(node_type, idx)

    def forward(self):
        """forward pass through entire graph"""
        for node in self.topological():
            node.eval()

    def backward(self, start_from=None):
        """Backward Pass (BackPropagation)"""
        topo_order = self.topological()
        if start_from is None:
            start_from = topo_order[-1]
        start_from._dout += 1.0
        for node in reversed(topo_order):
            node.propagate()

    def _check(self, ins, name):
        """
        Check whether the dependent nodes are in the current graph,
        and whether the name has been used.
        :param ins: a list of Node instances.
        :param name: the name of new Node.
        """
        # whether all in-nodes are in the current graph
        for in_node in ins:
            if in_node not in self:
                raise ValueError('Node "{}" should be in the current graph.'.format(repr(in_node)))
        # whether the name is duplicate
        if name in (n.name for n in self._g.nodes()):
            raise ValueError('Node name "{}" exists.'.format(name))

    def add(self, ins, node):
        """
        Add the new node and edges to the graph.
        :param ins: a list of nodes
        :param node: the new node
        """
        self._check(ins, node.name)
        self._g.add_node(node, name=node.name)
        for in_node in ins:
            self._g.add_edge(in_node, node)

    def show(self, figsize=(6, 6)):
        """
        Show the graph.
        :param figsize: figure size.
        """
        plt.figure(figsize=figsize)
        plt.axis('off')
        nx.draw_networkx(self._g, node_size=2000, node_color='c',
                         style='dashdot', font_size=10, pos=nx.spring_layout(self._g))
        plt.show()


DEFAULTS = {'graph': Graph()}


def get_default_graph():
    """
    Get default(current) graph.
    """
    return DEFAULTS['graph']
