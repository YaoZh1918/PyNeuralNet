import numpy as np


def gradient_check(in_node, out_node, delta=1e-3, feed_dict=None):
    graph = in_node.graph
    graph.reset_nodes()
    # analytical gradient
    with graph.one_pass(feed_dict=feed_dict):
        out_node.eval()
        graph.backward(start_from=out_node)
        true_grad = in_node.gradient
    # numerical gradient
    backup = in_node._value
    in_node._value = backup.copy()
    num_grad = np.empty_like(backup)
    # iter over elements
    for idx, v in np.ndenumerate(backup):
        in_node._value[idx] = v + delta
        with graph.one_pass(feed_dict=feed_dict):
            f_r = out_node.eval()
        in_node._value[idx] = v - delta
        with graph.one_pass(feed_dict=feed_dict):
            f_l = out_node.eval()
        num_grad[idx] = (f_r - f_l) / (2 * delta)
    in_node._value = backup
    re = np.abs(num_grad - true_grad) / np.maximum(np.abs(num_grad) , np.abs(true_grad))
    return re