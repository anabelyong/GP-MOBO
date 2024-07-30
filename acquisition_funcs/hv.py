import numpy as np
from pareto import pareto_front


class HyperVolume:
    """
    Hypervolume computation based on cross 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!
    """

    def __init__(self, reference_point):
        """Constructor."""
        self.reference_point = reference_point
        self.list = []

    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].
        """
        relevant_points = []
        reference_point = self.reference_point
        dimensions = len(reference_point)

        # Use pareto_front to find non-dominated points
        is_efficient = pareto_front(np.array(front), maximize=False)
        for i in range(len(front)):
            if is_efficient[i]:
                relevant_points.append(front[i])

        print(f"Relevant points before shifting: {relevant_points}")

        if any(reference_point):
            # shift points so that referencePoint == [0, ..., 0]
            for j in range(len(relevant_points)):
                relevant_points[j] = [relevant_points[j][i] - reference_point[i] for i in range(dimensions)]
        print(f"Relevant points after shifting: {relevant_points}")

        self.pre_process(relevant_points)
        bounds = [-1.0e308] * dimensions
        hyper_volume = self.hv_recursive(dimensions - 1, len(relevant_points), bounds)
        return hyper_volume

    def hv_recursive(self, dim_index, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.
        """
        hvol = 0.0
        sentinel = self.list.sentinel
        print(f"Starting hv_recursive with dim_index={dim_index}, length={length}, bounds={bounds}")
        if length == 0:
            print(f"Base case with length=0, returning hvol={hvol}")
            return hvol
        elif dim_index == 0:
            # special case: only one dimension
            hvol = -sentinel.next[0].cargo[0]
            print(f"Base case with dim_index=0, returning hvol={hvol}")
            return hvol
        elif dim_index == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                p_cargo = p.cargo
                hvol += h * (q.cargo[1] - p_cargo[1])
                if p_cargo[0] < h:
                    h = p_cargo[0]
                q = p
                p = q.next[1]
                print(f"Adding hvol={hvol}, h={h}, q.cargo[1]={q.cargo[1]}, p.cargo[1]={p.cargo[1]}")
            hvol += h * q.cargo[1]
            print(f"Base case with dim_index=1, returning hvol={hvol}")
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hv_recursive = self.hv_recursive
            p = sentinel
            q = p.prev[dim_index]
            while q.cargo is not None:
                if q.ignore < dim_index:
                    q.ignore = 0
                q = q.prev[dim_index]
            q = p.prev[dim_index]
            while length > 1 and (
                q.cargo[dim_index] > bounds[dim_index] or q.prev[dim_index].cargo[dim_index] >= bounds[dim_index]
            ):
                p = q
                remove(p, dim_index, bounds)
                q = p.prev[dim_index]
                length -= 1
            q_area = q.area
            q_cargo = q.cargo
            q_prev_dim_index = q.prev[dim_index]
            if length > 1:
                hvol = q_prev_dim_index.volume[dim_index] + q_prev_dim_index.area[dim_index] * (
                    q_cargo[dim_index] - q_prev_dim_index.cargo[dim_index]
                )
            else:
                q_area[0] = 1
                q_area[1 : dim_index + 1] = [q_area[i] * -q_cargo[i] for i in range(dim_index)]
            q.volume[dim_index] = hvol
            if q.ignore >= dim_index:
                q_area[dim_index] = q_prev_dim_index.area[dim_index]
            else:
                q_area[dim_index] = hv_recursive(dim_index - 1, length, bounds)
                if q_area[dim_index] <= q_prev_dim_index.area[dim_index]:
                    q.ignore = dim_index
            while p is not sentinel:
                p_cargo_dim_index = p.cargo[dim_index]
                hvol += q.area[dim_index] * (p_cargo_dim_index - q.cargo[dim_index])
                bounds[dim_index] = p_cargo_dim_index
                reinsert(p, dim_index, bounds)
                length += 1
                q = p
                p = p.next[dim_index]
                q.volume[dim_index] = hvol
                if q.ignore >= dim_index:
                    q.area[dim_index] = q.prev[dim_index].area[dim_index]
                else:
                    q.area[dim_index] = hv_recursive(dim_index - 1, length, bounds)
                    if q.area[dim_index] <= q.prev[dim_index].area[dim_index]:
                        q.ignore = dim_index
                print(f"Updated hvol={hvol}, q.area[dim_index]={q.area[dim_index]}")
            hvol -= q.area[dim_index] * q.cargo[dim_index]
            print(f"Returning hvol={hvol} for dim_index={dim_index}")
            return hvol

    def pre_process(self, front):
        """Sets up the list data structure needed for calculation."""
        dimensions = len(self.reference_point)
        node_list = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sort_by_dimension(nodes, i)
            node_list.extend(nodes, i)
        self.list = node_list

    def sort_by_dimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], index, node) for index, node in enumerate(nodes)]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, _, node) in decorated]


class MultiList:
    """A special data structure needed by FonsecaHyperVolume.

    It consists of several doubly linked lists that share common nodes. So,
    every node has multiple predecessors and successors, one in every list.
    """

    class Node:
        def __init__(self, number_lists, cargo=None):
            self.cargo = cargo
            self.next = [None] * number_lists
            self.prev = [None] * number_lists
            self.ignore = 0
            self.area = [0.0] * number_lists
            self.volume = [0.0] * number_lists

        def __str__(self):
            return str(self.cargo)

    def __init__(self, number_lists):
        """Constructor.

        Builds 'number_lists' doubly linked lists.
        """
        self.number_lists = number_lists
        self.sentinel = MultiList.Node(number_lists)
        self.sentinel.next = [self.sentinel] * number_lists
        self.sentinel.prev = [self.sentinel] * number_lists

    def __str__(self):
        strings = []
        for i in range(self.number_lists):
            current_list = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                current_list.append(str(node))
                node = node.next[i]
            strings.append(str(current_list))
        string_repr = ""
        for string in strings:
            string_repr += string + "\n"
        return string_repr

    def __len__(self):
        """Returns the number of lists that are included in this MultiList."""
        return self.number_lists

    def get_length(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        last_but_one = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last_but_one
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last_but_one.next[index] = node

    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            last_but_one = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = last_but_one
            # set the last element as the new one
            sentinel.prev[index] = node
            last_but_one.next[index] = node

    def remove(self, node, index, bounds):
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


if __name__ == "__main__":
    # Example:
    reference_point = [0.0, 0.0]
    hv = HyperVolume(reference_point)
    front = [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]]
    volume = hv.compute(front)
    print(f"Computed Hypervolume: {volume}")
