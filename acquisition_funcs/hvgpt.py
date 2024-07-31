import numpy as np
from pareto import pareto_front


class Node:
    def __init__(self, m, cargo=None):
        self.cargo = cargo
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = [0.0] * m
        self.volume = [0.0] * m

    def __str__(self):
        return str(self.cargo)


class MultiList:
    def __init__(self, m):
        self.m = m
        self.sentinel = Node(m)
        self.sentinel.next = [self.sentinel] * m
        self.sentinel.prev = [self.sentinel] * m

    def append(self, node, index):
        last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last
        self.sentinel.prev[index] = node
        last.next[index] = node

    def extend(self, nodes, index):
        for node in nodes:
            self.append(node, index)

    def remove(self, node, index, bounds):
        for i in range(index + 1):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        for i in range(index + 1):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


class Hypervolume:
    def __init__(self, reference_point):
        self.reference_point = reference_point
        self.list = None

    def compute(self, front):
        is_efficient = pareto_front(np.array(front), maximize=True)
        relevant_points = [front[i] for i in range(len(front)) if is_efficient[i]]

        dimensions = len(self.reference_point)
        for j in range(len(relevant_points)):
            relevant_points[j] = [self.reference_point[i] - relevant_points[j][i] for i in range(dimensions)]

        self.pre_process(relevant_points)
        bounds = [-1.0e308] * dimensions
        hypervolume = self.hv_recursive(dimensions - 1, len(relevant_points), bounds)
        return hypervolume

    def hv_recursive(self, dim_index, length, bounds):
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dim_index == 0:
            return -sentinel.next[0].cargo[0]
        elif dim_index == 1:
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
            hvol += h * q.cargo[1]
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
            hvol -= q.area[dim_index] * q.cargo[dim_index]
            return hvol

    def pre_process(self, front):
        dimensions = len(self.reference_point)
        node_list = MultiList(dimensions)
        nodes = [Node(dimensions, point) for point in front]
        for i in range(dimensions):
            nodes = self.sort_by_dimension(nodes, i)
            node_list.extend(nodes, i)
        self.list = node_list

    def sort_by_dimension(self, nodes, i):
        decorated = [(node.cargo[i], index, node) for index, node in enumerate(nodes)]
        decorated.sort()
        nodes[:] = [node for (_, _, node) in decorated]
        return nodes


if __name__ == "__main__":
    reference_point = [0.0, 0.0]
    hv = Hypervolume(reference_point)
    front = [[1, 2], [2, 1]]
    volume = hv.compute(front)
    print("Computed Hypervolume:", volume)

    test_cases = [
        {
            "ref_point": [0.0, 0.0],
            "pareto_Y": np.array([[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]]),
            "expected_volume": 37.75,
        },
        {
            "ref_point": [1.0, 0.5],
            "pareto_Y": np.array([[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]]),
            "expected_volume": 28.75,
        },
        {
            "ref_point": [-2.1, -2.5, -2.3],
            "pareto_Y": np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            "expected_volume": 11.075,
        },
        {
            "ref_point": [-2.1, -2.5, -2.3, -2.0],
            "pareto_Y": np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                ]
            ),
            "expected_volume": 23.15,
        },
        {
            "ref_point": [-1.1, -1.1, -1.1, -1.1, -1.1],
            "pareto_Y": np.array(
                [
                    [-0.4289, -0.1446, -0.1034, -0.4950, -0.7344],
                    [-0.5125, -0.5332, -0.3678, -0.5262, -0.2024],
                    [-0.5960, -0.3249, -0.5815, -0.0838, -0.4404],
                    [-0.6135, -0.5659, -0.3968, -0.3798, -0.0396],
                    [-0.3957, -0.4045, -0.0728, -0.5700, -0.5913],
                    [-0.0639, -0.1720, -0.6621, -0.7241, -0.0602],
                ]
            ),
            "expected_volume": 0.42127855991587,
        },
    ]

    for i, case in enumerate(test_cases):
        ref_point = case["ref_point"]
        pareto_Y = case["pareto_Y"]
        expected_volume = case["expected_volume"]

        hv = Hypervolume(ref_point)
        computed_volume = hv.compute(pareto_Y)

        print(f"Test case {i+1}:")
        print(f"  Reference point: {ref_point}")
        print(f"  Pareto front: {pareto_Y}")
        print(f"  Expected volume: {expected_volume}")
        print(f"  Computed volume: {computed_volume}")

        if np.isclose(computed_volume, expected_volume, atol=1e-5):
            print(f"  Test case {i+1} passed.")
        else:
            print(f"  Test case {i+1} failed.")
