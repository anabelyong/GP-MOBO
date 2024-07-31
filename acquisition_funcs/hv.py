import numpy as np


class Node:
    def __init__(self, m, data=None):
        self.data = data
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = [0.0] * m
        self.volume = [0.0] * m


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
        bounds[:] = np.minimum(bounds, node.data)
        return node

    def reinsert(self, node, index, bounds):
        for i in range(index + 1):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds[:] = np.minimum(bounds, node.data)


class Hypervolume:
    def __init__(self, reference_point):
        self.reference_point = np.array(reference_point)
        self.list = None

    def compute(self, pareto_Y):
        pareto_Y = np.array(pareto_Y)
        if pareto_Y.shape[1] != len(self.reference_point):
            raise ValueError("pareto_Y must have the same number of objectives as reference_point.")
        if pareto_Y.ndim != 2:
            raise ValueError("pareto_Y must be a 2D array.")

        pareto_Y = -pareto_Y  # Convert to minimization
        better_than_ref = np.all(pareto_Y <= self.reference_point, axis=1)
        pareto_Y = pareto_Y[better_than_ref]
        pareto_Y = pareto_Y + self.reference_point
        self._initialize_multilist(pareto_Y)
        bounds = np.full_like(self.reference_point, -np.inf)
        return self._hv_recursive(len(self.reference_point) - 1, len(pareto_Y), bounds)

    def _hv_recursive(self, i, n_pareto, bounds):
        hvol = 0.0
        sentinel = self.list.sentinel
        if n_pareto == 0:
            return hvol
        elif i == 0:
            return -sentinel.next[0].data[0]
        elif i == 1:
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                hvol += h * (q.data[1] - p.data[1])
                if p.data[0] < h:
                    h = p.data[0]
                q = p
                p = q.next[1]
            hvol += h * q.data[1]
            return hvol
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
            else:
                q.area[0] = 1
                q.area[1 : i + 1] = q.area[:i] * -q.data[:i]
            q.volume[i] = hvol
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                if q.area[i] <= q_prev.area[i]:
                    q.ignore = i
            while p is not sentinel:
                p_data = p.data[i]
                hvol += q.area[i] * (p_data - q.data[i])
                bounds[i] = p_data
                self.list.reinsert(p, i, bounds)
                n_pareto += 1
                q = p
                p = p.next[i]
                q.volume[i] = hvol
                if q.ignore >= i:
                    q.area[i] = q.prev[i].area[i]
                else:
                    q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol

    def _initialize_multilist(self, pareto_Y):
        m = pareto_Y.shape[1]
        nodes = [Node(m, point) for point in pareto_Y]
        self.list = MultiList(m)
        for i in range(m):
            nodes.sort(key=lambda node: node.data[i])
            self.list.extend(nodes, i)


# Example usage
if __name__ == "__main__":
    reference_point = [-2.1, -2.5, -2.3]
    hv = Hypervolume(reference_point)
    front = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
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
