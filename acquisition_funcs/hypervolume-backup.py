import numpy as np

MIN_Y_RANGE = 1e-7


class Node:
    def __init__(self, m, data):
        self.data = np.array(data)
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = np.zeros(m)
        self.volume = np.zeros_like(self.area)


class MultiList:
    def __init__(self, m):
        self.m = m
        self.sentinel = Node(m, None)
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
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
        bounds[:] = np.minimum(bounds, node.data)
        return node

    def reinsert(self, node, index, bounds):
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds[:] = np.minimum(bounds, node.data)


class Hypervolume:
    def __init__(self, reference_point):
        self.reference_point = np.array(reference_point)
        self.list = []
        print(f"Initialized Hypervolume with reference point: {self.reference_point}")

    @property
    def ref_point(self):
        return -self.reference_point

    @ref_point.setter
    def ref_point(self, ref_point):
        self.reference_point = -np.array(ref_point)
        print(f"Set reference point: {self.reference_point}")

    def compute(self, pareto_Y):
        ref_point = self.reference_point
        pareto_Y = np.array(pareto_Y)

        if pareto_Y.shape[-1] != ref_point.shape[0]:
            raise ValueError("pareto_Y must have the same number of objectives as ref_point.")

        if pareto_Y.ndim != 2:
            raise ValueError("pareto_Y must have exactly two dimensions.")

        pareto_Y = -pareto_Y
        print(f"Negated Pareto front: {pareto_Y}")
        better_than_ref = (pareto_Y <= ref_point).all(axis=-1)
        pareto_Y = pareto_Y[better_than_ref]

        print(f"Filtered Pareto front (better_than_ref): {pareto_Y}")
        pareto_Y = pareto_Y - ref_point
        print(f"Shifted Pareto front: {pareto_Y}")

        self.pre_process(pareto_Y)
        bounds = np.full_like(ref_point, float("-inf"))
        hv = self.hv_recursive(ref_point.shape[0] - 1, pareto_Y.shape[0], bounds)
        print(f"Final hypervolume: {hv}")
        return hv

    def hv_recursive(self, i, n_pareto, bounds):
        hvol = 0.0
        sentinel = self.list.sentinel
        print(f"Recursive step: dimension {i}, n_pareto {n_pareto}, bounds {bounds}")
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
                q.area[i] = self.hv_recursive(i - 1, n_pareto, bounds)
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
                    q.area[i] = self.hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol

    def pre_process(self, front):
        dimensions = len(self.reference_point)
        node_list = MultiList(dimensions)
        nodes = [Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sort_by_dimension(nodes, i)
            node_list.extend(nodes, i)
        self.list = node_list

    def sort_by_dimension(self, nodes, i):
        decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
        decorated.sort()
        nodes[:] = [node for (_, _, node) in decorated]


def compute_hypervolume(front, reference_point):
    hv = Hypervolume(reference_point)
    return hv.compute(front)


def infer_reference_point(pareto_Y, max_ref_point=None, scale=0.1, scale_max_ref_point=False):
    if pareto_Y.shape[0] == 0:
        if max_ref_point is None:
            raise ValueError("Empty pareto set and no max ref point provided")
        if np.isnan(max_ref_point).any():
            raise ValueError("Empty pareto set and max ref point includes NaN.")
        if scale_max_ref_point:
            return max_ref_point - scale * np.abs(max_ref_point)
        return max_ref_point

    if max_ref_point is not None:
        non_nan_idx = ~np.isnan(max_ref_point)
        better_than_ref = np.all(pareto_Y[:, non_nan_idx] > max_ref_point[non_nan_idx], axis=-1)
    else:
        non_nan_idx = np.ones(pareto_Y.shape[-1], dtype=bool)
        better_than_ref = np.ones(pareto_Y.shape[:1], dtype=bool)

    if max_ref_point is not None and np.any(better_than_ref) and np.all(non_nan_idx):
        Y_range = pareto_Y[better_than_ref].max(axis=0) - max_ref_point
        if scale_max_ref_point:
            return max_ref_point - scale * Y_range
        return max_ref_point
    elif pareto_Y.shape[0] == 1:
        Y_range = np.abs(pareto_Y).clip(min=MIN_Y_RANGE).reshape(-1)
        ref_point = pareto_Y.reshape(-1) - scale * Y_range
    else:
        nadir = pareto_Y.min(axis=0)
        if max_ref_point is not None:
            nadir[non_nan_idx] = np.minimum(nadir[non_nan_idx], max_ref_point[non_nan_idx])
        ideal = pareto_Y.max(axis=0)
        Y_range = np.clip(ideal - nadir, MIN_Y_RANGE, None)
        ref_point = nadir - scale * Y_range

    if np.any(non_nan_idx) and not np.all(non_nan_idx) and np.any(better_than_ref):
        if scale_max_ref_point:
            ref_point[non_nan_idx] = (max_ref_point - scale * Y_range)[non_nan_idx]
        else:
            ref_point[non_nan_idx] = max_ref_point[non_nan_idx]

    print(f"Inferred reference point: {ref_point}")
    return ref_point
