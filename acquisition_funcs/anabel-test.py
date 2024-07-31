import math

import numpy as np
from hvgpt import Hypervolume


def estimate_hypervolume_via_sampling(
    reference: np.ndarray,  # shape D
    points: np.ndarray,  # shape N x D
    rng: np.random.Generator,  # random number generator
    n_sample: int = 100_000,
) -> float:
    # Check shapes of inputs
    assert reference.ndim == 1
    assert points.ndim == 2
    assert points.shape[1] == reference.shape[0]

    # Check that reference is <= all input points
    assert np.all(np.min(points, axis=0) >= reference), "Reference point is not dominated by all input points"

    # Compute a rectangle which contains the reference and all points.
    max_points = np.max(points, axis=0)  # all points satisfy reference[i] <= points[i] <= max_points[i]
    rectangle_dims = max_points - reference

    # Generate samples and check
    n_samples_in_hypervolume = 0
    for _ in range(n_sample):
        # Draw a random point in the unit hypercube
        random_point = rng.random(size=len(reference))

        # Shift and scale the point to be a random point within the hyperrectangle
        # between the reference point and max_points
        random_point = random_point * rectangle_dims + reference

        # Is this point dominated by any point in points?
        # If so, this sampled point is in our hypervolume region,
        # so we note that down by incrementing n_samples_in_hypervolume
        for p in points:
            if np.all(random_point <= p):
                n_samples_in_hypervolume += 1
                break

    # Our volume estimate is (fraction of points in the hyperrectangle) * (volume of hyperrectangle)
    return (n_samples_in_hypervolume / n_sample) * float(np.prod(rectangle_dims))


if __name__ == "__main__":
    # Run some tests
    rng = np.random.default_rng(seed=1234)

    def run_test_case(reference, points, expected):
        # Estimate via sampling
        output_sampling = estimate_hypervolume_via_sampling(reference=reference, points=points, rng=rng)

        # Calculate using your hypervolume implementation
        hv = Hypervolume(reference)
        output_hypervolume = hv.compute(points)

        print(f"Reference: {reference}, Points: {points}")
        print(f"Expected: {expected}, Sampling Output: {output_sampling}, Hypervolume Output: {output_hypervolume}")

        assert math.isclose(output_sampling, expected, rel_tol=1e-2), f"Sampling: {output_sampling} vs {expected}"
        assert math.isclose(
            output_hypervolume, expected, rel_tol=1e-2
        ), f"Hypervolume: {output_hypervolume} vs {expected}"

    # Test 1: single points (should always output the correct answer)
    run_test_case(
        np.zeros((2,)),
        np.array([[1, 2]]),
        2.0,
    )
    run_test_case(
        np.zeros((2,)),
        np.array([[3, 4]]),
        12.0,
    )
    run_test_case(
        np.array([1, 1]),  # offset reference
        np.array([[3, 4]]),
        6.0,
    )

    # Test #2: some 2D cases
    run_test_case(
        np.zeros((2,)),
        np.array([[1, 2], [2, 1]]),
        3.0,
    )
    run_test_case(
        np.zeros((2,)),
        np.array([[10, 9], [9, 10]]),
        99.0,
    )

    # Test #3: a few 3D cases
    run_test_case(
        np.zeros((3,)),
        np.array([[1, 2, 1], [2, 1, 1], [1, 1, 2]]),
        4.0,
    )

    # Additional Test #4: some other 3D cases
    run_test_case(
        np.zeros((3,)),
        np.array([[3, 4, 5], [5, 3, 4], [4, 5, 3]]),
        60.0,
    )
