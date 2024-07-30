from hypervolume import Hypervolume


def test_hypervolume():
    reference_point = [0.0, 0.0]
    hv = Hypervolume(reference_point)

    front = [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]]
    expected_volume = 37.75
    computed_volume = hv.compute(front)
    assert (
        abs(computed_volume - expected_volume) < 1e-4
    ), f"Test failed: computed {computed_volume}, expected {expected_volume}"

    reference_point = [1.0, 0.5]
    hv = Hypervolume(reference_point)
    expected_volume = 28.75
    computed_volume = hv.compute(front)
    assert (
        abs(computed_volume - expected_volume) < 1e-4
    ), f"Test failed: computed {computed_volume}, expected {expected_volume}"

    front = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    reference_point = [-2.1, -2.5, -2.3]
    hv = Hypervolume(reference_point)
    expected_volume = 11.075
    computed_volume = hv.compute(front)
    assert (
        abs(computed_volume - expected_volume) < 1e-4
    ), f"Test failed: computed {computed_volume}, expected {expected_volume}"

    print("All tests passed!")


if __name__ == "__main__":
    test_hypervolume()
