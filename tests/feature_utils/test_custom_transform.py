import numpy as np

from ml_project.features.custom_transform import RoundTransformer


def test_custom_transform() -> None:
    data = np.array([
        [1.1111, 2.3333],
        [2.2222, 2.5555],
        [3.5555, 2.7777]
    ])
    expected_data = np.array([
        [1.111, 2.333],
        [2.222, 2.556],
        [3.556, 2.778]
    ])
    rounder = RoundTransformer(significant=3)
    rounder.fit(data)
    data = rounder.transform(data)

    assert np.allclose(expected_data, data, atol=1e-3)
    assert data[0][0] != 1.1111
