import numpy as np

from steelsight.coil import roll_diagnosis


def test_dense_random_defects_do_not_fake_a_pitch():
    rng = np.random.default_rng(3)
    dense = list(rng.uniform(0, 30, 3000))        # 100 events per metre, no periodicity
    diag = roll_diagnosis(dense)
    assert diag.pitch_m is None


def test_periodic_signal_survives_sparse_noise():
    rng = np.random.default_rng(4)
    pitch = 2.2
    pos = [k * pitch + rng.normal(0, 0.01) for k in range(1, 14)] + list(rng.uniform(0, 30, 10))
    diag = roll_diagnosis(pos)
    assert diag.pitch_m is not None and abs(diag.pitch_m - pitch) < 0.03 and diag.confidence > 0.5
