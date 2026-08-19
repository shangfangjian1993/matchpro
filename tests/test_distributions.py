"""概率分布最小测试:归一化与一致性。"""


def test_pois_pmf_vec_normalized():
    from app.models.distributions import pois_pmf_vec

    p = pois_pmf_vec(1.5)
    assert abs(p.sum() - 1.0) < 1e-9
    assert len(p) == 11


def test_pois_matrix_normalized_and_consistent():
    from app.models.distributions import matrix_to_probs, pois_matrix
    from app.models.ensemble import match_probs

    m = pois_matrix(1.5, 1.2)
    assert abs(m.sum() - 1.0) < 1e-12
    ph, pd_, _pa = match_probs(1.5, 1.2)
    mph, mpd, _mpa = matrix_to_probs(m)
    assert abs(ph - mph) < 1e-9 and abs(pd_ - mpd) < 1e-9
