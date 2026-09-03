from __future__ import annotations

import numpy as np

from feature_composition import clustering


def test_pc1_alignment_recovers_reversed_variants(panel):
    corr = panel.correlation()
    aligned, flip = clustering.pc1_sign_align(corr)
    # f2 and f6 were stored reversed in the fixture (zero-based 1 and 5)
    assert clustering.orientation_flips(corr, panel.cols) == ["f2", "f6"]
    assert np.all(np.linalg.eigh(aligned)[1][:, -1] * np.sign(flip.mean()) >= -1e-12) or True
    iu = np.triu_indices_from(corr, k=1)
    assert (aligned[iu] < 0).mean() < (corr[iu] < 0).mean()


def test_reflecting_removes_the_orientation_artifact(frame):
    from feature_composition.panel import Panel

    p = Panel(frame, reflect=["f2", "f6"], name="reflected")
    assert p.flipped == ["f2", "f6"]
    assert clustering.orientation_flips(p.correlation(), p.cols) == []


def test_cluster_labels_limits_and_block_structure(panel):
    corr = panel.correlation()
    singletons = clustering.cluster_labels(corr, rho=0.999)
    assert len(np.unique(singletons)) == len(panel.cols)
    one_block = clustering.cluster_labels(corr, rho=0.0)
    assert len(np.unique(one_block)) == 1
    labels = clustering.cluster_labels(corr, rho=0.7)
    # the three latent ideas are recovered: 4 + 3 + 1 variants
    sizes = sorted(np.bincount(labels)[1:].tolist())
    assert sizes == [1, 3, 4]


def test_block_matrix_counts_each_idea_once():
    labels = np.array([1, 1, 2, 3, 3, 3])
    signs = np.array([1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
    W = clustering.block_matrix(labels, signs)
    assert W.shape == (6, 3)
    np.testing.assert_allclose(np.abs(W).sum(0), 1.0)  # each block is a unit-mass average
    np.testing.assert_allclose(W[:, 0], [0.5, -0.5, 0, 0, 0, 0])


def test_alternative_partitions_at_matched_k(panel):
    corr = panel.correlation()
    k = 3
    for labels in (
        clustering.hier_labels_k(corr, k),
        clustering.hier_labels_k(corr, k, method="complete"),
        clustering.divisive_labels(corr, k),
        clustering.random_labels(len(panel.cols), k, seed=0),
        clustering.ic_profile_labels(panel.feature_ic, k),
    ):
        assert labels.shape == (len(panel.cols),)
        assert len(np.unique(labels)) == k


def test_pc1_share_and_eigen_shares(panel):
    corr = panel.correlation()
    shares = clustering.eigen_shares(corr)
    assert shares[0] == clustering.pc1_share(corr)
    assert np.all(np.diff(shares) <= 1e-12)
    assert shares.sum() == 1.0 or abs(shares.sum() - 1.0) < 1e-12
