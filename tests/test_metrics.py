import pandas as pd
import pytest

from meeting_asr.metrics import bootstrap_error_rate, error_counts, error_rate, paired_bootstrap_delta
from meeting_asr.overlap import overlap_ratio


def test_word_counts_known_case():
    # ref: a b c d ; hyp: a x c (sub b->x, del d)
    c = error_counts(["a b c d"], ["a x c"])
    assert c.loc[0, ["sub", "del", "ins", "hit", "n_ref"]].tolist() == [1, 1, 0, 2, 4]
    assert error_rate(c) == pytest.approx(0.5)


def test_corpus_wer_is_not_mean_of_utterance_wers():
    c = error_counts(["a", "a b c d e f g h i j"], ["b", "a b c d e f g h i j"])
    # utterance WERs 1.0 and 0.0 -> mean 0.5, corpus WER 1/11
    assert error_rate(c) == pytest.approx(1 / 11)


def test_empty_reference_and_empty_hypothesis():
    c = error_counts(["", "a b"], ["x y", ""])
    assert c.loc[0, ["ins", "n_ref"]].tolist() == [2, 0]
    assert c.loc[1, ["del", "n_ref"]].tolist() == [2, 2]
    assert error_rate(c) == pytest.approx(2.0)  # (2 ins + 2 del) / 2 ref words


def test_char_counts_ignore_spaces():
    c = error_counts(["ab cd"], ["abcd"], unit="char")
    assert error_rate(c) == 0.0


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible():
    refs = ["a b c d"] * 40
    hyps = ["a b c d", "a x c d", "a b", "a b c d e"] * 10
    clusters = pd.Series([f"m{i % 8}" for i in range(40)])
    c = error_counts(refs, hyps)
    e1 = bootstrap_error_rate(c, clusters, n_boot=500, seed=1)
    e2 = bootstrap_error_rate(c, clusters, n_boot=500, seed=1)
    assert e1 == e2
    assert e1.lo <= e1.value <= e1.hi


def test_paired_delta_sign_and_mismatch_guard():
    refs = ["a b c d"] * 20
    clusters = pd.Series([f"m{i % 5}" for i in range(20)])
    worse = error_counts(refs, ["a x c"] * 20)
    better = error_counts(refs, ["a b c d"] * 20)
    d = paired_bootstrap_delta(worse, better, clusters, n_boot=200)
    assert d["delta"] == pytest.approx(-0.5)
    assert d["p_b_not_better"] == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_delta(worse, error_counts(["a"] * 20, ["a"] * 20), clusters)


def test_overlap_ratio_toy_meeting():
    m = pd.DataFrame({
        "meeting_id": ["m"] * 3,
        "speaker_id": ["A", "B", "A"],
        "begin_time": [0.0, 5.0, 20.0],
        "end_time": [10.0, 15.0, 30.0],
    })
    r = overlap_ratio(m)
    assert r.tolist() == pytest.approx([0.5, 0.5, 0.0])
