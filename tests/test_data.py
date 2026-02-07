import pytest

pytest.importorskip("torch")

from auditml.data.datasets import create_member_nonmember_split, get_dataset_info, get_shadow_data_splits


def test_member_nonmember_split_nonoverlap(sample_dataset):
    members, nonmembers = create_member_nonmember_split(sample_dataset, member_ratio=0.5, seed=42)
    assert set(members.indices).isdisjoint(set(nonmembers.indices))
    assert len(members) + len(nonmembers) == len(sample_dataset)


def test_split_reproducibility(sample_dataset):
    a1, b1 = create_member_nonmember_split(sample_dataset, member_ratio=0.5, seed=7)
    a2, b2 = create_member_nonmember_split(sample_dataset, member_ratio=0.5, seed=7)
    assert a1.indices == a2.indices
    assert b1.indices == b2.indices


def test_shadow_splits_count(sample_dataset):
    splits = get_shadow_data_splits(sample_dataset, n_shadows=3, seed=123)
    assert len(splits) == 3


def test_dataset_info_metadata():
    info = get_dataset_info("mnist")
    assert info.num_classes == 10
    assert info.input_shape == (1, 28, 28)
