import pytest
from Modules.diffusion.utils import exists, iff, is_sequence, default, to_list, prod, closest_power_2, rand_bool, group_dict_by_prefix, groupby, prefix_dict

# Test exists function
def test_exists():
    assert exists(5) is True
    assert exists(None) is False
    assert exists("Hello") is True

# Test iff function
def test_iff():
    assert iff(True, 10) == 10
    assert iff(False, 10) is None
    assert iff(True, "Hello") == "Hello"

# Test is_sequence function
def test_is_sequence():
    assert is_sequence([1, 2, 3]) is True
    assert is_sequence((1, 2, 3)) is True
    assert is_sequence("Hello") is False

# Test default function
def test_default():
    assert default(None, 5) == 5
    assert default(10, 5) == 10
    assert default(None, lambda: "Hello") == "Hello"

# Test to_list function
def test_to_list():
    assert to_list(5) == [5]
    assert to_list([1, 2, 3]) == [1, 2, 3]
    assert to_list((1, 2, 3)) == [1, 2, 3]

# Test prod function
def test_prod():
    assert prod([1, 2, 3]) == 6
    assert prod([4, 5, 6]) == 120
    assert prod([2, 3, 4, 5]) == 120

# Test closest_power_2 function
def test_closest_power_2():
    assert closest_power_2(10) == 8
    assert closest_power_2(15) == 16
    assert closest_power_2(20) == 16

# Test rand_bool function
def test_rand_bool():
    assert rand_bool((2, 2), 1).tolist() == [[True, True], [True, True]]
    assert rand_bool((2, 2), 0).tolist() == [[False, False], [False, False]]
    assert rand_bool((2, 2), 0.5).shape == (2, 2)

# Test group_dict_by_prefix function
def test_group_dict_by_prefix():
    d = {'prefix_key1': 1, 'prefix_key2': 2, 'other_key1': 3, 'other_key2': 4}
    kwargs_with_prefix, kwargs = group_dict_by_prefix('prefix_', d)
    assert kwargs_with_prefix == {'prefix_key1': 1, 'prefix_key2': 2}
    assert kwargs == {'other_key1': 3, 'other_key2': 4}

# Test groupby function
def test_groupby():
    d = {'prefix_key1': 1, 'prefix_key2': 2, 'other_key1': 3, 'other_key2': 4}
    kwargs_with_prefix, kwargs = groupby('prefix_', d)
    assert kwargs_with_prefix == {'key1': 1, 'key2': 2}
    assert kwargs == {'other_key1': 3, 'other_key2': 4}

# Test prefix_dict function
def test_prefix_dict():
    d = {'key1': 1, 'key2': 2, 'key3': 3}
    prefixed_dict = prefix_dict('prefix_', d)
    assert prefixed_dict == {'prefix_key1': 1, 'prefix_key2': 2, 'prefix_key3': 3}