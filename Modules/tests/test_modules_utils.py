import pytest
from Modules.utils import get_padding

def test_get_padding():
    kernel_size = 3
    dilation = 2
    expected_padding = 2
    assert get_padding(kernel_size, dilation) == expected_padding
    # TODO: Add more assertions to test the behavior of get_padding function