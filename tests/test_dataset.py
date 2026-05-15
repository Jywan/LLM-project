import pytest
import torch

from llm_project.dataset import get_batch


def test_get_batch_shapes():
    data = torch.arange(10)
    batch_size = 4
    block_size = 3

    x, y = get_batch(data, batch_size=batch_size, block_size=block_size)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)


def test_get_batch_y_is_shifted_by_one():
    data = torch.arange(10)
    batch_size = 8
    block_size = 4

    x, y = get_batch(data, batch_size=batch_size, block_size=block_size)

    assert torch.equal(y, x + 1)


def test_get_batch_requires_1d_data():
    data = torch.arange(12).reshape(3, 4)

    with pytest.raises(ValueError):
        get_batch(data, batch_size=2, block_size=3)


def test_get_batch_requires_data_longer_than_block_size():
    data = torch.arange(4)

    with pytest.raises(ValueError):
        get_batch(data, batch_size=2, block_size=4)