import pytest
import torch

from torchdde import TorchLinearInterpolator


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("query_value", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("query_shape", [(), (1,)])
@pytest.mark.parametrize("features", [(2,), (2, 3)])
def test_linear_lookup_values_and_gradients(dtype, query_value, left, query_shape, features):
    times = torch.tensor([0.0, 1.0, 2.0], dtype=dtype, requires_grad=True)
    count = 2 * 3
    for size in features:
        count *= size
    ys = torch.arange(count, dtype=dtype).square().reshape(2, 3, *features).requires_grad_()
    query = torch.full(query_shape, query_value, dtype=dtype, requires_grad=True)
    actual = TorchLinearInterpolator(times, ys)(query, left=left)
    interval = 0 if query_value < 1.0 or (query_value == 1.0 and left) else 1
    prev, next_ = ys[:, interval], ys[:, interval + 1]
    if query_shape:
        prev, next_ = prev.unsqueeze(1), next_.unsqueeze(1)
    fraction = (query - times[interval]) / (times[interval + 1] - times[interval])
    expected = prev + (next_ - prev) * fraction
    torch.testing.assert_close(actual, expected)
    inputs = (times, ys, query)
    gradients = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("query", [-10.0, 1.0, 10.0])
def test_single_point_interpolation_is_constant_and_differentiable(query):
    ys = torch.tensor([[[3.0]]], requires_grad=True)
    actual = TorchLinearInterpolator(torch.tensor([1.0]), ys)(query)
    torch.testing.assert_close(actual, torch.tensor([[3.0]]))
    actual.sum().backward()
    torch.testing.assert_close(ys.grad, torch.ones_like(ys))


def test_linear_interpolation_keeps_existing_outside_domain_behavior():
    interpolator = TorchLinearInterpolator(
        torch.tensor([0.0, 1.0]), torch.tensor([[[2.0], [4.0]]]),
    )
    torch.testing.assert_close(interpolator(-0.001), torch.tensor([[2.0]]))
    torch.testing.assert_close(interpolator(1.001), torch.tensor([[4.0]]))
    with pytest.raises(ValueError, match="outside data range"):
        interpolator(1.1)
