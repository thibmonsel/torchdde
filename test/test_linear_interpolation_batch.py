import pytest
import torch

from torchdde import TorchLinearInterpolator


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("features", [(2,), (2, 3)])
@pytest.mark.parametrize("query_values", [
    [1.5, 0.0, 2.0, 1.0, 1.5],
    [-0.001, 0.25, 2.001],
    [-0.001, 2.001],
])
def test_batch_matches_scalar_values_and_gradients(dtype, left, features, query_values):
    times = torch.tensor([0.0, 1.0, 2.0], dtype=dtype, requires_grad=True)
    size = 6
    for feature in features:
        size *= feature
    ys = torch.arange(size, dtype=dtype).square().reshape(2, 3, *features).requires_grad_()
    queries = torch.tensor(query_values, dtype=dtype, requires_grad=True)
    interpolator = TorchLinearInterpolator(times, ys)
    actual = interpolator.evaluate_many(queries, left=left)
    expected = torch.stack([interpolator(t, left=left) for t in queries.unbind()])
    torch.testing.assert_close(actual, expected)
    inputs = (times, ys, queries)
    gradients = torch.autograd.grad(
        actual.square().sum(), inputs, retain_graph=True, allow_unused=True,
    )
    reference = torch.autograd.grad(expected.square().sum(), inputs, allow_unused=True)
    for gradient, expected_gradient in zip(gradients, reference):
        if expected_gradient is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, expected_gradient)


def test_batch_keeps_first_endpoint_precedence_for_short_intervals():
    interpolator = TorchLinearInterpolator(
        torch.tensor([0.0, 0.001]), torch.tensor([[[2.0], [9.0]]]),
    )
    queries = torch.tensor([0.002, -0.001, 0.0005])
    expected = torch.stack([interpolator(t) for t in queries.unbind()])
    torch.testing.assert_close(interpolator.evaluate_many(queries), expected)


def test_single_point_batch_is_constant_with_matching_gradients():
    times = torch.tensor([1.0], requires_grad=True)
    ys = torch.tensor([[[3.0], [4.0]]]).transpose(0, 1).requires_grad_()
    queries = torch.tensor([-10.0, 1.0, 10.0], requires_grad=True)
    interpolator = TorchLinearInterpolator(times, ys)
    actual = interpolator.evaluate_many(queries)
    expected = torch.stack([interpolator(t) for t in queries.unbind()])
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual.sum(), (times, ys, queries), allow_unused=True)
    expected_gradients = torch.autograd.grad(expected.sum(), (times, ys, queries), allow_unused=True)
    assert actual_gradients[0] is expected_gradients[0] is None
    assert actual_gradients[2] is expected_gradients[2] is None
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1])


@pytest.mark.parametrize("queries", [torch.empty(0), torch.ones(2, 1)])
def test_batch_rejects_invalid_shape(queries):
    interpolator = TorchLinearInterpolator(torch.tensor([0.0, 1.0]), torch.ones(1, 2, 1))
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        interpolator.evaluate_many(queries)


def test_batch_rejects_outside_query_even_when_other_queries_are_valid():
    interpolator = TorchLinearInterpolator(torch.tensor([0.0, 1.0]), torch.ones(1, 2, 1))
    with pytest.raises(ValueError, match="outside data range"):
        interpolator.evaluate_many(torch.tensor([0.5, 1.1]))


def test_batch_respects_custom_scalar_override():
    class CustomInterpolator(TorchLinearInterpolator):
        def __call__(self, t, left=True):
            return 2 * super().__call__(t, left=left)

    interpolator = CustomInterpolator(
        torch.tensor([0.0, 1.0]), torch.tensor([[[1.0], [2.0]]]),
    )
    torch.testing.assert_close(
        interpolator.evaluate_many(torch.tensor([0.25, 0.75])),
        torch.tensor([[[2.5]], [[3.5]]]),
    )
