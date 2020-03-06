from torch.testing._internal.common_utils import TestCase, run_tests
import torch
from torch import vmap, Tensor
import torch.nn.functional as F


class TestBatching(TestCase):

    def test_batched_batched(self):
        x23 = torch.randn(2, 3)
        output = vmap(torch.add, [0, 0])(x23, x23)
        self.assertEqual(output, x23 + x23)

    def test_batched_unbatched(self):
        x3 = torch.randn(3)
        x23 = torch.randn(2, 3)
        output = vmap(torch.add, [0, None])(x23, x3)
        self.assertEqual(output, x23 + x3)

    def test_aligned_broadcasting(self):
        x23 = torch.randn(2, 3)
        x573 = torch.randn(5, 7, 3)
        output = vmap(torch.mul, [0, None])(x23, x573)
        self.assertEqual(output, x23.view(2, 1, 1, 3) * x573)

    def test_nested(self):
        x23 = torch.randn(2, 3)
        x53 = torch.randn(5, 3)
        output = vmap(lambda xx: vmap(lambda yy: torch.add(xx, yy), [0])(x53), [0])(x23)
        self.assertEqual(output, x23.view(2, 1, 3) + x53)

    def test_nested_three_layers(self):
        x23 = torch.ones(2, 3)
        x53 = torch.ones(5, 3)
        x73 = torch.ones(7, 3)
        output = (vmap(lambda x:
                       vmap(lambda y:
                            vmap(lambda z:
                                 torch.add(torch.add(x, z), y),
                                 [0])(x73),
                            [0])(x53),
                       [0])(x23))
        expected = x23.view(2, 1, 1, 3) + x53.view(5, 1, 3) + x73
        self.assertEqual(output, expected)

    def test_independent_output(self):
        x23 = torch.randn(2, 3)
        output = vmap(lambda x: torch.tensor(1.), [0])(x23)
        self.assertEqual(output, torch.ones(2))

    def test_batched_jacobian(self):
        # TODO: we probably want an API so the user isn't using BatchedTensor directly.
        x3 = torch.randn(3, requires_grad=True)
        y3 = torch.randn(3)
        batched_grad = torch._make_batched(torch.eye(3), 0, 1)
        result = torch.autograd.grad([x3 * y3], [x3], grad_outputs=[batched_grad])
        jacobian = torch._unwrap_batched(result[0], 0)
        self.assertEqual(jacobian, torch.diagflat(y3))

    def test_conv2d(self):
        y25739 = torch.randn(2, 5, 7, 3, 9)
        weight = torch.randn(13, 7, 2, 2, requires_grad=True)
        bias = torch.randn(13, requires_grad=True)

        output = vmap(F.conv2d, (0, None, None))(y25739, weight, bias)
        expected = F.conv2d(y25739.view(10, 7, 3, 9), weight, bias).view(2, 5, 13, 2, 8)
        self.assertEqual(output, expected)

    def test_conv2d_not_front_batch(self):
        y75B39 = torch.randn(7, 5, 2, 3, 9)
        weight = torch.randn(13, 7, 2, 2, requires_grad=True)
        bias = torch.randn(13, requires_grad=True)

        output = vmap(F.conv2d, (2, None, None))(y75B39, weight, bias)
        expected = F.conv2d(y75B39.transpose(0, 2).reshape(10, 7, 3, 9), weight, bias).view(2, 5, 13, 2, 8)
        self.assertEqual(output, expected)



if __name__ == '__main__':
    run_tests()
