import unittest
import torch
from models.submodules.rotation_covariance import sample_rotation_covariance


class RotationCovarianceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)[None, None]

    def test_zero_covariance_exact_and_reproducible(self):
        cov = torch.zeros_like(self.r)
        out, xi, weights = sample_rotation_covariance(self.r, cov, 4)
        self.assertTrue(torch.equal(out, self.r.expand(4, 1, 1, 3, 3)))
        self.assertEqual(xi.count_nonzero().item(), 0)
        self.assertAlmostEqual(weights.sum().item(), 1.)
        cov = torch.eye(3, dtype=torch.float64)[None, None] * 0.001
        a = sample_rotation_covariance(self.r, cov, 4, generator=torch.Generator().manual_seed(41))
        b = sample_rotation_covariance(self.r, cov, 4, generator=torch.Generator().manual_seed(41))
        self.assertTrue(torch.equal(a[0], b[0]))

    def test_right_multiplication_and_so3(self):
        out, xi, _ = sample_rotation_covariance(self.r, torch.eye(3, dtype=torch.float64)[None, None] * .01, 8)
        # Independent Rodrigues construction checks tangent sign and multiplication side.
        for rotation, vector in zip(out[:, 0, 0], xi[:, 0, 0]):
            theta = vector.norm()
            x, y, z = vector / theta
            k = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64)
            delta = torch.eye(3, dtype=torch.float64) + theta.sin() * k + (1 - theta.cos()) * (k @ k)
            torch.testing.assert_close(rotation, self.r[0, 0] @ delta)
            torch.testing.assert_close(rotation.T @ rotation, torch.eye(3, dtype=torch.float64))
            self.assertAlmostEqual(torch.linalg.det(rotation).item(), 1., places=10)

    def test_full_correlated_covariance(self):
        factor = torch.tensor([[.03, 0, 0], [.012, .02, 0], [-.005, .009, .015]], dtype=torch.float64)
        cov = (factor @ factor.T)[None, None]
        _, xi, _ = sample_rotation_covariance(self.r, cov, 20000, generator=torch.Generator().manual_seed(1234))
        vectors = xi[:, 0, 0]
        self.assertLess(vectors.mean(0).abs().max().item(), 0.0008)
        empirical = vectors.T @ vectors / len(vectors)
        torch.testing.assert_close(empirical, cov[0, 0], atol=2e-5, rtol=.08)

    def test_invalid_covariance_and_reflection_rejected(self):
        for cov in (torch.eye(3) * -1, torch.diag(torch.tensor([1., 0., 1.])), torch.full((3, 3), float('nan'))):
            with self.assertRaises(ValueError):
                sample_rotation_covariance(self.r, cov[None, None], 3)
        cov = torch.eye(3, dtype=torch.float64)[None, None]
        bad = cov.clone()
        bad[..., 0, 1] = .1
        with self.assertRaises(ValueError):
            sample_rotation_covariance(self.r, bad, 3)
        with self.assertRaises(ValueError):
            sample_rotation_covariance(-self.r, cov, 3)

    def test_mixed_zero_and_positive_covariance(self):
        r = self.r.expand(2, 2, 3, 3).clone()
        cov = torch.eye(3, dtype=torch.float64).expand_as(r).clone() * .001
        cov[0, 1] = 0
        out, xi, _ = sample_rotation_covariance(r, cov, 3)
        self.assertTrue(torch.equal(out[:, 0, 1], r[0, 1].expand(3, 3, 3)))
        self.assertEqual(xi[:, 0, 1].count_nonzero().item(), 0)
        self.assertGreater(xi[:, 1, 1].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
