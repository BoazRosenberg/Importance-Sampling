import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from importance_sampling import Sampler, export_subject_summary, sigmoid, softplus


class TestExportSubjectSummary(unittest.TestCase):
    def setUp(self):
        """Set up test environment and mock data."""
        self.tmp_dir = tempfile.mkdtemp()
        self.n_subjects = 3
        self.data = [
            {"trial": np.array([0, 1, 2]), "choice": np.array([1, 0, 1])},
            {"trial": np.array([0, 1, 2]), "choice": np.array([0, 1, 0])},
            {"trial": np.array([0, 1, 2]), "choice": np.array([1, 1, 0])},
        ]
        self.hyper_params = {
            "param1": {"mean": 0.0, "sd": 1.0, "transform": sigmoid},
            "param2": {"mean": 1.0, "sd": 0.5, "transform": softplus},
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_fitted_sampler(self, model_name="test_model"):
        """Helper to create a sampler with known raw samples and log_likelihoods."""
        def dummy_model(data, params, mode="log_likelihood"):
            return np.array([-10.0, -12.0, -8.0])

        sampler = Sampler(
            self.data,
            dummy_model,
            self.hyper_params,
            model_name=model_name,
            random_state=42,
        )

        # Populate samples directly for exact numeric verification
        np.random.seed(42)
        n_particles = 1000
        sampler.samples = {
            "param1": [
                np.random.normal(loc=-0.5, scale=0.8, size=n_particles),
                np.random.normal(loc=0.2, scale=0.6, size=n_particles),
                np.random.normal(loc=1.0, scale=0.5, size=n_particles),
            ],
            "param2": [
                np.random.normal(loc=0.5, scale=0.4, size=n_particles),
                np.random.normal(loc=1.2, scale=0.3, size=n_particles),
                np.random.normal(loc=-0.1, scale=0.5, size=n_particles),
            ],
            "log_likelihood": [
                np.random.normal(loc=-15.0, scale=2.0, size=n_particles),
                np.random.normal(loc=-20.0, scale=1.5, size=n_particles),
                np.random.normal(loc=-12.0, scale=2.5, size=n_particles),
            ],
        }
        return sampler

    def test_export_subject_summary_columns_and_format(self):
        """Verify the summary CSV has columns: subject, parameter, mean, ci_high, ci_low."""
        sampler = self._create_fitted_sampler(model_name="m1")
        csv_path = os.path.join(self.tmp_dir, "summary.csv")

        df = sampler.export_subject_summary(file_name="summary.csv", save_to_folder=self.tmp_dir)

        # 1. Verify returned DataFrame and file on disk
        self.assertTrue(os.path.exists(csv_path))
        file_df = pd.read_csv(csv_path)

        for checked_df in [df, file_df]:
            self.assertEqual(
                list(checked_df.columns),
                ["subject", "parameter", "mean", "ci_high", "ci_low"],
            )
            # 3 subjects * (1 loglikelihood + 2 parameters) = 9 rows
            self.assertEqual(len(checked_df), 3 * (1 + 2))

            # Verify subjects and parameters present
            for s in range(self.n_subjects):
                subj_rows = checked_df[checked_df["subject"] == s]
                params_in_subj = list(subj_rows["parameter"])
                self.assertIn("loglikelihood", params_in_subj)
                self.assertIn("param1", params_in_subj)
                self.assertIn("param2", params_in_subj)

    def test_loglikelihood_mean_and_ci(self):
        """Verify log-likelihood mean is log(mean(likelihood)) = logsumexp(LL) - log(N), not mean(LL)."""
        sampler = self._create_fitted_sampler()
        df = sampler.export_subject_summary(save_to_folder=self.tmp_dir)

        for s in range(self.n_subjects):
            ll_particles = sampler.samples["log_likelihood"][s]
            expected_mean = float(logsumexp(ll_particles) - np.log(len(ll_particles)))
            expected_ci_low = float(np.percentile(ll_particles, 2.5))
            expected_ci_high = float(np.percentile(ll_particles, 97.5))

            row = df[(df["subject"] == s) & (df["parameter"] == "loglikelihood")].iloc[0]

            self.assertAlmostEqual(row["mean"], expected_mean, places=6)
            self.assertAlmostEqual(row["ci_low"], expected_ci_low, places=6)
            self.assertAlmostEqual(row["ci_high"], expected_ci_high, places=6)

            # Ensure mean(log(likelihood)) is strictly smaller due to Jensen's inequality
            arithmetic_mean_ll = float(np.mean(ll_particles))
            self.assertNotAlmostEqual(row["mean"], arithmetic_mean_ll, places=2)
            self.assertGreater(row["mean"], arithmetic_mean_ll)

    def test_parameter_mean_and_ci_before_transformation(self):
        """Verify parameter mean and CI limits are calculated before transformation and then transformed."""
        sampler = self._create_fitted_sampler()
        df = sampler.export_subject_summary(save_to_folder=self.tmp_dir)

        for s in range(self.n_subjects):
            # Test param1 (transformed with sigmoid)
            raw_p1 = sampler.samples["param1"][s]
            raw_mean_p1 = float(np.mean(raw_p1))
            raw_low_p1 = float(np.percentile(raw_p1, 2.5))
            raw_high_p1 = float(np.percentile(raw_p1, 97.5))

            expected_p1_mean = float(sigmoid(raw_mean_p1))
            expected_p1_low = float(sigmoid(raw_low_p1))
            expected_p1_high = float(sigmoid(raw_high_p1))

            row_p1 = df[(df["subject"] == s) & (df["parameter"] == "param1")].iloc[0]
            self.assertAlmostEqual(row_p1["mean"], expected_p1_mean, places=6)
            self.assertAlmostEqual(row_p1["ci_low"], expected_p1_low, places=6)
            self.assertAlmostEqual(row_p1["ci_high"], expected_p1_high, places=6)

            # Test param2 (transformed with softplus)
            raw_p2 = sampler.samples["param2"][s]
            raw_mean_p2 = float(np.mean(raw_p2))
            raw_low_p2 = float(np.percentile(raw_p2, 2.5))
            raw_high_p2 = float(np.percentile(raw_p2, 97.5))

            expected_p2_mean = float(softplus(raw_mean_p2))
            expected_p2_low = float(softplus(raw_low_p2))
            expected_p2_high = float(softplus(raw_high_p2))

            row_p2 = df[(df["subject"] == s) & (df["parameter"] == "param2")].iloc[0]
            self.assertAlmostEqual(row_p2["mean"], expected_p2_mean, places=6)
            self.assertAlmostEqual(row_p2["ci_low"], expected_p2_low, places=6)
            self.assertAlmostEqual(row_p2["ci_high"], expected_p2_high, places=6)

    def test_default_filename_and_module_function(self):
        """Verify default filename 'subject_summary_{model_name}.csv' and top-level module function."""
        sampler = self._create_fitted_sampler(model_name="my_model")

        # Top-level export_subject_summary(sampler, ...)
        df = export_subject_summary(sampler, save_to_folder=self.tmp_dir)
        expected_path = os.path.join(self.tmp_dir, "subject_summary_my_model.csv")

        self.assertTrue(os.path.exists(expected_path))
        self.assertEqual(len(df), 3 * 3)

    def test_unfitted_model_raises_error(self):
        """Verify calling export_subject_summary before fitting raises ValueError."""
        def dummy_model(data, params, mode="log_likelihood"):
            return np.array([0.0])

        unfitted = Sampler(self.data, dummy_model, self.hyper_params)
        with self.assertRaises(ValueError) as ctx:
            unfitted.export_subject_summary()
        self.assertIn("fitted", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
