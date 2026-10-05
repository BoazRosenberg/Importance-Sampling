import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from importance_sampling import Sampler


class TestFileNamingConventions(unittest.TestCase):
    def setUp(self):
        """Set up test data and temporary directory for file exports."""
        self.tmp_dir = tempfile.mkdtemp()
        self.n_subjects = 3
        # Single-phase dataset
        self.data_single = [
            {"trial": np.array([0, 1, 2]), "choice": np.array([1, 0, 1])},
            {"trial": np.array([0, 1, 2]), "choice": np.array([0, 1, 0])},
            {"trial": np.array([0, 1, 2]), "choice": np.array([1, 1, 0])},
        ]
        # Multi-phase dataset (several files / sub-indices)
        self.data_multi = [
            {
                "phase1": pd.DataFrame({"trial": [0, 1], "choice": [1, 0]}),
                "phase2": pd.DataFrame({"trial": [2, 3], "choice": [0, 1]}),
            },
            {
                "phase1": pd.DataFrame({"trial": [0, 1], "choice": [0, 1]}),
                "phase2": pd.DataFrame({"trial": [2, 3], "choice": [1, 0]}),
            },
            {
                "phase1": pd.DataFrame({"trial": [0, 1], "choice": [1, 1]}),
                "phase2": pd.DataFrame({"trial": [2, 3], "choice": [0, 0]}),
            },
        ]
        self.hyper_params = {
            "param1": {"mean": 0.5, "sd": 0.2},
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        # Also clean up default folders if created in test execution
        for folder in ["simulate", "deep_simulate"]:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)

    def test_simulate_default_single_file_name(self):
        """Verify default single file for simulate is simulate_{model_name}.csv."""
        def dummy_model(data, params, mode="simulate"):
            return pd.DataFrame({"trial": [0, 1], "p": [0.6, 0.7]})

        sampler = Sampler(
            self.data_single,
            dummy_model,
            self.hyper_params,
            model_name="m0",
            random_state=42,
        )

        saved = sampler.simulate(save_to_folder=self.tmp_dir)
        expected_file = os.path.join(self.tmp_dir, "simulate_m0.csv")
        self.assertTrue(
            os.path.exists(expected_file),
            f"Expected {expected_file} to exist, but got {saved}",
        )

    def test_deep_simulate_default_single_file_name(self):
        """Verify default single file for deep_simulate is deep_simulate_{model_name}.csv."""
        def dummy_model(data, params, mode="simulate", f=None):
            return pd.DataFrame({"trial": [0, 1], "p": [0.5, 0.5]})

        sampler = Sampler(
            self.data_single,
            dummy_model,
            self.hyper_params,
            model_name="m0",
            random_state=42,
        )

        fn = lambda d, p: None
        saved = sampler.deep_simulate(functions=fn, save_to_folder=self.tmp_dir)
        expected_file = os.path.join(self.tmp_dir, "deep_simulate_m0.csv")
        self.assertTrue(
            os.path.exists(expected_file),
            f"Expected {expected_file} to exist, but got {saved}",
        )

    def test_custom_file_name_single_file(self):
        """Verify user custom name file_name='my_custom_name' is used directly for single file."""
        def dummy_model(data, params, mode="simulate"):
            return pd.DataFrame({"trial": [0, 1], "p": [0.9, 0.1]})

        sampler = Sampler(
            self.data_single,
            dummy_model,
            self.hyper_params,
            model_name="m0",
            random_state=42,
        )

        sampler.simulate(file_name="my_custom_name", save_to_folder=self.tmp_dir)
        expected_file = os.path.join(self.tmp_dir, "my_custom_name.csv")
        self.assertTrue(os.path.exists(expected_file))

        fn = lambda d, p: None
        sampler.deep_simulate(
            functions=fn,
            file_name="deep_custom_sim",
            save_to_folder=self.tmp_dir,
        )
        expected_deep_file = os.path.join(self.tmp_dir, "deep_custom_sim.csv")
        self.assertTrue(os.path.exists(expected_deep_file))

    def test_several_files_put_in_simulate_or_deep_simulate_folder(self):
        """Verify several files use sub-index names and are placed in 'simulate' or 'deep_simulate' folder."""
        def multi_phase_model(data, params, mode="simulate", f=None):
            return {
                "phase1": pd.DataFrame({"trial": [0, 1], "choice": [1, 0]}),
                "phase2": pd.DataFrame({"trial": [2, 3], "choice": [0, 1]}),
            }

        sampler = Sampler(
            self.data_multi,
            multi_phase_model,
            self.hyper_params,
            model_name="m0",
            random_state=42,
        )

        # 1. simulate() with multiple sub-indices defaults to folder "simulate"
        saved_sim = sampler.simulate(save=True)
        self.assertTrue(os.path.exists(os.path.join("simulate", "phase1.csv")))
        self.assertTrue(os.path.exists(os.path.join("simulate", "phase2.csv")))

        # 2. deep_simulate() with multiple sub-indices defaults to folder "deep_simulate"
        fn = lambda d, p: None
        saved_deep = sampler.deep_simulate(functions=fn, save=True)
        self.assertTrue(os.path.exists(os.path.join("deep_simulate", "phase1.csv")))
        self.assertTrue(os.path.exists(os.path.join("deep_simulate", "phase2.csv")))

    def test_custom_file_name_multiple_files_appends_index(self):
        """Verify multiple files with custom file_name appends the subject ID or sub-index."""
        def multi_phase_model(data, params, mode="simulate"):
            return {
                "phase1": pd.DataFrame({"trial": [0, 1], "choice": [1, 0]}),
                "phase2": pd.DataFrame({"trial": [2, 3], "choice": [0, 1]}),
            }

        sampler = Sampler(
            self.data_multi,
            multi_phase_model,
            self.hyper_params,
            model_name="m0",
            random_state=42,
        )

        # Multiple sub-indices with custom file_name
        sampler.simulate(file_name="my_study", save_to_folder=self.tmp_dir)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "my_study_phase1.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "my_study_phase2.csv")))

        # Multiple files by subject with custom file_name
        sampler.simulate(
            file_name="subject_export",
            by_subject=True,
            save_to_folder=self.tmp_dir,
        )
        for s in range(self.n_subjects):
            self.assertTrue(
                os.path.exists(os.path.join(self.tmp_dir, f"subject_export_{s}.csv")),
                f"Missing subject file subject_export_{s}.csv",
            )

    def test_single_model_report_default_name(self):
        """Verify single model report defaults to 'report_{model_name}.html' and not just 'report'."""
        def dummy_model(data, params, mode="log_likelihood"):
            return np.array([0.0])

        sampler = Sampler(
            self.data_single,
            dummy_model,
            self.hyper_params,
            model_name="m0",
            random_state=42,
        )
        # Mock evidence so report generation works
        sampler.evidence = [-100.0]
        sampler.hyper_params_list = [sampler.hyper_params]
        sampler.subj_evidence = [np.array([-30.0, -35.0, -35.0])]

        # 1. create_report without filename
        dashboard = sampler.create_report(show=False)
        self.assertEqual(dashboard.filename, "report_m0.html")

        # 2. Saving dashboard without filename argument uses report_{model_name}.html
        target_path = os.path.join(self.tmp_dir, "report_m0.html")
        dashboard.save(target_path)
        self.assertTrue(os.path.exists(target_path))

        # 3. Passing filename='report' normalizes to report_{model_name}.html
        dashboard_norm = sampler.create_report(filename=os.path.join(self.tmp_dir, "report"), show=False)
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "report_m0.html")))


if __name__ == "__main__":
    unittest.main()
