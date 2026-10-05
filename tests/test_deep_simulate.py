import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, call

from importance_sampling import Sampler, SimulatedDataList


class TestDeepSimulate(unittest.TestCase):
    def setUp(self):
        """Set up sample subject data and hyper_parameters for testing."""
        self.n_subjects = 3
        self.data = [
            {"trial": np.array([0, 1, 2]), "choice": np.array([1, 0, 1])},
            {"trial": np.array([0, 1, 2]), "choice": np.array([0, 1, 0])},
            {"trial": np.array([0, 1, 2]), "choice": np.array([1, 1, 0])},
        ]
        self.hyper_params = {
            "param1": {"mean": 0.5, "sd": 0.2},
            "param2": {"mean": 2.0, "sd": 0.5},
        }

    def test_single_function_delegates_to_all_subjects(self):
        """Verify passing a single callable delegates f to all subject model calls with mode='deep_simulate'."""
        recorded_calls = []

        def mock_model(subj_data, params, mode="log_likelihood", f=None):
            if mode == "deep_simulate":
                recorded_calls.append({"data": subj_data, "params": params, "f": f, "mode": mode})
                # Execute the passed function to generate simulated results
                sim_output = f(subj_data, params)
                return sim_output
            # Simple dummy log-likelihood for fitting/evaluation
            return np.array([0.0])

        sampler = Sampler(
            data=self.data,
            model=mock_model,
            hyper_params=self.hyper_params,
            random_state=42,
        )

        # Single function to delegate across all subjects
        def shared_function(data, params):
            return {"simulated_choice": data["choice"] ^ 1, "custom_tag": "shared"}

        result = sampler.deep_simulate(functions=shared_function)

        # 1. Output format check
        self.assertIsInstance(result, SimulatedDataList)
        self.assertEqual(len(result), self.n_subjects)

        # 2. Check that all subjects received the exact shared function
        self.assertEqual(len(recorded_calls), self.n_subjects)
        for i, call_info in enumerate(recorded_calls):
            self.assertIs(
                call_info["f"],
                shared_function,
                f"Subject {i} did not receive shared_function as f",
            )
            self.assertEqual(result[i]["custom_tag"], "shared")
            # Verify inversion applied by shared_function
            np.testing.assert_array_equal(
                result[i]["simulated_choice"], self.data[i]["choice"] ^ 1
            )

    def test_list_of_functions_delegates_each_to_subject(self):
        """Verify passing a list of functions delegates functions[i] -> f to subject i."""
        recorded_f = []

        def mock_model(subj_data, params, mode="simulate", f=None):
            recorded_f.append(f)
            return f(subj_data)

        sampler = Sampler(
            data=self.data,
            model=mock_model,
            hyper_params=self.hyper_params,
            random_state=42,
        )

        def fn_sub0(data):
            return {"subject": 0, "val": "func_0_result"}

        def fn_sub1(data):
            return {"subject": 1, "val": "func_1_result"}

        def fn_sub2(data):
            return {"subject": 2, "val": "func_2_result"}

        functions_list = [fn_sub0, fn_sub1, fn_sub2]
        result = sampler.deep_simulate(functions=functions_list)

        # Check that exactly 3 calls were made with respective functions
        self.assertEqual(len(recorded_f), self.n_subjects)
        self.assertIs(recorded_f[0], fn_sub0)
        self.assertIs(recorded_f[1], fn_sub1)
        self.assertIs(recorded_f[2], fn_sub2)

        # Check results
        self.assertEqual(result[0]["val"], "func_0_result")
        self.assertEqual(result[1]["val"], "func_1_result")
        self.assertEqual(result[2]["val"], "func_2_result")

    def test_mismatched_list_lengths_raise_value_error(self):
        """Verify mismatched list lengths raise an informative ValueError."""
        def dummy_model(subj_data, params, mode="simulate", f=None):
            return None

        sampler = Sampler(
            data=self.data,
            model=dummy_model,
            hyper_params=self.hyper_params,
            random_state=42,
        )

        # Dataset has 3 subjects; test length 2
        with self.assertRaises(ValueError) as ctx:
            sampler.deep_simulate(functions=[lambda d: 1, lambda d: 2])
        self.assertIn("3", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

        # Test length 4
        with self.assertRaises(ValueError) as ctx:
            sampler.deep_simulate(
                functions=[lambda d: 1, lambda d: 2, lambda d: 3, lambda d: 4]
            )
        self.assertIn("3", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))

        # Test empty list
        with self.assertRaises(ValueError) as ctx:
            sampler.deep_simulate(functions=[])
        self.assertIn("0", str(ctx.exception))

    def test_invalid_input_types_raise_type_error(self):
        """Verify invalid input types or non-callable elements raise TypeError."""
        def dummy_model(subj_data, params, mode="simulate", f=None):
            return None

        sampler = Sampler(
            data=self.data,
            model=dummy_model,
            hyper_params=self.hyper_params,
            random_state=42,
        )

        # Test non-callable scalar
        with self.assertRaises(TypeError):
            sampler.deep_simulate(functions="not_a_function")

        with self.assertRaises(TypeError):
            sampler.deep_simulate(functions=12345)

        with self.assertRaises(TypeError):
            sampler.deep_simulate(functions=None)

        with self.assertRaises(TypeError):
            sampler.deep_simulate(functions={"fn": lambda x: x})

        # Test list with correct length but non-callable item
        with self.assertRaises(TypeError) as ctx:
            sampler.deep_simulate(
                functions=[lambda d: 1, "non_callable_string", lambda d: 3]
            )
        self.assertIn("index 1", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            sampler.deep_simulate(functions=[lambda d: 1, 999, lambda d: 3])
        self.assertIn("index 1", str(ctx.exception))

    def test_deep_simulate_standard_arguments(self):
        """Verify deep_simulate accepts all standard simulate arguments (e.g. resample, n_simulations, to_df)."""
        recorded_calls = []

        def mock_model(subj_data, params, mode="simulate", f=None, **kwargs):
            recorded_calls.append({"n_samples": len(params["param1"]), "f": f})
            # Return DataFrame-like rows
            return pd.DataFrame({
                "trial": subj_data["trial"],
                "p_choice": [0.8] * len(subj_data["trial"]),
            })

        sampler = Sampler(
            data=self.data,
            model=mock_model,
            hyper_params=self.hyper_params,
            random_state=42,
        )

        fn = lambda d, p: None

        # 1. n_simulations
        sampler.deep_simulate(functions=fn, n_simulations=75, resample=False)
        self.assertEqual(recorded_calls[-1]["n_samples"], 75)

        # 2. to_df=True returns reconstructed DataFrame
        df_result = sampler.deep_simulate(functions=fn, to_df=True)
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertIn("subject", df_result.columns)
        self.assertEqual(len(df_result), 3 * 3)  # 3 subjects * 3 trials

        # 3. override_params
        custom_params = {"param1": np.array([0.9]), "param2": np.array([4.0])}
        res_override = sampler.deep_simulate(
            functions=fn,
            mode="override_params",
            override_params=custom_params,
        )
        self.assertEqual(len(res_override), self.n_subjects)

        # 4. Specific subjects subset
        fn_list = [lambda d, s=i: f"sub_{s}" for i in range(self.n_subjects)]
        res_subset = sampler.deep_simulate(functions=fn_list, subjects=[0, 2])
        self.assertEqual(len(res_subset), 2)

    def test_deep_simulate_with_various_model_signatures(self):
        """Verify f is correctly delegated across different model signature styles."""
        test_fn = lambda x: 99

        # Case A: model(data, params, f)
        def model_pos_f(data, params, f):
            return f(data)

        s1 = Sampler(self.data, model_pos_f, self.hyper_params)
        res1 = s1.deep_simulate(test_fn)
        self.assertEqual(res1[0], 99)

        # Case B: model(data, params, mode="simulate", f=None)
        def model_kw_f(data, params, mode="simulate", f=None):
            return f(data)

        s2 = Sampler(self.data, model_kw_f, self.hyper_params)
        res2 = s2.deep_simulate(test_fn)
        self.assertEqual(res2[0], 99)

        # Case C: model(data, params, **kwargs)
        def model_var_kw(data, params, **kwargs):
            return kwargs["f"](data)

        s3 = Sampler(self.data, model_var_kw, self.hyper_params)
        res3 = s3.deep_simulate(test_fn)
        self.assertEqual(res3[0], 99)

    def test_mode_deep_simulate_vs_simulate(self):
        """Verify model receives mode='deep_simulate' in deep_simulate and mode='simulate' in simulate."""
        received_modes = []

        def branching_model(data, params, mode="log_likelihood", f=None):
            received_modes.append(mode)
            if mode == "deep_simulate":
                return {"mode_received": mode, "f_eval": f(data) if f else None}
            elif mode == "simulate":
                return {"mode_received": mode}
            return np.array([0.0])

        sampler = Sampler(self.data, branching_model, self.hyper_params, random_state=42)

        # 1. deep_simulate call
        deep_res = sampler.deep_simulate(functions=lambda d: "deep_metric")
        self.assertTrue(all(m == "deep_simulate" for m in received_modes[:self.n_subjects]))
        for subj_out in deep_res:
            self.assertEqual(subj_out["mode_received"], "deep_simulate")
            self.assertEqual(subj_out["f_eval"], "deep_metric")

        # 2. simulate call
        received_modes.clear()
        sim_res = sampler.simulate()
        self.assertTrue(all(m == "simulate" for m in received_modes[:self.n_subjects]))
        for subj_out in sim_res:
            self.assertEqual(subj_out["mode_received"], "simulate")

    def test_fallback_to_simulate_mode_for_legacy_model(self):
        """Verify models that only inspect mode == 'simulate' still work seamlessly via fallback."""
        def legacy_model(data, params, mode="log_likelihood", f=None):
            # Model only knows about "simulate", not "deep_simulate"
            if mode == "simulate":
                return {"legacy_result": True, "f_val": f(data) if f else None}
            return None

        sampler = Sampler(self.data, legacy_model, self.hyper_params, random_state=42)
        res = sampler.deep_simulate(functions=lambda d: "legacy_f")
        for subj_out in res:
            self.assertTrue(subj_out["legacy_result"])
            self.assertEqual(subj_out["f_val"], "legacy_f")


if __name__ == "__main__":
    unittest.main()

