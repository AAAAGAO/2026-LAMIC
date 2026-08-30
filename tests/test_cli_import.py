import importlib
import unittest


class CliImportTests(unittest.TestCase):
    def test_experiment_runner_imports(self) -> None:
        experiments = importlib.import_module("LAMIC.experiments")

        self.assertTrue(callable(experiments.run_rq_experiment))


if __name__ == "__main__":
    unittest.main()
