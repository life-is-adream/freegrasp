import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent


EXPECTED_SUBSETS = {
    "evaluate1.py": "easy_no_ambi",
    "evaluate2.py": "easy_yes_ambi",
    "evaluate3.py": "medium_no_ambi",
    "evaluate4.py": "medium_yes_ambi",
    "evaluate5.py": "hard_no_ambi",
    "evaluate6.py": "hard_yes_ambi",
}


def get_subset_name(script_path: pathlib.Path) -> str | None:
    tree = ast.parse(script_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "subset_name":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return None


class EvaluateEntrypointsTest(unittest.TestCase):
    def test_subset_entrypoints_exist_and_pin_expected_subset(self) -> None:
        for filename, expected_subset in EXPECTED_SUBSETS.items():
            script_path = PROJECT_ROOT / filename
            self.assertTrue(script_path.exists(), f"{filename} should exist")
            self.assertEqual(get_subset_name(script_path), expected_subset)

    def test_evaluate_all_exists_and_lists_all_subsets_in_order(self) -> None:
        script_path = PROJECT_ROOT / "evaluate_all.py"
        self.assertTrue(script_path.exists(), "evaluate_all.py should exist")

        tree = ast.parse(script_path.read_text())
        subset_values = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "subset_names":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    subset_values.append(elt.value)

        self.assertEqual(subset_values, list(EXPECTED_SUBSETS.values()))


if __name__ == "__main__":
    unittest.main()
