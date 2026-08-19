import ast
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import simple_parsing as sp


def load_config_class():
    train_path = Path(__file__).parents[1] / "train.py"
    tree = ast.parse(train_path.read_text())
    config_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config"
    )
    namespace = {"dataclass": dataclass, "os": os}
    exec(compile(ast.Module([config_node], []), train_path, "exec"), namespace)
    return namespace["Config"]


class ConfigTest(unittest.TestCase):
    def test_wandb_group_uses_env_default_and_cli_override(self):
        with patch.dict(os.environ, {"WANDB_RUN_GROUP": "eval-group"}):
            config_class = load_config_class()

        self.assertEqual(config_class().wandb_group, "eval-group")
        self.assertEqual(
            sp.parse(config_class, args=["--wandb_group", "explicit-group"]).wandb_group,
            "explicit-group",
        )


if __name__ == "__main__":
    unittest.main()
