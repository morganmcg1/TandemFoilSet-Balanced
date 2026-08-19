import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import simple_parsing as sp

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.contract import manifest_sha256, require_materialized_manifest
from eval_runtime import (
    apply_torch_seed,
    arm_hard_timeout,
    is_complete_test_result,
    resolve_trial_seed,
    resolve_wandb_group,
    timeout_minutes_from_env,
)


def load_config_class():
    train_path = Path(__file__).parents[1] / "train.py"
    tree = ast.parse(train_path.read_text())
    config_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config"
    )
    namespace = {"dataclass": dataclass, "os": os}
    exec(compile(ast.Module([config_node], []), train_path, "exec"), namespace)
    return namespace["Config"]


def load_wandb_init_keywords():
    train_path = Path(__file__).parents[1] / "train.py"
    tree = ast.parse(train_path.read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "wandb"
        and node.func.attr == "init"
    )
    return {keyword.arg: keyword.value for keyword in call.keywords}


class ConfigTest(unittest.TestCase):
    def test_prepare_splits_records_the_source_manifest_digest(self):
        tree = ast.parse((Path(__file__).parents[1] / "data" / "prepare_splits.py").read_text())
        digest_entries = [
            (key, value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "split_manifest_sha256"
        ]
        self.assertEqual(len(digest_entries), 1)
        self.assertEqual(ast.unparse(digest_entries[0][1]), "manifest_digest")

    def test_wandb_init_records_the_effective_group_and_eval_contract(self):
        keywords = load_wandb_init_keywords()
        self.assertEqual(ast.unparse(keywords["group"]), "cfg.wandb_group")
        config_keys = {
            key.value
            for key in keywords["config"].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertGreaterEqual(
            config_keys,
            {
                "wandb_run_group",
                "senpai_timeout_minutes",
                "senpai_trial_index",
                "senpai_trial_seed",
                "training_source_sha256",
                "split_manifest_sha256",
                "materialized_split_manifest_sha256",
                "data_contract_satisfied",
                "scoring_source_sha256",
                "loader_source_sha256",
                "metric_contract",
            },
        )

    def test_wandb_group_env_overrides_cli(self):
        config_class = load_config_class()

        parsed = sp.parse(config_class, args=["--wandb_group", "explicit-group"])
        with patch.dict(os.environ, {"WANDB_RUN_GROUP": "eval-group"}):
            self.assertEqual(resolve_wandb_group(parsed.wandb_group), "eval-group")

    def test_wandb_group_uses_cli_outside_harness(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_wandb_group("local-group"), "local-group")

    def test_trial_seed_env_overrides_cli_and_local_seed_still_works(self):
        with patch.dict(os.environ, {"SENPAI_TRIAL_SEED": "29"}):
            self.assertEqual(resolve_trial_seed(7), 29)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_trial_seed(7), 7)

    def test_authoritative_seed_is_applied_to_cpu_and_cuda(self):
        torch_module = Mock()
        torch_module.cuda.is_available.return_value = True

        apply_torch_seed(torch_module, 29)

        torch_module.manual_seed.assert_called_once_with(29)
        torch_module.cuda.manual_seed_all.assert_called_once_with(29)

    def test_test_ranking_requires_exact_splits_finite_values_and_equal_mean(self):
        splits = ["test_a", "test_b", "test_c", "test_d"]
        metrics = {
            name: {"mae_surf_p": value, "loss": value / 10}
            for name, value in zip(splits, (1.0, 2.0, 3.0, 4.0))
        }
        averages = {"avg/mae_surf_p": 2.5, "avg/loss": 0.25}
        self.assertTrue(is_complete_test_result(metrics, averages, splits))

        self.assertFalse(is_complete_test_result(metrics, {**averages, "avg/mae_surf_p": 2.4}, splits))
        self.assertFalse(is_complete_test_result({k: v for k, v in metrics.items() if k != "test_d"}, averages, splits))
        nonfinite = {name: dict(values) for name, values in metrics.items()}
        nonfinite["test_d"]["loss"] = float("nan")
        self.assertFalse(is_complete_test_result(nonfinite, averages, splits))

    def test_materialized_dataset_is_bound_to_exact_manifest_and_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "split_manifest.json"
            splits_dir = root / "splits"
            manifest = {
                "split_counts": {"train": 2, "val_a": 1, "test_a": 1},
                "test_splits": ["test_a"],
            }
            manifest_path.write_text(json.dumps(manifest))
            for split_name, count in manifest["split_counts"].items():
                split_dir = splits_dir / split_name
                split_dir.mkdir(parents=True)
                for index in range(count):
                    (split_dir / f"{index:06d}.pt").touch()
            gt_dir = splits_dir / ".test_a_gt"
            gt_dir.mkdir()
            (gt_dir / "000000.pt").touch()
            meta = {
                "split_manifest_sha256": manifest_sha256(manifest_path),
                "split_counts": manifest["split_counts"],
            }
            (splits_dir / "meta.json").write_text(json.dumps(meta))

            self.assertEqual(
                require_materialized_manifest(splits_dir, manifest_path),
                manifest_sha256(manifest_path),
            )

            meta["split_manifest_sha256"] = "stale"
            (splits_dir / "meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(RuntimeError, "Materialized splits are stale"):
                require_materialized_manifest(splits_dir, manifest_path)

            meta["split_manifest_sha256"] = manifest_sha256(manifest_path)
            (splits_dir / "meta.json").write_text(json.dumps(meta))
            (splits_dir / "train" / "000002.pt").touch()
            with self.assertRaisesRegex(RuntimeError, "has 3 files"):
                require_materialized_manifest(splits_dir, manifest_path)

    def test_timeout_requires_a_positive_finite_value(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"SENPAI_TIMEOUT_MINUTES": value}
            ):
                with self.assertRaises(ValueError):
                    timeout_minutes_from_env()

    @patch("eval_runtime.threading.Timer")
    def test_hard_timeout_starts_daemon_timer(self, timer_class):
        timer = timer_class.return_value

        self.assertIs(arm_hard_timeout(1.5), timer)
        timer_class.assert_called_once()
        self.assertEqual(timer_class.call_args.args[0], 90.0)
        self.assertTrue(timer.daemon)
        timer.start.assert_called_once_with()

        terminate = timer_class.call_args.args[1]
        with patch("eval_runtime.os.write"), patch("eval_runtime.os._exit") as hard_exit:
            terminate()
        hard_exit.assert_called_once_with(124)

    def test_watchdog_actually_terminates_a_process(self):
        env = {**os.environ, "SENPAI_TIMEOUT_MINUTES": "0.001"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from eval_runtime import arm_hard_timeout, timeout_minutes_from_env; "
                "import time; arm_hard_timeout(timeout_minutes_from_env()); time.sleep(2)",
            ],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
        )
        self.assertEqual(result.returncode, 124)
        self.assertIn("expired; terminating process", result.stderr)


if __name__ == "__main__":
    unittest.main()
