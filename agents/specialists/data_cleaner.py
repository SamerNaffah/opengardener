"""
DataCleanerAgent — specializes in CSV data cleaning tasks.

Strategies it can employ (emergent via soil trails):
  - pandas_dropna: drop null rows, deduplicate
  - regex_cleaning: apply regex substitutions
  - schema_validation: validate against a JSON schema
  - fill_missing: fill NaN with median/mode

Round-2 eval additions:
  - clean() accepts difficulty: float (0-1); success threshold = 1 - difficulty
  - INJECT_MALFORMED env var: corrupt a fraction of rows before cleaning
    so the agent faces real work on hard tasks
"""

import io
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from base.agent import EmergentAgent, TaskOutcome

logger = logging.getLogger(__name__)


class DataCleanerAgent(EmergentAgent):
    def __init__(self, agent_id: Optional[str] = None):
        genome = {
            "specialization": "data_cleaning",
            "default_approach": {
                "method": "pandas_dropna",
                "drop_duplicates": True,
                "fill_strategy": None,
            },
            "fallback_approach": {
                "method": "regex_cleaning",
                "patterns": ["\\s+", "\\x00"],
            },
        }
        super().__init__(agent_id=agent_id, genome=genome)

    @property
    def task_domain(self) -> str:
        return "data_cleaning"

    def clean(self, file_path: str, schema: Optional[dict] = None, difficulty: float = 0.0) -> dict:
        """
        Clean a CSV file using the strategy from the Soil.
        difficulty (0-1): higher values require retaining more rows to count as success
                          and trigger malformed-CSV injection when INJECT_MALFORMED=true.
        Returns a result dict with metadata about what was done.
        """
        task_desc = f"clean CSV file: {Path(file_path).name}"
        if schema:
            task_desc += f" with schema: {list(schema.keys())}"

        # Optionally inject malformation for hard tasks
        inject = os.getenv("INJECT_MALFORMED", "false").lower() in ("1", "true", "yes")
        working_path = file_path
        if inject and difficulty > 0.5:
            working_path = self._inject_malformed(file_path, difficulty)

        strategy = self.before_task(task_desc)
        start = time.time()

        try:
            result = self._execute_strategy(working_path, strategy, schema, difficulty)
            elapsed_ms = (time.time() - start) * 1000

            self.after_task(
                task_desc,
                outcome=TaskOutcome(success=result.get("success", False), data=result),
                resources={"cpu_ms": elapsed_ms, "memory_mb": 50.0},
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning(f"[{self.id}] Clean failed: {e}")

            self.after_task(
                task_desc,
                outcome=TaskOutcome(success=False, error=str(e)),
                resources={"cpu_ms": elapsed_ms},
            )

            # Retry once with mutated approach
            if self.experience_buffer:
                self.current_strategy = self.mutate_approach(task_desc)
                try:
                    result = self._execute_strategy(working_path, self.current_strategy, schema, difficulty)
                    self.after_task(
                        task_desc,
                        outcome=TaskOutcome(success=result.get("success", False), data=result),
                        resources={"cpu_ms": (time.time() - start) * 1000},
                    )
                    return result
                except Exception as e2:
                    logger.error(f"[{self.id}] Retry also failed: {e2}")

            return {"success": False, "error": str(e)}

    def _inject_malformed(self, file_path: str, difficulty: float) -> str:
        """Write a corrupted copy of the CSV to /tmp and return its path."""
        try:
            df = pd.read_csv(file_path)
        except Exception:
            return file_path  # can't corrupt what we can't read — use original

        rng = random.Random()
        n_corrupt = max(1, int(len(df) * rng.uniform(0.10, 0.30) * difficulty))
        indices = rng.sample(range(len(df)), min(n_corrupt, len(df)))

        for idx in indices:
            col = rng.choice(df.columns.tolist())
            df.at[idx, col] = rng.choice(["\x00JUNK\x00", "N/A;bad", 99999999, None])

        out = os.path.join("/tmp", "malformed_" + os.path.basename(file_path))
        df.to_csv(out, index=False)
        return out

    def _execute_strategy(self, file_path: str, strategy: dict, schema: Optional[dict], difficulty: float = 0.0) -> dict:
        method = strategy.get("method", "pandas_dropna")

        if method == "pandas_dropna":
            return self._pandas_dropna(file_path, strategy, difficulty)
        elif method == "regex_cleaning":
            return self._regex_cleaning(file_path, strategy)
        elif method == "fill_missing":
            return self._fill_missing(file_path, strategy, difficulty)
        elif method == "schema_validation":
            return self._schema_validation(file_path, schema or {})
        else:
            logger.warning(f"[{self.id}] Unknown method '{method}', falling back to pandas_dropna")
            return self._pandas_dropna(file_path, strategy, difficulty)

    def _pandas_dropna(self, file_path: str, strategy: dict, difficulty: float = 0.0) -> dict:
        df = pd.read_csv(file_path, on_bad_lines="skip")
        original_rows = len(df)
        original_cols = len(df.columns)

        df = df.dropna()
        if strategy.get("drop_duplicates", True):
            df = df.drop_duplicates()

        cleaned_rows = len(df)
        # Tightened threshold: must retain at least (1 - 0.5*difficulty) fraction.
        # At difficulty=0.6: need ≥70% rows (was 40%). Forces method choice to matter.
        threshold = max(0.0, 1.0 - 0.5 * difficulty)
        success = (cleaned_rows / original_rows) >= threshold if original_rows > 0 else True

        import os as _os
        output_path = _os.path.join("/tmp", _os.path.basename(file_path).replace(".csv", "_cleaned.csv"))
        df.to_csv(output_path, index=False)

        return {
            "success": success,
            "method": "pandas_dropna",
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "rows_removed": original_rows - cleaned_rows,
            "columns": original_cols,
            "retention_ratio": cleaned_rows / original_rows if original_rows > 0 else 1.0,
            "threshold": threshold,
            "output": output_path,
        }

    def _regex_cleaning(self, file_path: str, strategy: dict) -> dict:
        patterns = strategy.get("patterns", [r"\s+", r"\x00"])

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        original_len = len(content)
        for pattern in patterns:
            content = re.sub(pattern, " ", content)

        import os as _os
        output_path = _os.path.join("/tmp", _os.path.basename(file_path).replace(".csv", "_regex_cleaned.csv"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "method": "regex_cleaning",
            "original_chars": original_len,
            "cleaned_chars": len(content),
            "output": output_path,
        }

    def _fill_missing(self, file_path: str, strategy: dict, difficulty: float = 0.0) -> dict:
        df = pd.read_csv(file_path, on_bad_lines="skip")
        original_rows = len(df)
        fill_strategy = strategy.get("fill_strategy", "median")

        for col in df.select_dtypes(include=["number"]).columns:
            if fill_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_strategy == "zero":
                df[col] = df[col].fillna(0)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("UNKNOWN")

        cleaned_rows = len(df)
        # Tightened threshold: mirrors _pandas_dropna change (1 - 0.5*difficulty).
        threshold = max(0.0, 1.0 - 0.5 * difficulty)
        success = (cleaned_rows / original_rows) >= threshold if original_rows > 0 else True

        import os as _os
        output_path = _os.path.join("/tmp", _os.path.basename(file_path).replace(".csv", "_filled.csv"))
        df.to_csv(output_path, index=False)

        return {
            "success": success,
            "method": "fill_missing",
            "fill_strategy": fill_strategy,
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "output": output_path,
        }

    def _schema_validation(self, file_path: str, schema: dict) -> dict:
        df = pd.read_csv(file_path, on_bad_lines="skip")
        violations = []

        for col, expected_type in schema.items():
            if col not in df.columns:
                violations.append(f"Missing column: {col}")
                continue

            if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                violations.append(f"Column '{col}' expected numeric, got {df[col].dtype}")

        return {
            "success": len(violations) == 0,
            "method": "schema_validation",
            "violations": violations,
            "rows": len(df),
        }
