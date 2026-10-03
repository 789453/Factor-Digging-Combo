from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


class ExperimentStore:
    """Small durable job store for resumable expression evaluation."""

    def __init__(self, path: str):
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(db_path))
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                config_hash VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                manifest_json JSON
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_cache (
                evaluation_signature VARCHAR NOT NULL,
                expr_hash VARCHAR NOT NULL,
                result_json JSON NOT NULL,
                source_experiment_id VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL,
                PRIMARY KEY (evaluation_signature, expr_hash)
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS expression_jobs (
                experiment_id VARCHAR NOT NULL,
                expr_hash VARCHAR NOT NULL,
                expr VARCHAR NOT NULL,
                canonical VARCHAR NOT NULL,
                template_family VARCHAR NOT NULL,
                metadata_json JSON NOT NULL,
                status VARCHAR NOT NULL,
                result_json JSON,
                error VARCHAR,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (experiment_id, expr_hash)
            )
        """)

    def initialize(self, experiment_id: str, config_hash: str) -> None:
        existing = self.connection.execute(
            "SELECT config_hash, status FROM experiments WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
        if existing:
            if existing[0] != config_hash:
                raise ValueError("Experiment ID exists with a different configuration")
            if existing[1] == "COMPLETED":
                raise FileExistsError(f"Experiment {experiment_id} is already completed")
            return
        self.connection.execute(
            """
            INSERT INTO experiments
            (experiment_id, config_hash, status, started_at)
            VALUES (?, ?, 'RUNNING', ?)
            """,
            [experiment_id, config_hash, datetime.now(timezone.utc)],
        )

    def enqueue(self, experiment_id: str, records) -> None:
        now = datetime.now(timezone.utc)
        rows = []
        for record in records:
            metadata = {
                "template_name": record.template_name,
                "rank_equivalence_hash": record.rank_equivalence_hash,
                "template_order": record.template_order,
                "fields": record.fields,
                "operators": record.operators,
                "windows": record.windows,
                "depth": record.depth,
                "nodes": record.nodes,
                "priority_score": record.priority_score,
            }
            rows.append((
                experiment_id,
                record.expr_hash,
                record.expr,
                record.canonical,
                record.template_family,
                json.dumps(metadata, ensure_ascii=False),
                "PENDING",
                now,
            ))
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO expression_jobs
            (experiment_id, expr_hash, expr, canonical, template_family,
             metadata_json, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def completed_results(self, experiment_id: str) -> dict[str, dict]:
        rows = self.connection.execute(
            """
            SELECT expr_hash, result_json
            FROM expression_jobs
            WHERE experiment_id = ? AND status = 'COMPLETED'
            """,
            [experiment_id],
        ).fetchall()
        return {expr_hash: json.loads(result) for expr_hash, result in rows}

    def compatible_experiment_results(
        self,
        evaluation_signature: str,
        expr_hashes: set[str],
    ) -> dict[str, dict]:
        """Reuse immutable results from a completed experiment with identical inputs."""
        if not expr_hashes:
            return {}
        experiments = self.connection.execute(
            """
            SELECT experiment_id, manifest_json
            FROM experiments
            WHERE status = 'COMPLETED' AND manifest_json IS NOT NULL
            ORDER BY completed_at DESC
            """
        ).fetchall()
        compatible_ids = []
        for experiment_id, raw_manifest in experiments:
            manifest = json.loads(raw_manifest)
            if manifest.get("evaluation_signature") == evaluation_signature:
                compatible_ids.append(experiment_id)
        output: dict[str, dict] = {}
        for experiment_id in compatible_ids:
            rows = self.connection.execute(
                """
                SELECT expr_hash, result_json
                FROM expression_jobs
                WHERE experiment_id = ? AND status = 'COMPLETED'
                """,
                [experiment_id],
            ).fetchall()
            for expr_hash, result in rows:
                if expr_hash in expr_hashes and expr_hash not in output:
                    output[expr_hash] = json.loads(result)
            if len(output) == len(expr_hashes):
                break
        return output

    def write_result(
        self,
        experiment_id: str,
        expr_hash: str,
        result: dict,
    ) -> None:
        self.connection.execute(
            """
            UPDATE expression_jobs
            SET status = 'COMPLETED', result_json = ?, error = NULL, updated_at = ?
            WHERE experiment_id = ? AND expr_hash = ?
            """,
            [
                json.dumps(result, ensure_ascii=False),
                datetime.now(timezone.utc),
                experiment_id,
                expr_hash,
            ],
        )

    def write_results_batch(
        self,
        experiment_id: str,
        rows: list[tuple[str, dict]],
    ) -> None:
        """Persist a checkpoint batch in one DuckDB transaction."""
        if not rows:
            return
        now = datetime.now(timezone.utc)
        self.connection.executemany(
            """
            UPDATE expression_jobs
            SET status = 'COMPLETED', result_json = ?, error = NULL, updated_at = ?
            WHERE experiment_id = ? AND expr_hash = ?
            """,
            [
                (json.dumps(result, ensure_ascii=False), now, experiment_id, expr_hash)
                for expr_hash, result in rows
            ],
        )

    def cached_results(
        self,
        evaluation_signature: str,
        expr_hashes: set[str],
    ) -> dict[str, dict]:
        if not expr_hashes:
            return {}
        rows = self.connection.execute(
            """
            SELECT expr_hash, result_json
            FROM evaluation_cache
            WHERE evaluation_signature = ?
            """,
            [evaluation_signature],
        ).fetchall()
        return {
            expr_hash: json.loads(result)
            for expr_hash, result in rows
            if expr_hash in expr_hashes
        }

    def cache_result(
        self,
        evaluation_signature: str,
        experiment_id: str,
        expr_hash: str,
        result: dict,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO evaluation_cache
            (evaluation_signature, expr_hash, result_json,
             source_experiment_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                evaluation_signature,
                expr_hash,
                json.dumps(result, ensure_ascii=False),
                experiment_id,
                datetime.now(timezone.utc),
            ],
        )

    def cache_results_batch(
        self,
        evaluation_signature: str,
        experiment_id: str,
        rows: list[tuple[str, dict]],
    ) -> None:
        if not rows:
            return
        now = datetime.now(timezone.utc)
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO evaluation_cache
            (evaluation_signature, expr_hash, result_json,
             source_experiment_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    evaluation_signature,
                    expr_hash,
                    json.dumps(result, ensure_ascii=False),
                    experiment_id,
                    now,
                )
                for expr_hash, result in rows
            ],
        )

    def write_failure(
        self,
        experiment_id: str,
        expr_hash: str,
        error: str,
        result: dict,
    ) -> None:
        self.connection.execute(
            """
            UPDATE expression_jobs
            SET status = 'FAILED', result_json = ?, error = ?, updated_at = ?
            WHERE experiment_id = ? AND expr_hash = ?
            """,
            [
                json.dumps(result, ensure_ascii=False),
                error,
                datetime.now(timezone.utc),
                experiment_id,
                expr_hash,
            ],
        )

    def complete(self, experiment_id: str, manifest: dict) -> None:
        self.connection.execute(
            """
            UPDATE experiments
            SET status = 'COMPLETED', completed_at = ?, manifest_json = ?
            WHERE experiment_id = ?
            """,
            [
                datetime.now(timezone.utc),
                json.dumps(manifest, ensure_ascii=False),
                experiment_id,
            ],
        )

    def close(self) -> None:
        self.connection.close()
