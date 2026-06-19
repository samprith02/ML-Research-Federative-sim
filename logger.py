"""
logger.py
=========
Phase 2 -- Per-Round Experiment Logger

Logs training metrics for each federated learning experiment to:
  - CSV: one row per round (global metrics + timing)
  - JSON: nested per-client metrics per round (accuracy, loss, label distribution)

Usage
-----
    from logger import RoundLogger

    logger = RoundLogger(experiment_name="iid_baseline", output_dir="results/phase2")

    # At the start of each round:
    logger.log_round(
        round_idx=1,
        global_acc=0.75,
        global_loss=0.88,
        selected_clients=[0, 3, 7, ...],
        round_time=32.5,
        client_metrics={
            0: {"acc": 0.80, "loss": 0.72, "n_samples": 500, "label_dist": {...}},
            3: {"acc": 0.71, "loss": 0.94, "n_samples": 500, "label_dist": {...}},
            ...
        }
    )

    # At the end of the experiment:
    logger.save()
    logger.print_summary()
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional


class RoundLogger:
    """Records per-round and per-client metrics for a single FL experiment.

    Parameters
    ----------
    experiment_name : str
        Short identifier used in output filenames (e.g. "iid_baseline").
    output_dir : str
        Directory to save CSV and JSON files.
    """

    def __init__(self, experiment_name: str, output_dir: str = "results/phase2") -> None:
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # In-memory stores
        self._round_rows: List[Dict[str, Any]] = []  # one dict per round for CSV
        self._client_log: Dict[int, Dict] = {}       # {round: {client_id: metrics}}

        # File paths
        self.csv_path = os.path.join(output_dir, f"{experiment_name}_rounds.csv")
        self.json_path = os.path.join(output_dir, f"{experiment_name}_clients.json")

        # CSV header
        self._csv_header = [
            "round",
            "global_acc",
            "global_loss",
            "n_selected_clients",
            "selected_clients",
            "round_time_s",
            "cumulative_time_s",
        ]
        self._start_time = time.time()
        self._cumulative_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_round(
        self,
        round_idx: int,
        global_acc: float,
        global_loss: float,
        selected_clients: List[int],
        round_time: float,
        client_metrics: Optional[Dict[int, Dict]] = None,
    ) -> None:
        """Record metrics for one communication round.

        Parameters
        ----------
        round_idx : int
            1-indexed communication round number.
        global_acc : float
            Global test accuracy (0.0 - 1.0 or percentage, consistent with your code).
        global_loss : float
            Global test loss.
        selected_clients : list of int
            Client IDs that participated this round.
        round_time : float
            Wall-clock time for this round in seconds.
        client_metrics : dict, optional
            {client_id: {"acc": float, "loss": float, "n_samples": int,
                          "label_dist": {class: count}}}
        """
        self._cumulative_time += round_time

        # CSV row
        row = {
            "round": round_idx,
            "global_acc": round(global_acc, 6),
            "global_loss": round(global_loss, 6),
            "n_selected_clients": len(selected_clients),
            "selected_clients": json.dumps(selected_clients),
            "round_time_s": round(round_time, 2),
            "cumulative_time_s": round(self._cumulative_time, 2),
        }
        self._round_rows.append(row)

        # Per-client JSON entry
        if client_metrics is not None:
            self._client_log[round_idx] = {}
            for cid, metrics in client_metrics.items():
                self._client_log[round_idx][cid] = {
                    "acc": round(float(metrics.get("acc", 0.0)), 6),
                    "loss": round(float(metrics.get("loss", 0.0)), 6),
                    "n_samples": int(metrics.get("n_samples", 0)),
                    "label_dist": metrics.get("label_dist", {}),
                }

    def save(self) -> None:
        """Write all accumulated logs to CSV and JSON files."""
        self._write_csv()
        self._write_json()

    def print_summary(self) -> None:
        """Print a compact summary table of all logged rounds."""
        if not self._round_rows:
            print("No rounds logged yet.")
            return

        print(f"\n{'='*65}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*65}")
        print(f"  {'Round':>6}  {'Acc (%)':>8}  {'Loss':>8}  {'Time (s)':>9}  {'Total (s)':>10}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}")
        for row in self._round_rows:
            acc = row["global_acc"]
            # Handle both 0-1 and 0-100 ranges
            acc_pct = acc if acc > 1.0 else acc * 100.0
            print(
                f"  {row['round']:>6}  {acc_pct:>7.2f}%  "
                f"{row['global_loss']:>8.4f}  "
                f"{row['round_time_s']:>9.1f}  "
                f"{row['cumulative_time_s']:>10.1f}"
            )
        last = self._round_rows[-1]
        acc_final = last["global_acc"]
        acc_pct_final = acc_final if acc_final > 1.0 else acc_final * 100.0
        print(f"{'='*65}")
        print(f"  Final accuracy: {acc_pct_final:.2f}%  |  "
              f"Total time: {last['cumulative_time_s']:.1f}s")
        print(f"  CSV -> {self.csv_path}")
        print(f"  JSON -> {self.json_path}")

    def get_acc_curve(self) -> List[float]:
        """Return list of global_acc values across all logged rounds."""
        return [r["global_acc"] for r in self._round_rows]

    def get_loss_curve(self) -> List[float]:
        """Return list of global_loss values across all logged rounds."""
        return [r["global_loss"] for r in self._round_rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_csv(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_header)
            writer.writeheader()
            writer.writerows(self._round_rows)

    def _write_json(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._client_log, f, indent=2)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("RoundLogger -- Smoke Test")
    print("=" * 50)

    logger = RoundLogger(
        experiment_name="_smoke_test_logger",
        output_dir="results/phase2",
    )

    import random
    rng = random.Random(42)
    acc = 0.10
    loss = 2.30
    for r in range(1, 6):
        acc += rng.uniform(0.05, 0.15)
        loss -= rng.uniform(0.1, 0.4)
        clients = rng.sample(range(100), 10)
        client_metrics = {
            cid: {
                "acc": acc + rng.uniform(-0.05, 0.05),
                "loss": max(0.01, loss + rng.uniform(-0.2, 0.2)),
                "n_samples": 500,
                "label_dist": {str(i): rng.randint(0, 100) for i in range(10)},
            }
            for cid in clients
        }
        logger.log_round(
            round_idx=r,
            global_acc=min(acc, 0.99),
            global_loss=max(loss, 0.01),
            selected_clients=clients,
            round_time=rng.uniform(25, 40),
            client_metrics=client_metrics,
        )

    logger.save()
    logger.print_summary()

    import os
    assert os.path.exists(logger.csv_path), "CSV not created"
    assert os.path.exists(logger.json_path), "JSON not created"
    print("\n[OK] RoundLogger smoke test passed.")
