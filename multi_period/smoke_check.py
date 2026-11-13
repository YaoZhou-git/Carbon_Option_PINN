"""Load six packaged models and compare existing recorded prices; no training or integration."""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import sys

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "core"))
from factor_model import factors
from factor_pinn import FactorPINN


def main():
    torch.set_num_threads(4)
    with (PACKAGE / "reference/data/saved_PINN_input.csv").open(encoding="utf-8-sig") as stream:
        baseline = {int(row["N"]): float(row["PINN_C"]) for row in csv.DictReader(stream)
                    if float(row["A1"]) == 60.0}
    with (PACKAGE / "primary/saved/matched_pinn_prices.csv").open(encoding="utf-8-sig") as stream:
        matched = {row["curve"]: float(row["PINN_price"]) for row in csv.DictReader(stream)}
    cases = [(f"baseline_N{n}", PACKAGE / f"models/baseline/N{n}_factor_pinn.pt",
              [60.0, 45.0, 30.0, 20.0][:n], baseline[n]) for n in range(1, 5)]
    cases += [("matched_flat", PACKAGE / "models/matched/N4_matched_flat.pt", [60.0] * 4,
               matched["flat"]),
              ("matched_upward", PACKAGE / "models/matched/N4_matched_upward.pt",
               [60.0, 65.0, 70.0, 75.0], matched["upward"])]
    records = []
    for name, path, quotes, expected in cases:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SimpleNamespace(**checkpoint["parameters"])
        net = FactorPINN(cfg, checkpoint["N"])
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        net.eval()
        state = torch.tensor(np.r_[factors(quotes, cfg) / cfg.pi, 0.0][None], dtype=torch.float32)
        with torch.no_grad():
            actual = float(net(state).item() * cfg.pi)
        assert np.isfinite(actual) and abs(actual - expected) < 2e-4, (name, actual, expected)
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"case": name, "price": actual, "saved_price": expected,
                        "absolute_difference": abs(actual - expected), "checkpoint_sha256": before})
    output = PACKAGE / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "models_loaded": len(records), "training": False,
              "reference_integration": False, "records": records}
    (output / "smoke_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
