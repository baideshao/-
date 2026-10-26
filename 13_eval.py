# -*- coding: utf-8 -*-
"""Evaluate all trackers with TrackEval (HOTA + CLEAR + Identity) on the pig MOT sequences."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:/PIG_MOT_TEST_DATA/tracking_experiments/scripts")

from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
from trackeval import Evaluator
from trackeval import metrics as M

EXP = Path(r"D:/PIG_MOT_TEST_DATA/tracking_experiments")
RES_DIR = EXP / "results" / "trackers" / "mot_challenge"


class PigMot(MotChallenge2DBox):
    """MOTChallenge-format dataset with the pig class (class id 1)."""

    def __init__(self, config=None):
        config = dict(config or {})
        config["CLASSES_TO_EVAL"] = ["pedestrian"]  # satisfy the built-in check, patched below
        super().__init__(config)
        # 'pig' is class id 1; keep the built-in distractor names mapped to unused ids
        # so TrackEval's hard-coded distractor lookup does not fail (they match nothing)
        self.class_name_to_class_id = {
            "pig": 1,
            "person_on_vehicle": 90, "static_person": 91,
            "distractor": 92, "reflection": 93, "non_mot_vehicle": 94,
        }
        self.valid_class_numbers = [1]
        self.class_list = ["pig"]
        self.valid_classes = ["pig"]


def main():
    tracker_names = sorted(
        p.name for p in (RES_DIR / "pig-train").iterdir() if p.is_dir())
    config = {
        "GT_FOLDER": str(EXP / "data" / "gt" / "mot_challenge"),
        "TRACKERS_FOLDER": str(RES_DIR),
        "TRACKERS_TO_EVAL": tracker_names,
        "BENCHMARK": "pig",
        "SPLIT_TO_EVAL": "train",
        "SEQMAP_FILE": str(EXP / "data" / "gt" / "mot_challenge" / "pig-train" / "seqmaps" / "pig-train.txt"),
        "DO_PREPROC": True,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "OUTPUT_EMPTY_CLASSES": False,
    }
    dataset = PigMot(config)
    evaluator = Evaluator({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": False,
        "RETURN_ON_ERROR": True,
        "LOG_ON_ERROR": False,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": False,
        "DISPLAY_LESS_PROGRESS": True,
        "TIME_PROGRESS": False,
    })
    metrics_list = [
        M.HOTA(), M.CLEAR(), M.Identity(),
    ]
    raw_results, messages = evaluator.evaluate([dataset], metrics_list)
    # extract combined (averaged over sequences) results
    # raw_results: {dataset: {tracker: {seq_or_COMBINED_SEQ: {class: {metric: value}}}}}
    def scal(v):
        import numpy as np
        a = np.asarray(v).reshape(-1)
        return a[0]

    out = {}
    for tracker, tdata in raw_results["PigMot"].items():
        pig = tdata["COMBINED_SEQ"]["pig"]
        h = pig["HOTA"]; c = pig["CLEAR"]; i = pig["Identity"]
        out[tracker] = {
            "HOTA": round(float(scal(h["HOTA"])), 4),
            "DetA": round(float(scal(h["DetA"])), 4),
            "AssA": round(float(scal(h["AssA"])), 4),
            "MOTA": round(float(scal(c["MOTA"])), 4),
            "MOTP": round(float(scal(c["MOTP"])), 4),
            "IDF1": round(float(scal(i["IDF1"])), 4),
            "IDSW": int(scal(c["IDSW"])),
            "FP": int(scal(c["CLR_FP"])),
            "FN": int(scal(c["CLR_FN"])),
            "MT": int(scal(c["MT"])),
            "ML": int(scal(c["ML"])),
            "Frag": int(scal(c["Frag"])),
        }
        print(tracker, json.dumps(out[tracker]), flush=True)
    (EXP / "eval_outputs" / "metrics.json").write_text(json.dumps(out, indent=2))
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
