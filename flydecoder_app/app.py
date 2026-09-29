"""FlyDecoder entry point."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from .brain_adapter import ACTIVITY_NAMES, FlyBrainAdapter
from .features import extract_features
from .readout import ActivityReadout
from .service import decode_with_retry, run_blind_evaluation
from .training import train_readout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "out" / "decoder_reader.npz"


def _engine(model_path: Path, fresh: bool = False):
    os.environ["FLYBRAIN_DEVICE"] = "cpu"
    os.environ.setdefault("FLYBRAIN_DATA", str(ROOT / "data" / "malecns"))
    adapter = FlyBrainAdapter(device="cpu")
    if model_path.exists() and not fresh:
        model = ActivityReadout.load(model_path)
        if model.input_count != len(ACTIVITY_NAMES):
            print("Channel schema changed; creating a new external readout.", flush=True)
            model = ActivityReadout(len(ACTIVITY_NAMES), seed=17)
        else:
            print(f"Loaded readout weights: {model.updates} updates", flush=True)
    else:
        model = ActivityReadout(len(ACTIVITY_NAMES), seed=17)
        print("Created a fresh external readout model.", flush=True)
    return adapter, model


def _print_blind(metrics: dict) -> None:
    print(f"Blind samples: {metrics['count']}")
    print(f"Exact first choice: {metrics['top1_accuracy']:.1%}")
    print(f"Exact recovery with retries: {metrics['retry_accuracy']:.1%}")
    print("Confusion matrix (rows: sample format, columns: first choice):")
    actions = ("base64", "hex", "binary", "url", "skip")
    print("          " + " ".join(f"{action:>7}" for action in actions))
    for codec in actions:
        print(f"{codec:>9} " + " ".join(f"{metrics['confusion'][codec][action]:7d}" for action in actions))
    if metrics["failures"]:
        print("Failed cases:")
        for item in metrics["failures"]:
            print(f"  {item['expected']} → {item['first']}: {item['input']!r} ({item['result']})")


def _parser():
    parser = argparse.ArgumentParser(description="Local string decoder powered by the FlyBrain / MaleCNS v1.0 simulation")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="external readout weights file")
    parser.add_argument("--self-check", action="store_true", help="run one brain simulation pass and decode a sample")
    parser.add_argument("--blind-eval", type=int, metavar="N", help="blind-check N new samples")
    parser.add_argument("--train", type=int, metavar="N", help="train the readout on N new synthetic samples")
    parser.add_argument("--fresh", action="store_true", help="start training with zeroed external weights")
    parser.add_argument("--ui-smoke", type=float, metavar="SECONDS", help="open Pygame in headless mode for a smoke check")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    os.environ["FLYBRAIN_DEVICE"] = "cpu"
    os.environ.setdefault("FLYBRAIN_DATA", str(ROOT / "data" / "malecns"))
    if not any((args.self_check, args.blind_eval is not None, args.train is not None, args.ui_smoke is not None)):
        from .ui import DecoderWindow
        DecoderWindow(args.model).run()
        return 0
    try:
        adapter, model = _engine(args.model, fresh=args.fresh)
        if args.self_check:
            sample = "SGVsbG8gZmx5IQ=="
            features = extract_features(sample)
            started = time.time()
            activity = adapter.activity(features.values)
            result = decode_with_retry(sample, activity, model)
            nonzero = int((activity > 1e-6).sum())
            print(f"Simulation: {time.time() - started:.2f} s · {len(activity)} activity values, "
                  f"{nonzero} nonzero")
            print(f"Features: {features.summary}")
            print(f"The fly chose: {result.first_action}; result: {result.output!r}; {result.message}")
        if args.train is not None:
            def report(progress):
                if progress.completed % 5 == 0 or progress.completed == 1:
                    print(f"Training {progress.completed}/{progress.total}: "
                          f"{progress.recent_accuracy:.1%} correct rewards on recent samples", flush=True)
            result = train_readout(adapter, model, args.train, seed=9301,
                                   save_path=args.model, progress=report)
            print(f"Readout weights saved: {args.model} ({result.completed} samples)")
        if args.blind_eval is not None:
            _print_blind(run_blind_evaluation(adapter, model, count=args.blind_eval))
        if args.ui_smoke is not None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            from .ui import DecoderWindow
            window = DecoderWindow(args.model, headless=True, adapter=adapter, model=model)
            window.status = "Checking the Pygame UI on CPU"
            window.run(max_seconds=args.ui_smoke)
            print("Pygame window opened and closed in headless mode")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FlyDecoder could not start: {exc}", file=sys.stderr)
        print("Check that install_flydecoder.bat finished downloading MaleCNS and building the cache.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
