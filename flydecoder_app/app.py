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
            print("Схема каналів змінилася; створюю новий зовнішній зчитувач.", flush=True)
            model = ActivityReadout(len(ACTIVITY_NAMES), seed=17)
        else:
            print(f"Завантажено ваги зчитувача: {model.updates} оновлень", flush=True)
    else:
        model = ActivityReadout(len(ACTIVITY_NAMES), seed=17)
        print("Створено чисту модель зовнішнього зчитувача.", flush=True)
    return adapter, model


def _print_blind(metrics: dict) -> None:
    print(f"Сліпих прикладів: {metrics['count']}")
    print(f"Точний перший вибір: {metrics['top1_accuracy']:.1%}")
    print(f"Точне відновлення з повторами: {metrics['retry_accuracy']:.1%}")
    print("Матриця помилок (рядок — формат прикладу, стовпець — перший вибір):")
    actions = ("base64", "hex", "binary", "url", "skip")
    print("          " + " ".join(f"{action:>7}" for action in actions))
    for codec in actions:
        print(f"{codec:>9} " + " ".join(f"{metrics['confusion'][codec][action]:7d}" for action in actions))
    if metrics["failures"]:
        print("Невдалі рішення:")
        for item in metrics["failures"]:
            print(f"  {item['expected']} → {item['first']}: {item['input']!r} ({item['result']})")


def _parser():
    parser = argparse.ArgumentParser(description="Локальний декодер рядків на симуляції FlyBrain / MaleCNS v1.0")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="файл зовнішніх ваг зчитувача")
    parser.add_argument("--self-check", action="store_true", help="запустити один реальний прохід мозку й декодування")
    parser.add_argument("--blind-eval", type=int, metavar="N", help="сліпо перевірити N нових прикладів")
    parser.add_argument("--train", type=int, metavar="N", help="навчити зчитувач на N нових синтетичних прикладах")
    parser.add_argument("--fresh", action="store_true", help="почати навчання з нульових зовнішніх ваг")
    parser.add_argument("--ui-smoke", type=float, metavar="SECONDS", help="відкрити Pygame у прихованому режимі на тест")
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
            print(f"Симуляція: {time.time() - started:.2f} с · {len(activity)} ознак активності, "
                  f"{nonzero} ненульових")
            print(f"Ознаки: {features.summary}")
            print(f"Муха обрала: {result.first_action}; результат: {result.output!r}; {result.message}")
        if args.train is not None:
            def report(progress):
                if progress.completed % 5 == 0 or progress.completed == 1:
                    print(f"Навчання {progress.completed}/{progress.total}: "
                          f"{progress.recent_accuracy:.1%} правильних винагород за останні приклади", flush=True)
            result = train_readout(adapter, model, args.train, seed=9301,
                                   save_path=args.model, progress=report)
            print(f"Ваги зчитувача збережено: {args.model} ({result.completed} прикладів)")
        if args.blind_eval is not None:
            _print_blind(run_blind_evaluation(adapter, model, count=args.blind_eval))
        if args.ui_smoke is not None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            from .ui import DecoderWindow
            window = DecoderWindow(args.model, headless=True, adapter=adapter, model=model)
            window.status = "Перевірка Pygame UI на CPU"
            window.run(max_seconds=args.ui_smoke)
            print("Pygame-вікно відкрито й завершено в headless-режимі")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FlyDecoder не запустився: {exc}", file=sys.stderr)
        print("Перевірте, що install_flydecoder.bat завершив завантаження MaleCNS та збір кешу.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
