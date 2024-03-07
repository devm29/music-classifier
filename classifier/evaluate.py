"""
Reproducible benchmark of the classifiers this project compares.

The original README published an accuracy table with no code that regenerates
it. That is the part worth fixing before anything else: an unreproducible
number in a machine-learning project is not a result, it is a claim. Running
this module prints the table, and the table in the README is its output.

Protocol: stratified 5-fold cross-validation over the whole feature matrix,
seeded. The original selected a random subset per class with an unseeded
`random.choice`, so its numbers could not be reproduced even by itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from classifier.data import GENRES, WEB_APP_GENRES, load_features, subset
from classifier.models import RANDOM_STATE, candidates


@dataclass
class Result:
    name: str
    train: float
    test: float
    std: float

    @property
    def overfit_gap(self) -> float:
        return self.train - self.test


def evaluate(X: np.ndarray, y: np.ndarray, *, scaled: bool = True, folds: int = 5) -> list[Result]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, pipe in candidates(scaled=scaled).items():
        scores = cross_validate(pipe, X, y, cv=cv, return_train_score=True, n_jobs=-1)
        results.append(
            Result(
                name=name,
                train=float(scores["train_score"].mean()),
                test=float(scores["test_score"].mean()),
                std=float(scores["test_score"].std()),
            )
        )
    return sorted(results, key=lambda r: -r.test)


def print_table(results: list[Result], title: str, baseline: float) -> None:
    print(f"\n{title}")
    print(f"{'classifier':<22}{'train':>8}{'test':>8}{'sd':>7}{'gap':>7}")
    print("-" * 52)
    for r in results:
        print(f"{r.name:<22}{r.train:>8.3f}{r.test:>8.3f}{r.std:>7.3f}{r.overfit_gap:>7.3f}")
    print(f"{'random baseline':<22}{'':>8}{baseline:>8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--raw", action="store_true",
                        help="also evaluate without feature scaling, as the original did")
    args = parser.parse_args()

    X, y = load_features()
    print(f"features: {X.shape[0]} tracks x {X.shape[1]} features, {len(GENRES)} genres")
    print(f"feature std spread: {X.std(0).min():.2f} to {X.std(0).max():.2f} "
          "(this is why scaling matters)")

    ten = evaluate(X, y, scaled=True, folds=args.folds)
    print_table(ten, f"All {len(GENRES)} genres, standardised", 1 / len(GENRES))

    if args.raw:
        print_table(
            evaluate(X, y, scaled=False, folds=args.folds),
            f"All {len(GENRES)} genres, raw features (the original setup)",
            1 / len(GENRES),
        )

    Xs, ys = subset(X, y, WEB_APP_GENRES)
    six = evaluate(Xs, ys, scaled=True, folds=args.folds)
    print_table(six, f"The {len(WEB_APP_GENRES)} web-app genres, standardised",
                1 / len(WEB_APP_GENRES))

    print(f"\nbest overall: {ten[0].name} at {ten[0].test:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
