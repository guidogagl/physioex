"""Subject-level k-fold cross-validation splits.

The default :meth:`BasePhysioDataset.get_splits` is a seeded random 70/15/15
split, which is what the single-fold examples use. Benchmark protocols from the
literature are k-fold by subject, e.g.

* SeqSleepNet on MASS (Phan et al. 2019): 20-fold CV over 200 subjects,
  180 train / 10 validation / 10 test per fold;
* L-SeqSleepNet on SleepEDF-SC (Phan et al. 2023): leave-one-subject-out CV with
  4 subjects held out for validation, both nights of a subject kept together.

:func:`kfold_subject_splits` produces such partitions and
:func:`assign_kfold_splits` installs them on one or several datasets (e.g. the
five MASS cohorts combined in a :class:`~physioex.data.multi.MultiDataset`) so
that ``dataset.split(fold)`` / ``Trainer.train(..., fold=fold)`` follow the
protocol without subclassing.

Fold construction: subject groups are shuffled once with ``seed`` and cut into
``n_folds`` contiguous chunks; fold ``k`` tests on chunk ``k``, validates on the
next ``n_valid`` groups in cyclic order, and trains on the rest. Test sets are
therefore disjoint across folds and jointly cover every subject exactly once.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

Split = Tuple[List[str], List[str], List[str]]


def _group(ids: Sequence[str], group_fn: Optional[Callable[[str], Hashable]]) -> Dict[Hashable, List[str]]:
    groups: Dict[Hashable, List[str]] = {}
    for sid in ids:
        key = group_fn(sid) if group_fn is not None else sid
        groups.setdefault(key, []).append(sid)
    return groups


def kfold_subject_splits(
    subject_ids: Sequence[str],
    n_folds: int,
    fold: int,
    n_valid: int = 1,
    seed: int = 42,
    group_fn: Optional[Callable[[str], Hashable]] = None,
) -> Split:
    """Return ``(train, valid, test)`` subject ids for ``fold`` of an ``n_folds`` CV.

    Args:
        subject_ids: all subject ids of the dataset(s).
        n_folds: number of folds (``n_folds == n_groups`` gives leave-one-subject-out).
        fold: fold index in ``[0, n_folds)``.
        n_valid: number of *groups* held out for validation (taken from the training part).
        seed: shuffling seed (fixed once for the whole CV, not per fold).
        group_fn: maps a subject id to a group key; all ids of a group stay in the
            same split (e.g. the two nights of one SleepEDF subject).
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    groups = _group(subject_ids, group_fn)
    keys = sorted(groups.keys(), key=str)
    if n_folds > len(keys):
        raise ValueError(f"n_folds={n_folds} exceeds the {len(keys)} subject groups")
    rng = random.Random(seed)
    rng.shuffle(keys)

    # contiguous chunks, sizes differ by at most one (like numpy.array_split)
    q, r = divmod(len(keys), n_folds)
    bounds = [0]
    for k in range(n_folds):
        bounds.append(bounds[-1] + q + (1 if k < r else 0))
    test_keys = keys[bounds[fold] : bounds[fold + 1]]
    # cyclic order starting right after the test chunk
    rest = keys[bounds[fold + 1] :] + keys[: bounds[fold]]
    if n_valid < 0 or n_valid >= len(rest):
        raise ValueError(
            f"n_valid={n_valid} leaves no training subjects ({len(rest)} groups outside the test fold)"
        )
    valid_keys, train_keys = rest[:n_valid], rest[n_valid:]

    expand = lambda ks: [sid for k in ks for sid in groups[k]]  # noqa: E731
    return expand(train_keys), expand(valid_keys), expand(test_keys)


def assign_kfold_splits(
    datasets: Sequence,
    n_folds: int,
    n_valid: int = 1,
    seed: int = 42,
    group_fn: Optional[Callable[[str], Hashable]] = None,
) -> None:
    """Install a joint k-fold split on one or more datasets.

    The partition is computed over the union of the datasets' subjects, so a
    :class:`~physioex.data.multi.MultiDataset` built from ``datasets`` sees a
    single coherent CV (each fold's test subjects come from any cohort). Each
    dataset then gets a split function that restricts the global partition to
    its own subjects; ``dataset.split(fold)`` uses it instead of ``get_splits``.

    Subject ids are namespaced by dataset position to tolerate collisions.
    """
    tagged: List[str] = []
    for i, ds in enumerate(datasets):
        tagged.extend(f"{i}::{spec.subject_id}" for spec in ds._subjects)

    tagged_group_fn = None
    if group_fn is not None:
        tagged_group_fn = lambda t: (t.split("::", 1)[0], group_fn(t.split("::", 1)[1]))  # noqa: E731

    def _restrict(i: int) -> Callable[[int], Split]:
        prefix = f"{i}::"

        def fn(fold: int) -> Split:
            train, valid, test = kfold_subject_splits(
                tagged, n_folds, fold, n_valid=n_valid, seed=seed, group_fn=tagged_group_fn
            )
            pick = lambda part: [t[len(prefix):] for t in part if t.startswith(prefix)]  # noqa: E731
            return pick(train), pick(valid), pick(test)

        return fn

    for i, ds in enumerate(datasets):
        ds.set_split_fn(_restrict(i))
