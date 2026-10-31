"""Subject-grouped splits: reproducible, order independent, grouped by person, JSON round trip."""
import json


def test_grouped_splits_order_independent_and_sizes():
    from physioex.data.splits import grouped_splits

    ids = [f"tp{i:03d}" for i in range(255)]
    a = grouped_splits(ids, "dcsm")
    b = grouped_splits(list(reversed(ids)), "dcsm")
    assert a == b and [len(x) for x in a] == [203, 26, 26]
    assert not (set(a[0]) & set(a[1])) and not (set(a[0]) & set(a[2])) and not (set(a[1]) & set(a[2]))


def test_grouped_by_person_and_matches_build_splits(tmp_path):
    from physioex.data.splits import build_splits, check_no_overlap, grouped_splits, load_site_splits

    ids = [f"shhs{v}-{2000 + i}" for i in range(40) for v in (1, 2)]
    parts = grouped_splits(ids, "shhs", group="shhs")
    people = [{s.split("-")[1] for s in p} for p in parts]
    assert not (people[0] & people[1]) and not (people[0] & people[2]) and not (people[1] & people[2])
    cfgs = [{"name": "shhs1", "site": "shhs", "group": "shhs"}]
    splits, _ = build_splits({"shhs1": ids}, cfgs)
    check_no_overlap(splits, cfgs)
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits))
    assert load_site_splits(str(path), "shhs") == tuple(sorted(p) for p in parts)
