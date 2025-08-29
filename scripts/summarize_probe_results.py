import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
from statistics import mean


ACCURACY_REGEX = re.compile(r"Layer\s+(?P<layer>\d+)\s+test accuracy:\s+(?P<acc>[0-9]*\.?[0-9]+)")


GROUP_GLOBS = {
    "trained_large": "logs/test_trained_models_*.err",
    "random_large": "logs/test_random_models_*.err",
    "trained_small": "logs/test_trained_small_*.err",
    "random_small": "logs/test_random_small_*.err",
}


def parse_file_for_accuracies(filepath: str) -> list[tuple[int, float]]:
    results: list[tuple[int, float]] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ACCURACY_REGEX.search(line)
                if m:
                    layer = int(m.group("layer"))
                    acc = float(m.group("acc"))
                    results.append((layer, acc))
    except FileNotFoundError:
        return []
    return results


def collect_group_results() -> dict:
    group_to_records: dict[str, list[dict]] = {g: [] for g in GROUP_GLOBS}

    for group, pattern in GROUP_GLOBS.items():
        for filepath in sorted(glob.glob(pattern)):
            file_records = parse_file_for_accuracies(filepath)
            for layer, acc in file_records:
                group_to_records[group].append(
                    {
                        "group": group,
                        "file": os.path.basename(filepath),
                        "layer": layer,
                        "accuracy": acc,
                    }
                )

    return group_to_records


def compute_group_stats(group_to_records: dict) -> dict:
    stats: dict[str, dict] = {}
    for group, records in group_to_records.items():
        if not records:
            stats[group] = {
                "count": 0,
                "mean": None,
                "min": None,
                "max": None,
            }
            continue
        values = [r["accuracy"] for r in records]
        stats[group] = {
            "count": len(values),
            "mean": round(mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return stats


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_outputs(group_to_records: dict, stats: dict, out_dir: str) -> dict:
    ensure_dir(out_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(out_dir, f"probe_accuracy_by_group_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "file", "layer", "accuracy"])
        writer.writeheader()
        for group in sorted(group_to_records.keys()):
            for rec in sorted(group_to_records[group], key=lambda r: (r["file"], r["layer"])):
                writer.writerow(rec)

    stats_path = os.path.join(out_dir, f"probe_group_stats_{timestamp}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    md_path = os.path.join(out_dir, f"probe_summary_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Probe Results Summary\n\n")
        f.write(f"Generated: {dt.datetime.now().isoformat()}\n\n")
        f.write("## Group statistics (count, mean, min, max)\n\n")
        f.write("| Group | Count | Mean | Min | Max |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for group in sorted(stats.keys()):
            gs = stats[group]
            if gs["count"] == 0:
                f.write(f"| {group} | 0 | - | - | - |\n")
            else:
                f.write(
                    f"| {group} | {gs['count']} | {gs['mean']:.4f} | {gs['min']:.4f} | {gs['max']:.4f} |\n"
                )

        f.write("\n")
        f.write("Notes:\n")
        f.write("- trained_* groups are probes on trained checkpoints.\n")
        f.write("- random_* groups are probes on properly randomized checkpoints.\n")

    return {
        "csv": csv_path,
        "json": stats_path,
        "md": md_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize probe results from logs/")
    parser.add_argument("--out_dir", default="results", help="Directory to write summaries to")
    args = parser.parse_args()

    group_to_records = collect_group_results()
    stats = compute_group_stats(group_to_records)
    paths = write_outputs(group_to_records, stats, args.out_dir)

    print("Wrote:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()



