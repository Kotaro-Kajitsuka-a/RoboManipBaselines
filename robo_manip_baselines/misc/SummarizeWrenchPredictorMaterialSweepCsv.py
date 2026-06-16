import argparse
import csv
import os


TARGET_OBJECT_KEYS = [
    "WrenchPredObject0",
    "WrenchPredObject1",
    "WrenchPredObject2",
    "WrenchPredObject3",
]


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "material_sweep_csv",
        type=str,
        help="CSV file saved by EvalWrenchPredictorMaterialSweepDir.py",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="Fx/Fy mean",
        choices=[
            "Fx/Fy mean",
            "Fx",
            "Fy",
            "Fz",
            "Nx",
            "Ny",
            "Nz",
            "Fz only",
            "torque mean",
        ],
        help="metric used for the 4x4 summary table",
    )
    return parser.parse_args()


def load_rows(csv_path, metric):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "actual_object_key": row["actual_object_key"],
                    "material_object_key": row["material_object_key"],
                    "value": float(row[metric]),
                }
            )

    assert len(rows) > 0, csv_path
    return rows


def build_table(rows):
    table = {
        actual_object_key: {material_object_key: None for material_object_key in TARGET_OBJECT_KEYS}
        for actual_object_key in TARGET_OBJECT_KEYS
    }

    for row in rows:
        actual_object_key = row["actual_object_key"]
        material_object_key = row["material_object_key"]
        if actual_object_key not in TARGET_OBJECT_KEYS:
            continue
        if material_object_key not in TARGET_OBJECT_KEYS:
            continue

        assert table[actual_object_key][material_object_key] is None, row
        table[actual_object_key][material_object_key] = row["value"]

    for actual_object_key in TARGET_OBJECT_KEYS:
        for material_object_key in TARGET_OBJECT_KEYS:
            assert table[actual_object_key][material_object_key] is not None, (
                actual_object_key,
                material_object_key,
            )

    return table


def build_delta_table(table):
    delta_table = {}
    for actual_object_key in TARGET_OBJECT_KEYS:
        correct_value = table[actual_object_key][actual_object_key]
        delta_table[actual_object_key] = {
            material_object_key: table[actual_object_key][material_object_key]
            - correct_value
            for material_object_key in TARGET_OBJECT_KEYS
        }

    return delta_table


def save_table(table, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/material", *TARGET_OBJECT_KEYS])
        for actual_object_key in TARGET_OBJECT_KEYS:
            writer.writerow(
                [
                    actual_object_key,
                    *[
                        f"{table[actual_object_key][material_object_key]:.10g}"
                        for material_object_key in TARGET_OBJECT_KEYS
                    ],
                ]
            )


def print_table(title, table):
    print(f"[{title}]")
    print(",".join(["actual/material", *TARGET_OBJECT_KEYS]))
    for actual_object_key in TARGET_OBJECT_KEYS:
        values = [
            f"{table[actual_object_key][material_object_key]:.6f}"
            for material_object_key in TARGET_OBJECT_KEYS
        ]
        print(",".join([actual_object_key, *values]))


def summarize_material_sweep_csv(material_sweep_csv, metric="Fx/Fy mean"):
    rows = load_rows(material_sweep_csv, metric)
    table = build_table(rows)
    delta_table = build_delta_table(table)

    csv_root, _csv_ext = os.path.splitext(material_sweep_csv)
    metric_name = metric.replace("/", "").replace(" ", "_")
    raw_output_csv = f"{csv_root}_{metric_name}_table.csv"
    delta_output_csv = f"{csv_root}_{metric_name}_delta_from_correct_table.csv"

    save_table(table, raw_output_csv)
    save_table(delta_table, delta_output_csv)

    print_table(f"{metric} raw table", table)
    print()
    print_table(f"{metric} delta from correct", delta_table)
    print()
    print(f"Save raw table: {raw_output_csv}")
    print(f"Save delta table: {delta_output_csv}")

    return raw_output_csv, delta_output_csv


def main():
    args = parse_argument()
    summarize_material_sweep_csv(args.material_sweep_csv, args.metric)


if __name__ == "__main__":
    main()
