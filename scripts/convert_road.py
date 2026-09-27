from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ROAD SocketCAN logs into labelled CSV files."
    )
    parser.add_argument(
        "--road-root",
        required=True,
        help="ROAD directory containing ambient/, attacks/, etc."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for converted CSV files."
    )
    return parser.parse_args()


def load_metadata(attacks_dir: Path):
    metadata_path = attacks_dir / "capture_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_can_line(line: str):
    """
    Parse ROAD SocketCAN lines such as:

    (1364236952.554636) vcan0 0B4#00000813000013FC
    """

    line = line.strip()

    if not line or line.startswith("#"):
        return None

    if not line.startswith("("):
        return None

    try:
        timestamp_end = line.index(")")
        timestamp = float(line[1:timestamp_end])

        remainder = line[timestamp_end + 1:].strip()

        parts = remainder.split()
        if len(parts) < 2:
            return None

        can_frame = parts[-1]

        if "#" not in can_frame:
            return None

        can_id, payload = can_frame.split("#", 1)

        # ROAD can contain extended IDs, so preserve the ID string.
        can_id = can_id.upper()

        # Empty payload is allowed.
        payload = payload.strip()

        if len(payload) % 2 != 0:
            return None

        dlc = len(payload) // 2

        return timestamp, can_id, dlc

    except (ValueError, IndexError):
        return None


def read_log(log_path: Path):
    """
    Read a ROAD .log file.

    Returns:
        list of (absolute_timestamp, CAN_ID, DLC)
    """

    rows = []

    with log_path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:
        for line in f:
            parsed = parse_can_line(line)

            if parsed is not None:
                rows.append(parsed)

    return rows


def convert_capture(
    log_path: Path,
    output_path: Path,
    attack_interval=None,
):
    rows = read_log(log_path)

    if not rows:
        print(f"WARNING: no CAN frames found in {log_path.name}")
        return 0, 0

    first_timestamp = rows[0][0]

    output_rows = []
    attack_count = 0

    for timestamp, can_id, dlc in rows:

        relative_time = timestamp - first_timestamp

        label = 0

        if attack_interval is not None:
            start, end = attack_interval

            if start <= relative_time <= end:
                label = 1
                attack_count += 1

        output_rows.append(
            [
                timestamp,
                can_id,
                dlc,
                label,
            ]
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Timestamp",
                "ID",
                "DLC",
                "Label",
            ]
        )

        writer.writerows(output_rows)

    attack_rate = attack_count / len(output_rows)

    print(
        f"{log_path.name}: "
        f"{len(output_rows)} messages, "
        f"attack rate {attack_rate:.4f}"
    )

    return len(output_rows), attack_count


def main():

    args = parse_args()

    road_root = Path(args.road_root)
    output_root = Path(args.out)

    ambient_dir = road_root / "ambient"
    attacks_dir = road_root / "attacks"

    if not ambient_dir.exists():
        raise FileNotFoundError(
            f"Missing directory: {ambient_dir}"
        )

    if not attacks_dir.exists():
        raise FileNotFoundError(
            f"Missing directory: {attacks_dir}"
        )

    metadata = load_metadata(attacks_dir)

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    total_messages = 0
    total_attacks = 0

    print("=" * 70)
    print("ROAD DATASET CONVERSION")
    print("=" * 70)

    # ------------------------------------------------------------
    # AMBIENT CAPTURES
    # ------------------------------------------------------------

    print("\nAMBIENT CAPTURES")

    ambient_files = sorted(
        ambient_dir.glob("*.log")
    )

    for log_path in ambient_files:

        output_path = (
            output_root /
            f"{log_path.stem}.csv"
        )

        messages, attacks = convert_capture(
            log_path,
            output_path,
            attack_interval=None,
        )

        total_messages += messages
        total_attacks += attacks

    # ------------------------------------------------------------
    # ATTACK CAPTURES
    # ------------------------------------------------------------

    print("\nATTACK CAPTURES")

    attack_files = sorted(
        attacks_dir.glob("*.log")
    )

    skipped = []

    for log_path in attack_files:

        capture_name = log_path.stem

        info = metadata.get(capture_name)

        if info is None:
            print(
                f"WARNING: no metadata for "
                f"{capture_name}; SKIPPING"
            )
            skipped.append(capture_name)
            continue

        injection_interval = info.get(
            "injection_interval"
        )

        if injection_interval is None:
            print(
                f"SKIP: {capture_name}.log "
                f"(no CAN-level injection_interval)"
            )
            skipped.append(capture_name)
            continue

        if (
            not isinstance(injection_interval, list)
            or len(injection_interval) != 2
        ):
            print(
                f"WARNING: invalid injection_interval "
                f"for {capture_name}; SKIPPING"
            )
            skipped.append(capture_name)
            continue

        output_path = (
            output_root /
            f"{log_path.stem}.csv"
        )

        messages, attacks = convert_capture(
            log_path,
            output_path,
            attack_interval=injection_interval,
        )

        total_messages += messages
        total_attacks += attacks

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)

    print(f"Total messages: {total_messages}")
    print(f"Total attack-labelled messages: {total_attacks}")

    if total_messages:
        print(
            f"Overall attack rate: "
            f"{total_attacks / total_messages:.4f}"
        )

    print(f"Output directory: {output_root}")

    print("\nSkipped captures:")

    if skipped:
        for name in skipped:
            print(f"  - {name}")
    else:
        print("  None")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()