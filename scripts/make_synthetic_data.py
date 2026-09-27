"""Generate small SYNTHETIC CAN captures so the pipeline can be smoke-tested.

    python scripts/make_synthetic_data.py --out data/raw

READ THIS BEFORE USING THE OUTPUT
---------------------------------
This data is fake. It is a crude simulation of periodic ECU traffic with
injected flooding, fuzzing and spoofing, written in the file formats of
Car-Hacking and CAN-IDS so the adapters can be exercised.

Any number produced from it is a smoke-test result, not a research
result. Do not put it in a paper, a table, or a thesis. Its only purpose
is to let you confirm that the code runs before you download the real
datasets:

    Car-Hacking / Attack & Defense Challenge
        https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
    CAN-IDS (OTIDS)
        https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset
    ORNL ROAD
        https://0xsam.com/road/
    Survival Analysis Dataset
        https://ocslab.hksecurity.net/Datasets/survival-ids

The two synthetic "vehicles" deliberately use different ID sets and base
cycle times, so cross-dataset transfer between them is a meaningful
plumbing check rather than a copy of the same distribution.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def simulate_stream(
    n_messages: int,
    ecu_ids,
    periods_ms,
    attack_plan,
    jitter: float = 0.12,
    seed: int = 0,
):
    """Periodic ECU traffic plus injected attack bursts."""
    rng = np.random.default_rng(seed)

    # Baseline: each ECU emits its ID on its own cycle, with jitter.
    #
    # Every ECU must cover the SAME time span, otherwise the fast cycles
    # exhaust early, the message rate decays along the capture, and a
    # time fraction no longer corresponds to the same index fraction.
    # That in turn makes attack bursts land in an unintended part of the
    # stream once the loader takes a contiguous block.
    rate_hz = sum(1000.0 / p for p in periods_ms)
    span = n_messages / rate_hz * 1.15
    events = []
    for can_id, period in zip(ecu_ids, periods_ms):
        period_s = period / 1000.0
        count = int(span / period_s)
        times = np.cumsum(rng.normal(period_s, period_s * jitter, size=count))
        times = np.clip(times, 1e-6, None)
        dlc = int(rng.integers(4, 9))
        for t in times:
            events.append((t, can_id, dlc, 0))

    events.sort(key=lambda e: e[0])
    events = events[:n_messages]
    if not events:
        raise ValueError("no baseline events generated")
    duration = events[-1][0]

    # Attack bursts injected into the same time axis. Volume is expressed
    # as a fraction of the baseline message count so the resulting attack
    # rate stays in a plausible range regardless of capture length.
    for kind, start_frac, length_frac, volume_frac in attack_plan:
        start = duration * start_frac
        stop = start + duration * length_frac
        n_inject = max(1, int(n_messages * volume_frac))
        times = np.sort(rng.uniform(start, stop, size=n_inject))

        if kind == "flood":
            ids = np.zeros(n_inject, dtype=np.int64)           # ID 0x000 flood
        elif kind == "fuzz":
            ids = rng.integers(0, 0x7FF, size=n_inject)        # random IDs
        else:                                                   # spoof
            target = ecu_ids[int(rng.integers(len(ecu_ids)))]
            ids = np.full(n_inject, target, dtype=np.int64)

        for t, can_id in zip(times, ids):
            events.append((float(t), int(can_id), 8, 1))

    events.sort(key=lambda e: e[0])
    base_epoch = 1478198000.0
    return [(base_epoch + t, cid, dlc, label) for t, cid, dlc, label in events]


def write_car_hacking(events, path, rng):
    """Headerless Car-Hacking format: ts, ID(hex), DLC, data bytes..., flag."""
    with open(path, "w", encoding="utf-8") as fh:
        for ts, can_id, dlc, label in events:
            data = " ".join(f"{int(b):02x}" for b in rng.integers(0, 256, size=dlc))
            fields = [f"{ts:.6f}", f"{can_id:04x}", str(dlc)]
            fields += data.split()
            fields.append("T" if label else "R")
            fh.write(",".join(fields) + "\n")


def write_can_ids_csv(events, path):
    """CAN-IDS as a labelled CSV, which the adapter reads via its CSV path.

    The real OTIDS distribution is an unlabelled text log whose attack
    type lives in the file name, and the adapter handles that too (see
    write_can_ids_log). The synthetic version is written WITH per-frame
    labels on purpose: file-name labelling marks an entire capture as
    attack, which would make the smoke test measure capture identity
    rather than intrusion detection.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Timestamp,ID,DLC,Label\n")
        for ts, can_id, dlc, label in events:
            fh.write(f"{ts:.6f},{can_id:04x},{dlc},{'Attack' if label else 'Normal'}\n")


def write_can_ids_log(events, path, rng):
    """OTIDS-style text log. No per-line label -- the file name carries it."""
    with open(path, "w", encoding="utf-8") as fh:
        for ts, can_id, dlc, _ in events:
            data = " ".join(f"{int(b):02x}" for b in rng.integers(0, 256, size=dlc))
            fh.write(f"Timestamp: {ts:.6f}        ID: {can_id:04x}    000    "
                     f"DLC: {dlc}    {data}\n")


def _spread(kind, n_bursts, length_frac, volume_frac):
    """Short attack bursts spread evenly over the whole capture.

    Real captures often hold one long attack window, but for a smoke-test
    fixture that is a trap: any chronological split then puts whole
    classes on one side of the cut. Several short bursts keep both
    classes present in every contiguous block of the stream, which is
    what the leakage-safe split needs in order to be exercised at all.
    """
    per = volume_frac / n_bursts
    return [(kind, 0.06 + i * (0.88 / n_bursts), length_frac, per)
            for i in range(n_bursts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--messages", type=int, default=60000,
                        help="baseline messages per capture")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- synthetic "vehicle A" -> Car-Hacking format ------------------
    ch_dir = os.path.join(args.out, "car_hacking")
    os.makedirs(ch_dir, exist_ok=True)
    vehicle_a_ids = [0x0316, 0x018F, 0x0260, 0x02A0, 0x0329, 0x0545, 0x04B1]
    vehicle_a_periods = [10, 10, 20, 20, 50, 100, 200]

    for name, plan, seed in [
        ("normal_run", [], args.seed + 1),
        ("dos_dataset", _spread("flood", 16, 0.012, 0.032), args.seed + 2),
        ("fuzzy_dataset", _spread("fuzz", 16, 0.014, 0.026), args.seed + 3),
        ("spoofing_dataset", _spread("spoof", 16, 0.012, 0.022), args.seed + 4),
    ]:
        events = simulate_stream(args.messages, vehicle_a_ids, vehicle_a_periods,
                                 plan, seed=seed)
        write_car_hacking(events, os.path.join(ch_dir, f"{name}.csv"), rng)
        attack_rate = np.mean([e[3] for e in events])
        print(f"  car_hacking/{name}.csv  {len(events):>7,} msgs  "
              f"attack rate {attack_rate:.3f}")

    # --- synthetic "vehicle B" -> CAN-IDS log format ------------------
    # Different IDs and faster cycles, so transfer between A and B is a
    # real distribution shift rather than a relabelled copy.
    ci_dir = os.path.join(args.out, "can_ids")
    os.makedirs(ci_dir, exist_ok=True)
    vehicle_b_ids = [0x0153, 0x0164, 0x0220, 0x02C0, 0x0350, 0x05A1]
    vehicle_b_periods = [7, 7, 15, 25, 40, 120]

    for name, plan, seed in [
        ("Attack_free_dataset", [], args.seed + 11),
        ("DoS_dataset", _spread("flood", 16, 0.012, 0.034), args.seed + 12),
        ("Fuzzy_dataset", _spread("fuzz", 16, 0.014, 0.026), args.seed + 13),
        ("Impersonation_dataset", _spread("spoof", 16, 0.012, 0.022), args.seed + 14),
    ]:
        events = simulate_stream(args.messages, vehicle_b_ids, vehicle_b_periods,
                                 plan, seed=seed)
        write_can_ids_csv(events, os.path.join(ci_dir, f"{name}.csv"))
        attack_rate = np.mean([e[3] for e in events])
        print(f"  can_ids/{name}.csv  {len(events):>7,} msgs  "
              f"attack rate {attack_rate:.3f}")

    # One OTIDS-format text log too, so the log parser is exercised. It is
    # written to a sibling directory so it does not mix label provenances
    # inside the can_ids dataset.
    log_dir = os.path.join(args.out, "can_ids_logformat")
    os.makedirs(log_dir, exist_ok=True)
    events = simulate_stream(args.messages // 2, vehicle_b_ids, vehicle_b_periods,
                             _spread("flood", 16, 0.012, 0.034), seed=args.seed + 21)
    write_can_ids_log(events, os.path.join(log_dir, "DoS_dataset.txt"), rng)
    print(f"  can_ids_logformat/DoS_dataset.txt  {len(events):>7,} msgs "
          f"(unlabelled log; label comes from the file name)")

    print("\nSynthetic data written to", os.path.abspath(args.out))
    print("This data is FAKE. Use it only to verify that the pipeline runs.")
    print("Results from it must never be reported. Download the real datasets")
    print("before producing anything for the thesis.")


if __name__ == "__main__":
    main()
