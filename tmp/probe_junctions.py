"""
Quick probe to find the busiest TLS intersections in the SUMO network.
Runs 500 steps headless and ranks each TLS by total vehicle presence.

Usage:
    python -m tmp.probe_junctions
"""

import os
import sys
import traci
from collections import defaultdict

if "SUMO_HOME" not in os.environ:
    sys.exit("Set SUMO_HOME first")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools not in sys.path:
    sys.path.append(tools)

CONFIG = os.path.join("sumo_test", "simple.sumocfg")
STEPS = 500

traci.start(
    [
        "sumo",
        "-c",
        CONFIG,
        "--step-length",
        "1.0",
        "--start",
        "--quit-on-end",
        "--no-warnings",
    ]
)

tls_ids = sorted(traci.trafficlight.getIDList())
all_detectors = traci.inductionloop.getIDList()

# Build per-TLS detector map
tls_detectors = defaultdict(list)
for tls in tls_ids:
    lanes = set(traci.trafficlight.getControlledLanes(tls))
    for det in all_detectors:
        if traci.inductionloop.getLaneID(det) in lanes:
            tls_detectors[tls].append(det)

# Also track vehicles in controlled lanes directly
tls_score = defaultdict(float)

for step in range(STEPS):
    traci.simulationStep()
    for tls in tls_ids:
        lanes = set(traci.trafficlight.getControlledLanes(tls))
        count = 0
        for lane in lanes:
            try:
                count += traci.lane.getLastStepVehicleNumber(lane)
            except Exception:
                pass
        tls_score[tls] += count

traci.close()

# Rank intersections
ranked = sorted(tls_score.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'=' * 55}")
print(f"  TLS Rankings by Vehicle Presence ({STEPS} steps)")
print(f"{'=' * 55}")
print(f"  {'Rank':>4}  {'TLS ID':>15}  {'Detectors':>9}  {'Score':>10}")
print(f"  {'-' * 4}  {'-' * 15}  {'-' * 9}  {'-' * 10}")
for rank, (tls, score) in enumerate(ranked, 1):
    idx = tls_ids.index(tls)
    n_det = len(tls_detectors[tls])
    print(f"  {rank:>4}  {tls:>15}  {n_det:>9}  {score:>10.0f}    [index {idx}]")

print(
    f"\n  Busiest junction : '{ranked[0][0]}'  (tls_ids[{tls_ids.index(ranked[0][0])}])"
)
print(f"{'=' * 55}\n")
