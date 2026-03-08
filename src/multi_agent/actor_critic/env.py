"""
Multi-Agent SUMO Traffic Environment.

Manages *all* traffic-light intersections discovered in the loaded SUMO
network.  Each TLS gets its own local state/reward, while a single TraCI
connection drives the shared simulation.
"""

import os
import sys
import numpy as np
import traci
from collections import defaultdict


if "SUMO_HOME" not in os.environ:
    sys.exit("Environment variable SUMO_HOME not declared")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")

if tools not in sys.path:
    sys.path.append(tools)


class MultiAgentTraffic:
    """
    Multi-agent wrapper around a SUMO network.

    * Auto-discovers every traffic-light (TLS) in the simulation.
    * Builds a per-TLS detector map for queue-length observations.
    * Exposes dict-based reset / step API keyed by TLS ID.
    """

    NUM_ACTIONS = 2  # keep phase / switch phase

    def __init__(
        self,
        use_gui: bool = True,
        step_length: float = 1.0,
        delay: int = 0,
        min_green_steps: int = 5,
        max_steps: int = 5000,
    ):
        self.use_gui = use_gui
        self.step_length = step_length
        self.delay = delay
        self.min_green_steps = min_green_steps
        self.max_steps = max_steps

        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "sumo_test",
            "simple.sumocfg",
        )

        # Start SUMO and discover intersections
        traci.start(self._build_sumo_command())

        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.all_detectors = traci.inductionloop.getIDList()
        self._init_tls_detectors()

        # Determine per-TLS state size (num_detectors + 1 for phase)
        self.state_sizes: dict[str, int] = {}
        for tid in self.tls_ids:
            n_det = len(self.tls_detectors[tid])
            self.state_sizes[tid] = n_det + 1  # detectors + phase

        self.step_count = 0
        self.last_switch_step: dict[str, int] = {
            tid: -self.min_green_steps for tid in self.tls_ids
        }
        self.prev_total_waiting: dict[str, float] = {tid: 0.0 for tid in self.tls_ids}

    # ── Initialisation helpers ────────────────────────────────────────────

    def _init_tls_detectors(self):
        """Map each TLS to its induction-loop detectors."""
        self.tls_detectors: dict[str, list] = defaultdict(list)

        for tls in self.tls_ids:
            lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
            for det in self.all_detectors:
                lane = traci.inductionloop.getLaneID(det)
                if lane in lanes:
                    self.tls_detectors[tls].append(det)

    def _build_sumo_command(self) -> list:
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c",
            self.config_path,
            "--step-length",
            str(self.step_length),
            "--start",
            "--quit-on-end",
            "--no-step-log",
            "--verbose",
            "false",
        ]
        if self.use_gui:
            cmd += ["--delay", str(self.delay)]
        return cmd

    # ── State ─────────────────────────────────────────────────────────────

    def get_state(self, tls_id: str) -> tuple:
        """Return (queue_0, queue_1, …, current_phase) for one TLS."""
        queue_lengths = []
        for det_id in self.tls_detectors[tls_id]:
            try:
                q = traci.inductionloop.getLastStepVehicleNumber(det_id)
            except traci.exceptions.TraCIException:
                q = 0
            queue_lengths.append(q)

        current_phase = traci.trafficlight.getPhase(tls_id)
        return tuple(queue_lengths) + (current_phase,)

    def get_all_states(self) -> dict:
        """Return states for every TLS."""
        return {tid: self.get_state(tid) for tid in self.tls_ids}

    # ── Action ────────────────────────────────────────────────────────────

    def apply_action(self, tls_id: str, action: int):
        """Apply keep(0) / switch(1) to a single TLS."""
        if action == 1:
            if self.step_count - self.last_switch_step[tls_id] >= self.min_green_steps:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                current = traci.trafficlight.getPhase(tls_id)
                next_phase = (current + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
                self.last_switch_step[tls_id] = self.step_count

    # ── Reward ────────────────────────────────────────────────────────────

    def compute_reward(self, tls_id: str, state: tuple) -> float:
        """Per-intersection reward: queue penalty + waiting delta + throughput."""
        queue_lengths = state[:-1]
        total_queue = sum(queue_lengths)

        total_waiting = 0.0
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
        for lane_id in controlled_lanes:
            try:
                total_waiting += traci.lane.getWaitingTime(lane_id)
            except traci.exceptions.TraCIException:
                pass

        queue_penalty = -float(total_queue)
        waiting_delta = -(total_waiting - self.prev_total_waiting[tls_id]) * 0.1

        # Throughput bonus is global (departed vehicles) divided among agents
        departed = traci.simulation.getDepartedNumber()
        throughput_bonus = (departed / max(len(self.tls_ids), 1)) * 0.5

        self.prev_total_waiting[tls_id] = total_waiting

        return queue_penalty + waiting_delta + throughput_bonus

    # ── Episode API ───────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Restart the SUMO simulation and return initial states for all TLS."""
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        traci.start(self._build_sumo_command())
        self.step_count = 0
        self.last_switch_step = {tid: -self.min_green_steps for tid in self.tls_ids}
        self.prev_total_waiting = {tid: 0.0 for tid in self.tls_ids}

        return self.get_all_states()

    def step(self, actions: dict) -> tuple:
        """
        Apply all agent actions, advance one sim step, compute rewards.

        Parameters
        ----------
        actions : Dict[tls_id, int]
            Action per intersection (0 = keep, 1 = switch).

        Returns
        -------
        next_states : Dict[tls_id, tuple]
        rewards     : Dict[tls_id, float]
        done        : bool
        infos       : Dict[tls_id, dict]
        """
        # Apply all actions before stepping the simulation
        for tid, action in actions.items():
            self.apply_action(tid, action)

        traci.simulationStep()
        self.step_count += 1

        # Collect per-agent observations
        next_states = {}
        rewards = {}
        infos = {}

        for tid in self.tls_ids:
            state = self.get_state(tid)
            reward = self.compute_reward(tid, state)
            next_states[tid] = state
            rewards[tid] = reward
            infos[tid] = {
                "step": self.step_count,
                "sim_time": traci.simulation.getTime(),
                "total_queue": sum(state[:-1]),
                "phase": state[-1],
                "reward": reward,
            }

        remaining = traci.simulation.getMinExpectedNumber()
        done = (remaining <= 0) or (self.step_count >= self.max_steps)

        return next_states, rewards, done, infos

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    @property
    def total_steps(self):
        return self.step_count
