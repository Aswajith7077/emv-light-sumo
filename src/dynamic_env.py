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

class Traffic:

    NUM_ACTIONS = 2
    STATE_SIZE = 9

    def __init__(
        self,
        use_gui=True,
        step_length=0.1,
        delay=100,
        min_green_steps=50,
        max_steps=10000,
    ):
        self.use_gui = use_gui
        self.step_length = step_length
        self.delay = delay
        self.min_green_steps = min_green_steps
        self.max_steps = max_steps

        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sumo_test",
            "simple.sumocfg",
        )

        traci.start(self.build_sumo_command())

        self.tls_ids = sorted(traci.trafficlight.getIDList())
        self.tls_id = self.tls_ids[0]
        self.all_detectors = traci.inductionloop.getIDList()
        self.__init_tls_detectors()

        self.step_count = 0
        self.last_switch_step = {
            tls_id: -self.min_green_steps for tls_id in self.tls_ids
        }
        self.prev_total_waiting = 0
        self.prev_vehicle_count = 0

    def __init_tls_detectors(self):

        self.tls_detectors = defaultdict(list)

        for tls in self.tls_ids:
            lanes = list(set(traci.trafficlight.getControlledLanes(tls)))

            for det in self.all_detectors:
                lane = traci.inductionloop.getLaneID(det)
                if lane in lanes:
                    self.tls_detectors[tls].append(det)

    def build_sumo_command(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c",
            self.config_path,
            "--step-length",
            str(self.step_length),
            "--start",
            "--quit-on-end",
        ]
        if self.use_gui:
            cmd += ["--delay", str(self.delay)]
        return cmd

    def reset(self):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        traci.start(self.build_sumo_command())
        self.step_count = 0
        self.last_switch_step = -self.min_green_steps
        self.prev_total_waiting = 0
        self.prev_vehicle_count = 0

        return self.get_state()

    def get_state(self):

        queue_lengths = []
        for det_id in self.tls_detectors[self.tls_id]:
            try:
                q = traci.lanearea.getLastStepVehicleNumber(det_id)
            except traci.exceptions.TraCIException:
                q = 0
            queue_lengths.append(q)

        current_phase = traci.trafficlight.getPhase(self.tls_id)
        state = tuple(queue_lengths) + (current_phase,)
        return state

    def get_state_array(self):
        return np.array(self.get_state(), dtype=np.float32)

    def action(self, action):
        if action == 1:
            if self.step_count - self.last_switch_step >= self.min_green_steps:
                program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                num_phases = len(program.phases)
                current = traci.trafficlight.getPhase(self.tls_id)
                next_phase = (current + 1) % num_phases
                traci.trafficlight.setPhase(self.tls_id, next_phase)
                self.last_switch_step = self.step_count


    def reward(self, state):
        queue_lengths = state[:-1]
        total_queue = sum(queue_lengths)

        total_waiting = 0
        for lane_id in traci.trafficlight.getControlledLanes(self.tls_id):
            try:
                total_waiting += traci.lane.getWaitingTime(lane_id)
            except traci.exceptions.TraCIException:
                pass

        queue_penalty = -float(total_queue)
        waiting_delta = -(total_waiting - self.prev_total_waiting) * 0.1
        departed = traci.simulation.getDepartedNumber()
        throughput_bonus = departed * 0.5

        self.prev_total_waiting = total_waiting

        reward = queue_penalty + waiting_delta + throughput_bonus
        return reward

    def step(self, actions):
        self.action(actions)
        traci.simulationStep()
        self.step_count += 1

        new_state = self.get_state()
        reward = self.reward(new_state)

        remaining = traci.simulation.getMinExpectedNumber()
        done = (remaining <= 0) or (self.step_count >= self.max_steps)

        info = {
            "step": self.step_count,
            "sim_time": traci.simulation.getTime(),
            "total_queue": sum(new_state[:-1]),
            "phase": new_state[-1],
            "reward": reward,
        }

        return new_state, reward, done, info

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    @property
    def total_steps(self):
        return self.step_count
