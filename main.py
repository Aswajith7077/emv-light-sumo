import traci
import sys
import os
import time

SUMO_CFG = "sumo_test/test.sumocfg"
USE_GUI = True

if USE_GUI:
    sumoBinary = "sumo-gui"
else:
    sumoBinary = "sumo"

sumoCmd = [sumoBinary, "-c", SUMO_CFG, "--start"]

traci.start(sumoCmd)



print("Simulation started")

# Get all traffic lights
tls_ids = traci.trafficlight.getIDList()
print("Traffic Lights:", tls_ids)
