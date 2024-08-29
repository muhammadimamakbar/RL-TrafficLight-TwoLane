import os
import sys

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


class Trafic_light:
    def __init__(self, junction, duration_all_red=5, duration_yellow_red=3, duration_max_phase=180):
        self.junction = junction
        self.lanes = traci.trafficlight.getControlledLanes(junction)
        self.lanes_number = len(self.lanes)
        self.duration_all_red = duration_all_red
        self.duration_yellow_red = duration_yellow_red
        self.duration_max_phase = duration_max_phase
        self.curr_duration_phase = 0
        self.total_duration_phase = 0
        self.duration_green_red_fixed = int(((duration_max_phase - (self.lanes_number*(duration_all_red+duration_yellow_red)))/self.lanes_number))
        self.duration_green_red_dinamic = [0] * self.lanes_number

    def setPhaseTimeBased(self):
        self.total_duration_phase = (self.duration_all_red+self.duration_yellow_red+self.duration_green_red_fixed) * self.lanes_number

    def getTimeChangeAllRedFixed(self):
        change_on = []
        if self.duration_all_red != 0:
            change_on.append(0)
            change_on.append(self.duration_all_red+self.duration_yellow_red+self.duration_green_red_fixed)
        return change_on
    def getTimeChangeYellRedFixed(self):
        change_on = []
        if self.duration_yellow_red != 0:
            change_on.append(self.duration_all_red)
            change_on.append(self.duration_all_red+self.duration_yellow_red+self.duration_green_red_fixed+self.duration_all_red)
        return change_on
    def getTimeChangeGreenRedFixed(self):
        change_on = []
        if self.duration_green_red_fixed != 0:
            change_on.append(self.duration_all_red)
            change_on.append(self.duration_all_red+self.duration_yellow_red+self.duration_green_red_fixed+self.duration_all_red+self.duration_yellow_red)
        return change_on

    def statusLight(self):
        return traci.trafficlight.getRedYellowGreenState(self.junction)

    # menghitung jumlah kendaraan untuk setiap kaki simpang
    def totalVehiclePerLane(self):
        vehicle_per_lane = dict()
        for l in self.lanes:
            vehicle_per_lane[l] = 0
            for k in traci.lane.getLastStepVehicleIDs(l):
                if traci.vehicle.getLanePosition(k) > 10:
                    vehicle_per_lane[l] += 1
        return vehicle_per_lane

    # menghitung total waiting time untuk setiap kaki simpang
    def totalWaitingTimePerlane(self):
        waiting_time_per_lane = dict()
        for lane in self.lanes:
            waiting_time_per_lane[lane] = traci.lane.getWaitingTime(lane)
        return waiting_time_per_lane

    # menghitung rata2 kecepatan untuk setiap kaki simpang
    def avgSpeedPerLane(self):
        avg_per_lane = dict()
        for l in self.lanes:
            in_lane = list()
            for k in traci.lane.getLastStepVehicleIDs(l):
                if traci.vehicle.getLanePosition(k) > 10:
                    in_lane.append(traci.vehicle.getSpeed(k))
            avg_per_lane[l] = sum(in_lane) / len(in_lane) if len(in_lane) > 0  else 0
        return avg_per_lane

    # menghitung jumlah kendaraan untuk setiap kaki simpang
    def totalVehicleJunction(self):
        vehicle = 0
        for l in self.lanes:
            for k in traci.lane.getLastStepVehicleIDs(l):
                if traci.vehicle.getLanePosition(k) > 10:
                    vehicle += 1
        return vehicle

    # jumlah waiting time seluruh kaki simpang
    def totalWaitingTimeJunction(self):
        waiting_time = 0
        for lane in self.lanes:
            waiting_time += traci.lane.getWaitingTime(lane)
        return waiting_time

    # menghitung rata2 kecepatan untuk setiap kaki simpang
    def avgSpeedJunction(self):
        avg = 0
        in_lane = list()
        for l in self.lanes:
            for k in traci.lane.getLastStepVehicleIDs(l):
                if traci.vehicle.getLanePosition(k) > 10:
                    in_lane.append(traci.vehicle.getSpeed(k))
        avg = sum(in_lane) / len(in_lane) if len(in_lane) > 0  else 0
        return avg

    # set lampu dan durasi nya untuk setiap phase
    def phaseDuration(self, phase_time, phase_state):
        traci.trafficlight.setRedYellowGreenState(self.junction, phase_state)
        traci.trafficlight.setPhaseDuration(self.junction, phase_time)

