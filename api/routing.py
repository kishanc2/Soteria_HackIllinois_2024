from datetime import datetime
import heapq
import json
import requests
import numpy as np
import random
import openrouteservice


triage_one_ttl = 100
triage_zero_ttl = 50
simulation_radius = 1000

class Patient:
    def __init__(self, call_time, triage_code, location):
        self.call_time = call_time
        self.triage_code = triage_code
        self.location = location

class Ambulance:
    def __init__(self, station):
        self.station = station
        self.available = True
        self.current_patient = None

class Station:
    def __init__(self, max_ambulances, location):
        self.max_ambulances = max_ambulances
        self.ambulances = []
        for i in range(max_ambulances):
            self.ambulances.append(Ambulance(self))
        self.location = location


def run_simulation(patient_count, station_count, scene_service_time = 15, hospital_service_time = 20, end_time = 500):
    call_times = set()

    patients = {}
    for _ in range(patient_count):
        triage_code = random.randint(0, 2)
        call_time = random.randint(1, end_time-1)
        while call_time in call_times:
            call_time = random.randint(1, end_time-1)
        call_times.add(call_time)

        long = random.random()*simulation_radius
        lat = random.random()*simulation_radius

        patients[call_time] = Patient(call_time, triage_code, [long, lat])

    stations = []
    for _ in range(station_count):

        max_ambulances = random.randint(1, 5)
        long = random.random()*simulation_radius
        lat = random.random()*simulation_radius

        stations.append(Station(max_ambulances, [long, lat]))

    return ambulance_routing_simulation(patients, stations, scene_service_time, hospital_service_time, end_time)
        

def ambulance_routing_simulation(patients, stations, scene_service_time, hospital_service_time, end_time = 500):

    default_count = traditional_approach(patients.copy(), stations, scene_service_time, hospital_service_time, end_time)
    death_count = 0

    patient_priority_queue = []

    current_time = 0
    routed_ambulance_stack = []
    home_station_stack = []

    for station in stations:
        for ambulance in station.ambulances:
            home_station_stack.append(ambulance)

    def calculate_distance(patient_location, station_location):
        return np.sqrt((patient_location[0] - station_location[0])**2 + (patient_location[1] - station_location[1])**2)

    # while CurrentTime <= EndTime do
    while current_time<=end_time:
        # for Routed Ambulance do
        for ambulance in routed_ambulance_stack:
            # if Ambulance Finished Job then
                    
            if current_time >= (ambulance.current_patient.call_time + calculate_distance(ambulance.current_patient.location, ambulance.station.location) + scene_service_time):
                    # Remove ambulance from the routed ambulance stack  
                    routed_ambulance_stack.remove(ambulance)

                    # Place the ambulance onto the home station stack, ready for redeployment
                    home_station_stack.append(ambulance)

                    if ambulance.current_patient.triage_code == 1 and current_time>(ambulance.current_patient.call_time + triage_one_ttl):
                        death_count+=1
                    elif ambulance.current_patient.triage_code == 0 and current_time>(ambulance.current_patient.call_time + triage_zero_ttl):
                        death_count+=1

                    patients.pop(ambulance.current_patient.call_time, None)

        # if Patient Time = Current Time then
        if current_time in patients:
            # Add the new patient onto the patient queue and sort by priority 
            patient = patients[current_time]
            # print(patient)
            heapq.heappush(patient_priority_queue, (-1*patient.triage_code, patient.call_time, patient))

            if patient_priority_queue:
                # Pop, the highest priority patient from the queue
                triage_code, call_time, patient = patient_priority_queue[0]
                
                # if One or more ambulances avallable to respond then
                if home_station_stack:

                    heapq.heappop(patient_priority_queue)

                    closest_ambulance = home_station_stack[0]
                    min_dist = float("inf")

                    for ambulance in home_station_stack:
                        ## Find the best ambulance to respond to the patient's call
                        if (dist_to_patient := calculate_distance(ambulance.station.location, patient.location))<min_dist:
                            min_dist = dist_to_patient
                            closest_ambulance = ambulance

                    # Add the routed ambulance to the routed stack
                    routed_ambulance_stack.append(closest_ambulance)
                    closest_ambulance.current_patient = patient

                    # Remove the chosen ambulance from the station stack
                    home_station_stack.remove(closest_ambulance)

                # if No ambulances available to respond then
                # else:
                #     # Increase time step and wait for the next ambulance
                #     print(patient_priority_queue)
                #     heapq.heappush(patient_priority_queue, (-1*triage_code, patient))

        # Increase time step
        current_time+=1
            

    # while Remaining Patients != 0 do
    while patients and routed_ambulance_stack:
        # print(current_time)
        # for Routed Ambulance do
        for ambulance in routed_ambulance_stack:
            # if Ambulance Finished Job then
            if current_time >= (ambulance.current_patient.call_time + calculate_distance(ambulance.current_patient.location, ambulance.station.location) + scene_service_time):
                    # Remove ambulance from the routed ambulance stack  
                    routed_ambulance_stack.remove(ambulance)

                    # Place the ambulance onto the home station stack, ready for redeployment
                    home_station_stack.append(ambulance)

                    if ambulance.current_patient.triage_code == 1 and current_time>(ambulance.current_patient.call_time + triage_one_ttl):
                        death_count+=1
                    elif ambulance.current_patient.triage_code == 0 and current_time>(ambulance.current_patient.call_time + triage_zero_ttl):
                        death_count+=1

                    patients.pop(ambulance.current_patient.call_time, None)
        current_time+=1

    return (death_count+len(patients), default_count)

def traditional_approach(patients, stations, scene_service_time, hospital_service_time, end_time):
    
    death_count = 0

    patient_queue = []

    current_time = 0
    routed_ambulance_stack = []
    home_station_stack = []

    for station in stations:
        for ambulance in station.ambulances:
            home_station_stack.append(ambulance)

    def calculate_distance(patient_location, station_location):
        return np.sqrt((patient_location[0] - station_location[0])**2 + (patient_location[1] - station_location[1])**2)

    # while CurrentTime <= EndTime do
    while current_time<=end_time:
        # for Routed Ambulance do
        for ambulance in routed_ambulance_stack:
            # if Ambulance Finished Job then
                    
            if current_time >= (ambulance.current_patient.call_time + calculate_distance(ambulance.current_patient.location, ambulance.station.location) + scene_service_time):
                    # Remove ambulance from the routed ambulance stack  
                    routed_ambulance_stack.remove(ambulance)

                    # Place the ambulance onto the home station stack, ready for redeployment
                    home_station_stack.append(ambulance)

                    if ambulance.current_patient.triage_code == 1 and current_time>(ambulance.current_patient.call_time + triage_one_ttl):
                        death_count+=1
                    elif ambulance.current_patient.triage_code == 0 and current_time>(ambulance.current_patient.call_time + triage_zero_ttl):
                        death_count+=1

                    patients.pop(ambulance.current_patient.call_time, None)

        # if Patient Time = Current Time then
        if current_time in patients:
            # print(current_time)
            # Add the new patient onto the patient queue and sort by priority 
            patient = patients[current_time]
            # print(patient)

            patient_queue.append(patient)

            if patient_queue:
                # Pop, the highest priority patient from the queue
                patient = patient_queue[0]
                
                # if One or more ambulances avallable to respond then
                if home_station_stack:

                    patient_queue.pop(0)

                    chosen_ambulance = home_station_stack[random.randint(0, len(home_station_stack)-1)]

                    # Add the routed ambulance to the routed stack
                    routed_ambulance_stack.append(chosen_ambulance)
                    chosen_ambulance.current_patient = patient

                    # Remove the chosen ambulance from the station stack
                    home_station_stack.remove(chosen_ambulance)

                # if No ambulances available to respond then
                # else:
                #     # Increase time step and wait for the next ambulance
                #     patient_queue.insert(0, patient)
        # Increase time step
        current_time+=1
            

    # while Remaining Patients != 0 do
    while patients and routed_ambulance_stack:
        # print(current_time)
        # for Routed Ambulance do
        for ambulance in routed_ambulance_stack:
            # if Ambulance Finished Job then
            if current_time >= (ambulance.current_patient.call_time + calculate_distance(ambulance.current_patient.location, ambulance.station.location) + scene_service_time):
                    # Remove ambulance from the routed ambulance stack  
                    routed_ambulance_stack.remove(ambulance)

                    # Place the ambulance onto the home station stack, ready for redeployment
                    home_station_stack.append(ambulance)

                    if ambulance.current_patient.triage_code == 1 and current_time>(ambulance.current_patient.call_time + triage_one_ttl):
                        death_count+=1
                    elif ambulance.current_patient.triage_code == 0 and current_time>(ambulance.current_patient.call_time + triage_zero_ttl):
                        death_count+=1

                    patients.pop(ambulance.current_patient.call_time, None)
        current_time+=1

    return death_count+len(patients)


def individual_patient_routing(patient, stations, scene_service_time = 15, hospital_service_time = 20, end_time = 500):
    client = openrouteservice.Client(key='5b3ce3597851110001cf624877ddcde30c3e4adda166d28874f89cef')

    current_time = 0
    home_station_stack = []

    for station in stations:
        for ambulance in station.ambulances:
            home_station_stack.append(ambulance)

    def calculate_distance(patient_location, station_location):
        res = client.directions(((station_location[1], station_location[0]),(patient_location[1], patient_location[0])))
        duration = round(res['routes'][0]['summary']['duration']/60,1)
        return duration

    # while CurrentTime <= EndTime do
    while current_time<=end_time:

        # if Patient Time = Current Time then
        if current_time == patient.call_time:
            if home_station_stack:

                closest_ambulance = home_station_stack[0]
                min_dist = float("inf")

                for ambulance in home_station_stack:

                    ## Find the best ambulance to respond to the patient's call
                    if (dist_to_patient := calculate_distance(ambulance.station.location, patient.location))<min_dist:
                        print(ambulance.station.location)
                        print(calculate_distance(ambulance.station.location, patient.location))
                        print("____")
                        min_dist = dist_to_patient
                        closest_ambulance = ambulance

                return [f"{patient.location[0]},{patient.location[1]}", f"{closest_ambulance.station.location[0]},{closest_ambulance.station.location[1]}"]

        current_time+=1

    return ""

# print(run_simulation(8000, 100, 10000))
# UI_health = [41.869431, -87.670517]
# RUSH_University_Medical_Center = [41.874668, -87.667519]
# Northwestern_Hospital = [41.896404, -87.620872]
# St_Alexius = [42.0524446, -88.1412199]
# UChicago_Medicine_AdventHealth_La_Grange = [41.8056045,-87.9209912]
# Weiss_Memorial_Hospital = [41.9664149,-87.6495973]
# Insight_Hospital_Medical_Center_Chicago = [41.846743,-87.6213224]
# Kindred_Hospital_Chicago_North = [41.9615459,-87.6929729]
# NorthShore_Highland_Park_Hospital = [42.1910918,-87.8083698]

# stations_data = [Station(2, UI_health), Station(2, RUSH_University_Medical_Center), Station(3, Northwestern_Hospital), Station(1, St_Alexius), Station(4, UChicago_Medicine_AdventHealth_La_Grange), Station(2, Weiss_Memorial_Hospital), Station(3, Insight_Hospital_Medical_Center_Chicago), Station(1, Kindred_Hospital_Chicago_North), Station(5, NorthShore_Highland_Park_Hospital)]
# patient = Patient(30, 2, [42.1939, -87.8022546])

# print(individual_patient_routing(patient, stations_data, 15, 20, 500)[0])