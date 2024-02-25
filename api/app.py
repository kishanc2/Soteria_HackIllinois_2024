from flask import Flask, render_template, session, redirect, url_for, request
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from dateutil import parser
from sqlalchemy import create_engine, exc, text
import sqlalchemy
import os
import json
import pandas as pd 
import numpy as np
import numpy.linalg as la
from sklearn.model_selection import train_test_split
import pickle
from routing import individual_patient_routing, Station, Patient
# from flask_session import Session

app = Flask(__name__)

def classify(W : np.ndarray, x : np.ndarray) -> int:
    
    T = 1e2
    N = x.shape[0]
    return np.argmax([np.exp((W @ x)/T)/np.sum(np.exp((W @ x)/T))])



@app.route('/', methods = ['GET', 'POST'])
def home():
    
    return render_template('index.html')

@app.route('/pinfo.html', methods = ['GET', 'POST'])
def pinfo():
    f = open('weights-2.pkl', 'rb')
    W = pickle.load(f)
    f.close()
    f = open("stoi.pkl", 'rb')
    stoi = pickle.load(f)
    
    if request.method == 'POST':
        loc = request.form['loc']
        length_of_sim = request.form['length_of_sim']
        avg_scene = request.form['avg_scene']
        avg_hosp = request.form['avg_hosp']
        time_acc = request.form['time_acc']
        age_driver = stoi[request.form['age_driver']]
        sex_driver = stoi[request.form['sex_driver']]
        edu = stoi[request.form['edu']]
        driver_rel = stoi[request.form['driver_rel']]
        driver_exp = stoi[request.form['driver_exp']]
        lanes_medians = stoi[request.form['lanes_medians']]
        junction = stoi[request.form['junction']]
        surface = stoi[request.form['surface']]
        light = stoi[request.form['light']]
        weather = stoi[request.form['weather']]
        coll_type = stoi[request.form['coll_type']]
        vehicle_movement = stoi[request.form['vehicle_movement']]
        ped_movement = stoi[request.form['ped_movement']]
        accident_cause = stoi[request.form['accident_cause']]
     


        print(loc)
        print(length_of_sim)
        # print(num_of_ambulances)
        print(avg_scene)
        print(avg_hosp)
        print(time_acc)
        print(age_driver)
        print(sex_driver)
        print(edu)
        print(driver_rel)
        print(driver_exp)
        print(lanes_medians)
        print(junction)
        print(surface)
        print(light)
        print(weather)
        print(coll_type)
        print(vehicle_movement)
        print(ped_movement)
        print(accident_cause)

    
        arr = []
        arr.append(age_driver)
        arr.append(sex_driver)
        arr.append(edu)
        arr.append(driver_rel)
        arr.append(driver_exp)
        arr.append(lanes_medians)
        arr.append(junction)
        arr.append(surface)
        arr.append(light)
        arr.append(weather)
        arr.append(coll_type)
        arr.append(vehicle_movement)
        arr.append(ped_movement)
        arr.append(accident_cause)

        n_arr = np.array(arr)

        severity = classify(W, n_arr)

        print(severity)

        coords = (loc.split(","))

        UI_health = [41.869431, -87.670517]
        RUSH_University_Medical_Center = [41.874668, -87.667519]
        Northwestern_Hospital = [41.896404, -87.620872]
        St_Alexius = [42.0524446, -88.1412199]
        UChicago_Medicine_AdventHealth_La_Grange = [41.8056045,-87.9209912]
        Weiss_Memorial_Hospital = [41.9664149,-87.6495973]
        Insight_Hospital_Medical_Center_Chicago = [41.846743,-87.6213224]
        Kindred_Hospital_Chicago_North = [41.9615459,-87.6929729]
        NorthShore_Highland_Park_Hospital = [42.1910918,-87.8083698]

        stations_data = [Station(2, UI_health), Station(2, RUSH_University_Medical_Center), Station(3, Northwestern_Hospital), Station(1, St_Alexius), Station(4, UChicago_Medicine_AdventHealth_La_Grange), Station(2, Weiss_Memorial_Hospital), Station(3, Insight_Hospital_Medical_Center_Chicago), Station(1, Kindred_Hospital_Chicago_North), Station(5, NorthShore_Highland_Park_Hospital)]
        patient = Patient(int(time_acc), severity, [float(coords[0]), float(coords[1])])

        coordinate_tuple = individual_patient_routing(patient, stations_data, int(avg_scene), int(avg_hosp), int(length_of_sim))
        
        source_coords = coordinate_tuple[0]
        dest_coords = coordinate_tuple[1]

        print(source_coords)
        print(dest_coords)

        source_coords_json = json.dumps(source_coords)
        dest_coords_json = json.dumps(dest_coords)

        print(source_coords_json)
        print(dest_coords_json)


        return render_template('map.html', source_coord=source_coords_json, dest_coord=dest_coords_json)


    return render_template('pinfo.html')

@app.route('/map.html', methods = ['GET', 'POST'])
def map():
    return render_template('map.html')