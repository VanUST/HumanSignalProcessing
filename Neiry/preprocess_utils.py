from pathlib import Path
import numpy as np
import h5py
import json
import torch
from torch.utils.data import dataloader, dataset

json_filename="/home/giga_ivan/projects/HandyRobotics/Neiry/data/example/user_activity.json"

h5_filename = "/home/giga_ivan/projects/HandyRobotics/Neiry/data/example/session_eeg.h5" # Path to file

def _read_h5(filename):

    filename = Path(filename)
    datasets = []
    start_times  = []
    channels = []

    with h5py.File(filename, "r") as f:
        for df_keys in f.keys():      
            datasets.append(np.array(f[df_keys]))    
            for attrs_keys in f[df_keys].attrs.keys():
                if type(f[df_keys].attrs[attrs_keys]) != np.int64:
                    print(f"{df_keys} => {json.loads(f[df_keys].attrs[attrs_keys])}")
                    channels.append(json.loads(f[df_keys].attrs[attrs_keys]))
                else:
                    print(f"{df_keys} start time => {f[df_keys].attrs[attrs_keys]}", "\n")
                    start_times.append(f[df_keys].attrs[attrs_keys])


    return datasets,start_times, channels

def _read_timestamps(json_filename):
    # Opening JSON file
    f = open(json_filename)

    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    return data

def _pair_activity_with_eeg(dataset, start_time_index, end_time_index,activity):

    key = activity+"_"+str(start_time_index)

    marked_data = {key:[]}

    time_index = start_time_index + 0

    while time_index < end_time_index:
        marked_data[key].append(dataset[time_index])
        time_index +=1

    return marked_data

def _pair_eeg_dataset_with_activity_timestamps(dataset,global_start_time,timestamps):

    # Takes in dataset, return dictionary with activityName_activityStartTime as key, and the sequence itsef as value

    marked_dataset = {}

    for timestamp in timestamps:
        activity = timestamp['activity']
        times = timestamp['times']
        for time in times:
            if type(time) == dict:
                if 'start_time' in time.keys():
                    start_time_index = int((time['start_time'] - global_start_time)/(1000*4))
                    end_time_index = int((time['end_time'] - global_start_time)/(1000*4))

                    marked_dataset.update(_pair_activity_with_eeg(dataset=dataset,start_time_index= start_time_index, end_time_index= end_time_index, activity= activity))


                    # print("1 ",dataset[start_time][0])
                    # print("2",start_time*4)
                

    return marked_dataset 

def create_marked_dictionary_from_h5_and_json(h5_filename,json_filename):

    datasets,start_times, channels = _read_h5(filename=h5_filename)

    timestamps = _read_timestamps(json_filename=json_filename)

    marked_dataset = _pair_eeg_dataset_with_activity_timestamps(dataset = datasets[0],global_start_time=start_times[0],timestamps=timestamps)

    return marked_dataset



marked_dict = create_marked_dictionary_from_h5_and_json(h5_filename=h5_filename, json_filename= json_filename)

