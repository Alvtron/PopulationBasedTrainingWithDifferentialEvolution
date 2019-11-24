import os
import torch
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Checkpoint:
    '''Class for keeping track of a worker.'''
    id: int
    epochs: int
    steps: int
    model_state: dict
    optimizer_state: dict
    hyper_parameters: dict
    score: float

    def __str__(self):
        return f"Member {self.id} - epoch {self.epochs} / step {self.steps} - {self.score:.2f}%"

    def update(self, checkpoint):
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.hyper_parameters = checkpoint.hyper_parameters
        self.score = None


class SharedDatabase(object):
    def __init__(self, directory_path, shared_memory_dict):
        self.date_created = datetime.now()
        self.directory_path = directory_path
        date_time_string = self.date_created.strftime('%Y%m%d%H%M%S')
        self.database_path = f"{directory_path}/{date_time_string}"
        self.create()
        self.data = shared_memory_dict

    def create(self):
        """ Create database directory """
        if not os.path.isdir(self.directory_path):
            os.mkdir(self.directory_path)
        if not os.path.isdir(self.database_path):
            os.mkdir(self.database_path)

    def delete(self):
        """ Delete database directory """
        if os.path.isdir(self.directory_path):
            os.rmdir(self.directory_path)
        if os.path.isdir(self.database_path):
            os.rmdir(self.database_path)

    def save_entry_to_file(self, entry):
        file_path = f"{self.database_path}/{entry.id:03d}-{entry.steps:04d}.pth"
        torch.save(entry, file_path)

    def save_entry(self, entry):
        """ Save new entry to the database. """
        if not entry.id in self.data:
            self.data[entry.id] = list()
        self.data[entry.id] += [entry]

    def get_entry(self, id, steps):
        """ Returns the specific entry stored on the specified id and step. If there is no match, None is returned. """
        if 0 <= steps < len(self.data[id]):
            return self.data[id][steps]
        else: return None

    def get_latest_entry(self, id):
        """ Returns the specific entry stored on the specified id and step. If there is no match, None is returned. """
        return self.data[id][-1]

    def get_entries(self, id):
        """ Returns all entries stored on the specified id. If there is no match, None is returned. """
        if id in self.data:
            return self.data[id]
        else: return None

    def get_latest(self):
        """ Returns a list containing the latest entry from every member. """
        list_of_entries = []
        for entries in self.data.values():
            if len(entries) > 0:
                list_of_entries.append(entries[-1])
        return list_of_entries

    def to_dict(self):
        """ Returns a the database converted to a dictionary grouped by id/step/entry"""
        dict_of_entries = {}
        for entries in self.data.values():
            for entry in entries:
                entries[entry.id][entry.steps] = entry
        return dict_of_entries
        
    def to_list(self):
        """ Returns a the database converted to a list of all entries """
        list_of_entries = []
        for entries in self.data.values():
            for entry in entries:
                list_of_entries.append(entry)
        return list_of_entries

    def print(self):
        database_list = self.to_list()
        database_list_sorted = sorted(database_list, key=lambda x: (x.id, x.steps), reverse=False)
        for entry in database_list_sorted:
            print(entry)