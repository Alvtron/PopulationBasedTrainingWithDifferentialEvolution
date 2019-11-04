import os
import glob
import torch
import enum
import operator
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Checkpoint:
    '''Class for keeping track of a worker.'''
    id: int
    epoch: int
    model_state: dict
    optimizer_state: dict
    hyperparameters: dict
    batch_size: int
    score: float
    is_mutated: bool
    
    def __str__(self):
        if self.is_mutated:
            return f"Member {self.id} - epoch {self.epoch} - {self.score:.2f}% (mutated)"
        else:
            return f"Member {self.id} - epoch {self.epoch} - {self.score:.2f}%"

    def update(self, checkpoint):
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.hyperparameters = checkpoint.hyperparameters
        self.batch_size = checkpoint.batch_size
        score = None

class Database(object):
    def __init__(self, directory_path):
        self.date_created = datetime.now()
        self.directory_path = directory_path
        date_time_string = self.date_created.strftime('%Y%m%d%H%M%S')
        self.database_path = f"{directory_path}/{date_time_string}"
        self.create()

    def create(self):
        if not os.path.isdir(self.directory_path):
            os.mkdir(self.directory_path)
        if not os.path.isdir(self.database_path):
            os.mkdir(self.database_path)

    def delete(self):
        if os.path.isdir(self.directory_path):
            os.rmdir(self.directory_path)
        if os.path.isdir(self.database_path):
            os.rmdir(self.database_path)

    def create_entry_directoy_path(self, id):
        entry_directory = f"{self.database_path}/{id:03d}"
        # create entry directory if not exist
        if not os.path.isdir(entry_directory):
            os.mkdir(entry_directory)
        return entry_directory

    def create_entry_file_path(self, id, epoch):
        entry_directory = self.create_entry_directoy_path(id)
        return f"{entry_directory}/{epoch:03d}.pth"

    def save_entry(self, entry):
        """ Save new entry to the database. """
        entry_directory_path = self.create_entry_file_path(entry.id, entry.epoch)
        torch.save(entry, entry_directory_path)

    def get_entry(self, id, epoch):
        """ Returns the specific entry stored on the specified id and epoch. If there is no match, None is returned. """
        entry_file_path = self.create_entry_file_path(id, epoch)
        if os.path.isfile(entry_file_path):
            entry = torch.load(entry_file_path)
            return entry
        else:
            return None

    def get_entries(self, id):
        """ Returns all entries stored on the specified id. If there is no match, None is returned. """
        entry_directory_path = self.create_entry_directoy_path(id)
        entries = []
        with os.scandir(entry_directory_path) as files:
            for file in files:
                entry = torch.load(file.path)
                entries.append(entry)
        return entries

    def get_latest(self):
        """ Returns a list containing the latest entry from every member. """
        entries = []
        with os.scandir(self.database_path) as directories:
            for directory in directories:
                files = glob.glob(f"{directory.path}/*.pth")
                latest_file = max(files, key=os.path.getctime)
                entry = torch.load(latest_file)
                entries.append(entry)
        return entries

    def to_dict(self):
        """ Returns a the database converted to a dictionary grouped by id/epoch/entry"""
        entries = {}
        with os.scandir(self.database_path) as directories:
            for directory in directories:
                with os.scandir(directory.path) as files:
                    for file in files:
                        entry = torch.load(file.path)
                        entries[entry.id][entry.epoch] = entry
        return entries
        
    def to_list(self):
        """ Returns a the database converted to a list of all entries """
        entries = []
        with os.scandir(self.database_path) as directories:
            for directory in directories:
                with os.scandir(directory.path) as files:
                    for file in files:
                        entry = torch.load(file.path)
                        entries.append(entry)
        return entries

    def print(self):
        database_list = self.to_list()
        database_list_sorted = sorted(database_list, key=lambda x: (x.id, x.epoch), reverse=False)
        for entry in database_list_sorted:
            print(entry)