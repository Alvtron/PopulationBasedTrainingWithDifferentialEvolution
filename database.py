import os
import torch
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Checkpoint(object):
    '''Class for keeping track of a worker.'''
    def __init__(self, id, hyper_parameters):
        self.id = id
        self.epochs = 0
        self.steps = 0
        self.hyper_parameters = hyper_parameters
        self.model_state = None
        self.optimizer_state = None
        self.train_loss = None
        self.eval_score = None
        self.test_score = None
        self.train_time = None
        self.evolve_time = None

    def __str__(self):
        string = f"Member {self.id:03d}, epoch {self.epochs}, step {self.steps}"
        if self.train_loss:
            string += f", loss {self.train_loss:.5f}"
        if self.eval_score:
            string += f", eval {self.eval_score:.5f}"
        if self.test_score:
            string += f", test {self.test_score:.5f}"
        return string

    def update(self, checkpoint):
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.hyper_parameters = checkpoint.hyper_parameters
        self.train_loss = checkpoint.train_loss
        self.eval_score = checkpoint.eval_score
        self.test_score = checkpoint.test_score

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

    def save_to_file(self, filename, text):
        """ Append the provided string to the specified filename. The file will be saved in the database directory. """
        if not isinstance(text, str):
            raise ValueError("The provided text must be of type string!")
        file_path = f"{self.database_path}/{filename}"
        with open(file_path, 'a+') as file:
            file.write(text + '\n')

    def create_file_path(self, file_name):
        """ Creates and returns a filepath originating from the main database directory. """
        return f"{self.database_path}/{file_name}"

    def create_entry_directoy_path(self, id):
        """ Creates and returns a new entry directory file path. If the directory does not exists, create a new one. """
        entry_directory = f"{self.database_path}/{id:03d}"
        # create entry directory if not exist
        if not os.path.isdir(entry_directory):
            os.mkdir(entry_directory)
        return entry_directory

    def create_entry_file_path(self, id, steps):
        """ Creates and returns a new database entry file path in the appropriate entry directory """
        entry_directory = self.create_entry_directoy_path(id)
        return f"{entry_directory}/{steps:05d}.pth"

    def save_entry_to_file(self, entry):
        """ Save the provided database entry to a file inside the database directory. """
        entry_file_path = self.create_entry_file_path(entry.id, entry.steps)
        torch.save(entry, entry_file_path)

    def save_entry(self, entry):
        """ Saves the provided entry to the database. This method saves entry to memory, and will replace any old entry. In addition, the method saves the provided entry to the database directory. """
        # Save entry to memory. This replaces the old entry.
        self.data[entry.id] = entry
        # Save entry to database directory.
        self.save_entry_to_file(entry)

    def get_entry(self, id, steps=None):
        """ Returns the specific entry stored on the specified id. If there is no match, None is returned. """
        if steps == None:
            return self.data[id] if self.data[id] else None
        else:
            entry_file_path = self.create_entry_file_path(id, steps)
            if not os.path.isfile(entry_file_path):
                return None
            entry = torch.load(entry_file_path)
            return entry

    def get_latest(self):
        """ Returns a list containing the latest entry from every member. """
        list_of_entries = []
        for entry in self.data.values():
            list_of_entries.append(entry)
        return list_of_entries

    def get_entries_from_files(self, id):
        """ Retrieve all entries made on the specified id. """
        entry_directory_path = self.create_entry_directoy_path(id)
        entries = []
        with os.scandir(entry_directory_path) as files:
            for file in files:
                entry = torch.load(file.path)
                entries.append(entry)
        return entries

    def to_dict(self):
        """ Returns a the database converted to a dictionary grouped by id/step/entry"""
        dict_of_entries = dict()
        for id in self.data.keys():
            entries = self.get_entries_from_files(id)
            if not id in dict_of_entries:
                dict_of_entries[id] = dict()
            for entry in entries:
                dict_of_entries[id][entry.steps] = entry
        return dict_of_entries
        
    def to_list(self):
        """ Returns a the database converted to a list of all entries """
        list_of_entries = []
        for id in self.data.keys():
            entries = self.get_entries_from_files(id)
            list_of_entries.extend(entries)
        return list_of_entries

    def print(self):
        """ Prints all entries in the database. """
        database_list = self.to_list()
        database_list_sorted = sorted(database_list, key=lambda x: (x.id, x.steps), reverse=False)
        for entry in database_list_sorted:
            print(entry)