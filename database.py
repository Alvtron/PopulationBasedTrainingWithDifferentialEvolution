import os
import torch
from pathlib import Path
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

class ReadOnlyDatabase(object):
    def __init__(self, directory_path, database_name=None):
        self.ENTRIES_TAG = 'entries'
        self.DATE_CREATED = datetime.now()
        database_name = self.DATE_CREATED.strftime('%Y%m%d%H%M%S') if not database_name else database_name
        self.path = Path(f"{directory_path}/{database_name}")

    @property
    def exists(self):
        return self.path.is_dir

    def create_entry_directoy_path(self, id):
        """ Creates a new entry directory file path. """
        return Path(self.path, self.ENTRIES_TAG, f"{id:03d}")

    def create_entry_file_path(self, id, steps):
        """ Creates and returns a new database entry file path in the appropriate entry directory. """
        entry_directory = self.create_entry_directoy_path(id)
        return Path(entry_directory, f"{steps:05d}.pth")

    def get_entry(self, id, steps):
        """ Returns the specific entry stored on the specified id. If there is no match, None is returned. """
        entry_file_path = self.create_entry_file_path(id, steps)
        if not entry_file_path.is_file():
            return None
        entry = torch.load(entry_file_path)
        return entry

    def get_entry_directories(self):
        entries_path = Path(self.path, self.ENTRIES_TAG)
        return [content for content in entries_path.iterdir() if content.is_dir()]
            
    def get_entries(self, entry_directory):
        """ Retrieve all entries made on the specified id. """
        return [torch.load(content) for content in entry_directory.iterdir()]

    def to_dict(self):
        """ Returns a the database converted to a dictionary grouped by id/filename/entry"""
        dict_of_entries = dict()
        for directory in self.get_entry_directories():
            entries = self.get_entries(directory)
            if not id in dict_of_entries:
                dict_of_entries[id] = dict()
            for entry in entries:
                dict_of_entries[id][entry.steps] = entry
        return dict_of_entries
        
    def to_list(self):
        """ Returns a the database converted to a list of all entries """
        list_of_entries = []
        for directory in self.get_entry_directories():
            entries = self.get_entries(directory)
            list_of_entries.extend(entries)
        return list_of_entries

    def print(self):
        """ Prints all entries in the database. """
        database_list = self.to_list()
        database_list_sorted = sorted(database_list, key=lambda x: (x.id, x.steps), reverse=False)
        for entry in database_list_sorted:
            print(entry)

class SharedDatabase(ReadOnlyDatabase):
    def __init__(self, directory_path, database_name=None):
        super().__init__(directory_path, database_name)
        self.path.mkdir(parents=True, exist_ok=True)
        mp = torch.multiprocessing.get_context('spawn')
        manager = mp.Manager() 
        self.cache = manager.dict()

    def append_to_file(self, tag, file_name, text):
        """ Append the provided string to the specified filename. The file will be saved in a folder named after the specified tag-string, which is located in the database directory. """
        if not isinstance(text, str):
            raise ValueError("The provided text must be of type string!")
        file_path = Path(self.path, tag, file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('a+') as file:
            file.write(text + '\n')

    def save_entry_to_file(self, entry):
        """ Save the provided database entry to a file inside the database directory. """
        entry_file_path = self.create_entry_file_path(entry.id, entry.steps)
        entry_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(entry, entry_file_path)

    def save_entry(self, entry):
        """ Saves the provided entry to the database. This method saves entry to memory, and will replace any old entry. In addition, the method saves the provided entry to the database directory. """
        # Save entry to memory. This replaces the old entry.
        self.cache[entry.id] = entry
        # Save entry to database directory.
        self.save_entry_to_file(entry)

    def get_entry(self, id, steps=None):
        """ Returns the specific entry stored on the specified id. If there is no match, None is returned. """
        if steps == None:
            return self.cache[id] if self.cache[id] else None
        return super.get_entry(id, steps)

    def get_latest(self):
        """ Returns a list containing the latest entry from every member. """
        return self.cache.values()