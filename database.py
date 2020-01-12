import os
import glob
import pickle
from pathlib import Path
from datetime import datetime

class Checkpoint(object):
    '''Class for keeping track of a worker.'''
    def __init__(self, id, hyper_parameters, loss_metric, eval_metric):
        self.id = id
        self.epochs = 0
        self.steps = 0
        self.hyper_parameters = hyper_parameters
        self.model_state = None
        self.optimizer_state = None
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss = dict()
        self.time = dict()

    @property
    def score(self):
        return self.loss['eval'][self.eval_metric]

    def __str__(self):
        string = f"Member {self.id:03d}, epoch {self.epochs}, step {self.steps}"
        for loss_group, loss_values in self.loss.items():
            for loss_name, loss_value in loss_values.items():
                string += f", {loss_group}_{loss_name} {loss_value:.5f}"
        return string

    def update(self, checkpoint):
        self.hyper_parameters = checkpoint.hyper_parameters
        self.model_state = checkpoint.model_state
        self.optimizer_state = checkpoint.optimizer_state
        self.loss = dict()
        self.time = dict()

class ReadOnlyDatabase(object):
    def __init__(self, database_path, read_function=None):
        self.ENTRY_EXT = "pth"
        self.ENTRIES_TAG = 'entries'
        self.path = Path(database_path)
        # set read function
        def read(path): return pickle.load(path.open('rb'))
        self.read = read if not read_function else read_function

    @property
    def exists(self):
        return self.path.is_dir()

    def __iter__(self):
        for directory in self.entry_directories():
            for entry in self.entries_from_path(directory):
                yield entry

    def __len__(self):
        entries_path = Path(self.path, self.ENTRIES_TAG)
        return len(list(entries_path.glob(f'**/*.{self.ENTRY_EXT}')))

    def __contains__(self, id):
        return self.create_entry_directoy_path(id).exists()

    def create_entry_directoy_path(self, id):
        """ Creates a new entry directory file path. """
        return Path(self.path, self.ENTRIES_TAG, f"{id:03d}")

    def create_entry_file_path(self, id, key):
        """ Creates and returns a new database entry file path in the appropriate entry directory. """
        entry_directory = self.create_entry_directoy_path(id)
        entry_file_name = f"{key:05d}.{self.ENTRY_EXT}"
        return Path(entry_directory, entry_file_name)

    def entry(self, id, key):
        """ Returns the specific entry stored on the specified id. If there is no match, None is returned. """
        entry_file_path = self.create_entry_file_path(id, key)
        return self.read(entry_file_path) if entry_file_path.is_file() else None

    def entries(self, id):
        """ Iterate over the entry directory matching the specified id and yield all entries inside the directory. """
        entry_directory = self.create_entry_directoy_path(id)
        for content in entry_directory.glob(f"*.{self.ENTRY_EXT}"):
            yield self.read(content)

    def latest(self):
        """ Iterate over all entry directories and yield the latest entry. """
        for entry_dir in self.entry_directories():
            yield max(entry_dir.glob(f"*.{self.ENTRY_EXT}"), key=os.path.getctime)

    def entry_directories(self):
        entries_path = Path(self.path, self.ENTRIES_TAG)
        for content in entries_path.iterdir():
            if content.is_dir(): yield content

    def entries_from_path(self, entry_directory_path):
        """ Retrieve all entries made on the specified id. """
        for content in entry_directory_path.iterdir():
            yield self.read(content)

    def to_dict(self):
        """ Returns a the database converted to a dictionary grouped by id/filename/entry"""
        dict_of_entries = dict()
        for directory in self.entry_directories():
            entries = self.entries_from_path(directory)
            if not directory.name in dict_of_entries:
                dict_of_entries[directory.name] = dict()
            for entry in entries:
                dict_of_entries[directory.name][entry.steps] = entry
        return dict_of_entries

    def print(self):
        """ Prints all entries in the database. """
        for entry in sorted(self, key=lambda x: (x.id, x.steps), reverse=False):
            print(entry)

class Database(ReadOnlyDatabase):
    def __init__(self, directory_path, database_name=None, read_function=None, write_function=None):
        # set database path
        database_name = datetime.now().strftime('%Y%m%d%H%M%S') if not database_name else database_name
        database_path = Path(directory_path, database_name)
        # init parent object
        super().__init__(database_path=database_path, read_function=read_function)
        # create database directory
        self.path.mkdir(parents=True, exist_ok=True)
        # set write function
        def write(entry, path): pickle.dump(entry, path.open('wb'))
        self.write = write if not write_function else write_function

    def create_folder(self, name):
        """ Create a new folder located in the database base directory. Name supports nested directories of type dir/sub_dir/sub_sub_dir/etc. """
        folder_path = Path(self.path, name)
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def create_file(self, tag, file_name):
        """ Create a new file in a folder named after the specified tag-string, which is located in the database directory. """
        file_path = Path(self.path, tag, file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    def update(self, id, key, entry):
        """ Save the provided database entry to a file on id/key inside the database directory. """
        entry_file_path = self.create_entry_file_path(id, key)
        entry_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.write(entry, entry_file_path)

class SharedDatabase(Database):
    def __init__(self, context, directory_path, database_name=None, read_function=None, write_function=None):
        super().__init__(
            directory_path=directory_path,
            database_name=database_name,
            read_function=read_function,
            write_function=write_function)
        manager = context.Manager() 
        self.cache = manager.dict()

    def update(self, id, key, entry, ignore_exception=False):
        """
        Saves entry to memory, and will replace any old entry.
        In addition, the method saves the provided entry to a file on id/key inside the database directory.
        """
        # Save entry to cache memory. This replaces the old entry.
        try:
            self.cache[id] = entry
        except Exception as exception:
            print(f"Failed to write entry with id: {id}, key: {key}, to cache.")
            if not ignore_exception: raise exception
        # Save entry to database directory.
        super().update(id, key, entry)

    def entry(self, id, key=None):
        """
        Returns the specific entry stored on the specified id.
        If there is no match, None is returned.
        """
        return self.cache[id] if key == None and id in self.cache else super().entry(id, key)

    def latest(self):
        """ Returns a list containing the latest entry from every member. """
        return self.cache.values()