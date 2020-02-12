import os
import glob
import pickle
import copy
import time
from pathlib import Path
from datetime import datetime

class ReadOnlyDatabase(object):
    def __init__(self, database_path, read_function=None, extension = "obj"):
        self.ENTRIES_TAG = 'entries'
        self.extension = extension
        self.path = Path(database_path)
        # set read function
        def read(path): return pickle.load(path.open('rb'))
        self.read = read if not read_function else read_function

    @property
    def exists(self):
        return self.path.is_dir()

    def __len__(self):
        entries_path = Path(self.path, self.ENTRIES_TAG)
        return len(list(entries_path.glob(f'**/*.{self.extension}')))

    def __iter__(self):
        for directory in self.entry_directories():
            for entry in self.entries_from_path(directory):
                yield entry

    def __contains__(self, id):
        return self.create_entry_directoy_path(id).exists()

    def create_entry_file_name(self, key):
        return f"{key:05d}.{self.extension}"

    def create_entry_directoy_path(self, id):
        """ Creates a new entry directory file path. """
        return Path(self.path, self.ENTRIES_TAG, f"{id:03d}")

    def create_entry_file_path(self, id, key):
        """ Creates and returns a new database entry file path in the appropriate entry directory. """
        entry_directory = self.create_entry_directoy_path(id)
        entry_file_name = self.create_entry_file_name(key)
        return Path(entry_directory, entry_file_name)

    def entry(self, id, key):
        """ Returns the specific entry stored on the specified id. If there is no match, None is returned. """
        entry_file_path = self.create_entry_file_path(id, key)
        return self.read(entry_file_path) if entry_file_path.is_file() else None

    def entries(self, id):
        """ Iterate over the entry directory matching the specified id and yield all entries inside the directory. """
        entry_directory = self.create_entry_directoy_path(id)
        for content in entry_directory.glob(f"*.{self.extension}"):
            yield self.read(content)

    def latest(self):
        """ Iterate over all entry directories and yield the latest entry. """
        for entry_dir in self.entry_directories():
            newest_entry = max(entry_dir.glob(f"*.{self.extension}"), key=os.path.getctime)
            yield self.read(newest_entry)

    def entry_directories(self):
        entries_path = Path(self.path, self.ENTRIES_TAG)
        for content in entries_path.iterdir():
            if content.is_dir(): yield content

    def entries_from_path(self, entry_directory_path):
        """ Retrieve all entries made on the specified id. """
        for content in entry_directory_path.glob(f"*.{self.extension}"):
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
    def __init__(self, directory_path, database_name=None, read_function=None, write_function=None, extension = "obj"):
        # set database path
        if not database_name:
            database_name = datetime.now().strftime('%Y%m%d%H%M%S')
        database_path = Path(directory_path, database_name)
        if database_path.exists():
            raise ValueError(f"Database path is occupied: {database_path}")
        # init parent object
        super().__init__(database_path=database_path, read_function=read_function, extension=extension)
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