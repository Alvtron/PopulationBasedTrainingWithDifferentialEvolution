import os
import glob
import pickle
import copy
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Union, Callable, TypeVar, Generator

Entity = TypeVar('Entity')

class ReadOnlyDatabase(object):

    ENTRIES_TAG = 'entries'

    def __init__(self, database_path: Union[Path, str], read_function: Callable[[str], Entity] = None, extension: str = "obj"):
        if not isinstance(database_path, (Path, str)):
            raise TypeError(f"the 'database_path' specified was of wrong type {type(database_path)}, expected {Path} or {str}.")
        if read_function is not None and not callable(read_function):
            raise TypeError(f"the 'read_function' specified was not callable.")
        if not isinstance(extension, str):
            raise TypeError(f"the 'extension' specified was of wrong type {type(extension)}, expected {str}.")
        self.extension = extension
        self.path = Path(database_path)
        # set read function
        def read(path): return pickle.load(path.open('rb'))
        self.read = read if not read_function else read_function

    @property
    def exists(self) -> bool:
        return self.path.is_dir()

    def __len__(self) -> int:
        entries_path = Path(self.path, ReadOnlyDatabase.ENTRIES_TAG)
        return len(list(entries_path.glob(f'**/*.{self.extension}')))

    def __iter__(self) -> Generator[Entity, None, None]:
        for directory in self.entry_directories():
            for entry in self.entries_from_path(directory):
                yield entry

    def __contains__(self, uid: Any):
        return self.create_entry_directoy_path(uid).exists()

    def create_entry_file_name(self, key: Any):
        return f"{key}.{self.extension}"

    def create_entry_directoy_path(self, uid: Any):
        """ Creates a new entry directory file path. """
        return Path(self.path, ReadOnlyDatabase.ENTRIES_TAG, str(uid))

    def create_entry_file_path(self, uid, key):
        """ Creates and returns a new database entry file path in the appropriate entry directory. """
        entry_directory = self.create_entry_directoy_path(uid)
        entry_file_name = self.create_entry_file_name(key)
        return Path(entry_directory, entry_file_name)

    def entry(self, uid: Any, key: Any) -> Entity:
        """ Returns the specific entry stored on the specified uid. If there is no match, None is returned. """
        entry_file_path = self.create_entry_file_path(uid, key)
        return self.read(entry_file_path) if entry_file_path.is_file() else None

    def entries(self, uid: Any) -> Generator[Entity, None, None]:
        """ Iterate over the entry directory matching the specified uid and yield all entries inside the directory. """
        entry_directory = self.create_entry_directoy_path(uid)
        for content in entry_directory.glob(f"*.{self.extension}"):
            yield self.read(content)

    def entry_directories(self) -> Generator[Path, None, None]:
        entries_path = Path(self.path, ReadOnlyDatabase.ENTRIES_TAG)
        for content in entries_path.iterdir():
            if content.is_dir(): yield content

    def entries_from_path(self, entry_directory_path: Path) -> Generator[Entity, None, None]:
        """ Retrieve all entries made on the specified uid. """
        for content in entry_directory_path.glob(f"*.{self.extension}"):
            yield self.read(content)

    def identy_records(self) -> Dict[str, list]:
        """ Returns a uid/keys dictionary. """
        result = dict()
        for directory in self.entry_directories():
            result[directory.name] = list()
            for key_path in directory.glob("*"):
                result[directory.name].append(key_path.stem)
        return result

    def get_first(self) -> Generator[Entity, None, None]:
        """ Returns an iterator over the first entries in this database """
        min_steps = min(min(int(key) for key in keys) for keys in self.database.identy_records().values())
        for uid, keys in self.database.identy_records().items():
            min_key = min(int(key) for key in keys)
            if min_key > min_steps:
                continue
            yield self.database.entry(uid, min_key)

    def get_last(self) -> Generator[Entity, None, None]:
        """ Returns an iterator over the last entries in this database """
        max_steps = max(max(int(key) for key in keys) for keys in self.identy_records().values())
        for uid, keys in self.identy_records().items():
            max_key = max(int(key) for key in keys)
            if max_key < max_steps:
                continue
            yield self.entry(uid, max_key)

    def to_dict(self) -> dict:
        """ Returns a the database converted to a dictionary grouped by uid/filename/entry"""
        dict_of_entries = dict()
        for directory in self.entry_directories():
            entries = self.entries_from_path(directory)
            if not directory.name in dict_of_entries:
                dict_of_entries[directory.name] = dict()
            for entry in entries:
                dict_of_entries[directory.name][entry.steps] = entry
        return dict_of_entries

    def print(self) -> None:
        """ Prints all entries in the database. """
        for entry in sorted(self, key=lambda x: (x.uid, x.steps), reverse=False):
            print(entry)

class Database(ReadOnlyDatabase):
    def __init__(
            self, directory_path: Union[Path, str], database_name: str = None,
            read_function: Callable[[str], Entity] = None, write_function: Callable[[Entity], None] = None,
            extension: str = "obj"):
        if not isinstance(directory_path, (Path, str)):
            raise TypeError(f"the 'directory_path' specified was of wrong type {type(directory_path)}, expected {Path} or {str}.")
        if database_name is not None and not isinstance(database_name, str):
            raise TypeError(f"the 'database_name' specified was of wrong type {type(database_name)}, expected {str}.")
        if read_function is not None and not callable(read_function):
            raise TypeError(f"the 'read_function' specified was not callable.")
        if write_function is not None and not callable(write_function):
            raise TypeError(f"the 'write_function' specified was not callable.")
        if not isinstance(extension, str):
            raise TypeError(f"the 'extension' specified was of wrong type {type(extension)}, expected {str}.")
        # set database path
        if not database_name:
            database_name = datetime.now().strftime('%Y%m%d%H%M%S')
        database_path = Path(directory_path, database_name)
        if database_path.exists():
            raise FileExistsError(f"Database path is occupied: {database_path}")
        # init parent object
        super().__init__(database_path=database_path, read_function=read_function, extension=extension)
        # create database directory
        self.path.mkdir(parents=True, exist_ok=True)
        # set write function
        def write(entry, path): pickle.dump(entry, path.open('wb'))
        self.write = write if not write_function else write_function

    def create_folder(self, name: str) -> Path:
        """ Create a new folder located in the database base directory. Name supports nested directories of type dir/sub_dir/sub_sub_dir/etc. """
        folder_path = Path(self.path, name)
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def create_file(self, tag: str, file_name: str) -> Path:
        """ Create a new file in a folder named after the specified tag-string, which is located in the database directory. """
        file_path = Path(self.path, tag, file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    def update(self, uid: Any, key: Any, entry: Entity) -> None:
        """ Save the provided database entry to a file on uid/key inside the database directory. """
        entry_file_path = self.create_entry_file_path(uid, key)
        entry_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.write(entry, entry_file_path)