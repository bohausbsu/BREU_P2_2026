import Tasks
from typing import Final

# a single record. Should be uniquely identifiable. 
class Entry:

    def __init__(self, task: Tasks.Task, user: str, dataIn):
        self._task: Final = task
        self._user: Final = user
        self._dataIn: Final = dataIn

    @property
    def dataOut(self):
        pass
# Represents the 
class Entries:
    pass