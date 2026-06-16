
# import ProvenanceRecord
# import Tasks
# from typing import Final
# import Data

# # holds computational results (entries) of which PR will query
# class _Environment:
#     def __init__(self):
#         self._memory = set(set([ProvenanceRecord.Entry]))

#     def __getitem__(self, key):
#         if isinstance(key, tuple):



#     def postCompute(self, task: Tasks.Task, user: str, data):
#         wID = task.workflowID
#         tID = task.taskID
#         Data.DATABASE.post(task.uuID, dataID, data) # cannot allow data input later on. Can lie here


#     @property
#     def workflow(self, wID: int):
#         return self.

# # singleton environment object
# ENVIRONMENT: Final = _Environment()

# # access may be of form ((wID, tID), entryID)