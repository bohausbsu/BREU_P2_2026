import random
import Workflow
import Data
from typing import Final

from ProvenanceRecord import Entry

# Represents a single task in a workflow.
# Has unique taskID within the workflow,
# a list of tasks that produce data inputs,
# and a validity flag (not yet implemented)
# A task should be uniquely identifiable through the combination
# of its workflowID and taskID.
class Task:

    def __init__(self, workflowID, taskID, dataInput: list[Task] | Task | list[tuple[int, int]] | tuple[int, int] | None):
        
        # locator fields
        self.workflowID: Final = workflowID
        self.taskID: Final = taskID
        self.uuID: Final = (workflowID, taskID)
        
        # data processing fields
        self.inputTaskUUIDs: Final = []
        # self.inputTaskData: Final = dict()
        
        # parsing data in
        if dataInput is not None:
            self._addTaskInputs(dataInput)
    
    def addDataInput(self, inputTask: Task | tuple[int, int]):
        self._addTaskInputs(inputTask)


    # TODO: focus on local operation only.
    # variable dict will hold all values needed w/in compute.
    # the dict is provided from the datachain.
    # this is an isolated computational env
    def compute(self, variableDict):
        
        # simple passthrough for now
        memory = []
        for (var, val) in variableDict:
            memory.append((var, val))
        return memory
        # # queries from db for available data derived from specified raws
        # datachain = db.getDataChain(dataSourceID)

        # # placeholder data passthrough
        # data = []
        # for taskUUID in self.inputTaskUUIDs:
        #     data.append(datachain.query(taskUUID))
        # datachain.
        # return data
    
    # TODO: move logic to datachain
    # DEPRECATE THIS!
    def _computeDependencyValues(self, datasourceID, workbase: Data.Workbase, database: Data.Database = Data.GLOBAL_DB):
        
        # get requested data chains
        datachain = database.getDataChain(datasourceID)
        
        # get prereq tasks
        tasks = []
        for taskUUID in self.inputTaskUUIDs:
            tasks.append(workbase.getTask(taskUUID))

        # ensure data values exist in specified DB
        for task in tasks:
            if task not in datachain:
                task.compute(datasourceID)




    def __eq__(self, other):
        if isinstance(other, Task):
            return self.uuID == other.uuID
        elif isinstance(other, tuple) and len(other) == 2:
            return self.uuID == other
        else:
            return False

    def _addTaskInputs(self, src: Task | list[Task] | tuple[int, int] | list[tuple[int, int]]):
        uuids = []
        if isinstance(src, Task):
            uuids.append(src.uuID)
        elif isinstance(src, tuple):
            if len(src) == 2 and all(isinstance(e, int) for e in src):
                uuids.append(src)
        elif isinstance(src, list):
            for e in src:
                if isinstance(e, Task):
                    uuids.append(e.uuID)
                elif isinstance(e, tuple) and len(src) == 2 and all(isinstance(i, int) for i in e):
                    uuids.append(e)
        self.inputTaskUUIDs.extend(uuids)
                    
        # if isinstance(src, tuple) or (isinstance(src, list) and all(isinstance(item, tuple) for item in src)):
        #     # adds from tuples
        #     uuids = [id for id in src] if isinstance(src, list) else src
        # elif isinstance(src, Task) or (isinstance(src, list) and all(isinstance(item, Task) for item in src)):
        #     uuids = [task.uuID for task in src] if isinstance(src, list) else src.uuID

        
class TaskHead(Task):

    def __init__(self, workflowID):
        super().__init__(workflowID, 0, None)


    