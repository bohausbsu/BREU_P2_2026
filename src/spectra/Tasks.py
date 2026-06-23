import random
import src.spectra.Workflow as Workflow
import deprecated.spectra.Data as Data
from typing import Final

from deprecated.spectra.ProvenanceRecord import Entry

class Task:
    """
    Represents a single task within a workflow. The task holds a list of workbase pointers to other tasks that serve as data inputs for this task. The task operates on the data inputs to produce 
    its own data output, referenceable by latter tasks in a workbase. 
    TODO:
        - refactor to utilize locator style identification (wID, tID) only
        - separate blueprint code from provenance code
    """

    def __init__(self, workflowID, taskID, dataInput: list[Task] | Task | list[tuple[int, int]] | tuple[int, int] | None):
        
        # locator fields
        self.workflowID: Final = workflowID
        self.taskID: Final = taskID
        self.locator: Final = (workflowID, taskID)
        
        # data processing fields
        self.inputTaskLocators: Final = []
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
        for taskUUID in self.inputTaskLocators:
            tasks.append(workbase.getTask(taskUUID))

        # ensure data values exist in specified DB
        for task in tasks:
            if task not in datachain:
                task.compute(datasourceID)




    def __eq__(self, other):
        if isinstance(other, Task):
            return self.locator == other.locator
        elif isinstance(other, tuple) and len(other) == 2:
            return self.locator == other
        else:
            return False

    def _addTaskInputs(self, src: Task | list[Task] | tuple[int, int] | list[tuple[int, int]]):
        uuids = []
        if isinstance(src, Task):
            uuids.append(src.locator)
        elif isinstance(src, tuple):
            if len(src) == 2 and all(isinstance(e, int) for e in src):
                uuids.append(src)
        elif isinstance(src, list):
            for e in src:
                if isinstance(e, Task):
                    uuids.append(e.locator)
                elif isinstance(e, tuple) and len(src) == 2 and all(isinstance(i, int) for i in e):
                    uuids.append(e)
        self.inputTaskLocators.extend(uuids)
                    
        # if isinstance(src, tuple) or (isinstance(src, list) and all(isinstance(item, tuple) for item in src)):
        #     # adds from tuples
        #     uuids = [id for id in src] if isinstance(src, list) else src
        # elif isinstance(src, Task) or (isinstance(src, list) and all(isinstance(item, Task) for item in src)):
        #     uuids = [task.uuID for task in src] if isinstance(src, list) else src.uuID

        
class TaskHead(Task):

    def __init__(self, workflowID):
        super().__init__(workflowID, 0, None)


    