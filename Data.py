from typing import Final
from uuid import uuid4
from collections import deque
# import Workflow
# import Tasks

# records and provides all uses of a raw data source.
# Any new data input requires a new data chain
# tasks done on data are recorded on the corresponding datachain, as a provenance record.
    # provenance records should be quick access, as all permutations are recorded within the same object
# this is done in combination with the wID and tID. The most recently computed data is offered for further work.
    # previous data is kept to imitate block chain.

class Database:

    def __init__(self):
        self._DATABASE = dict()

    @property
    def DATA(self):
        return self._DATABASE

    def addSource(self, data):
            dataID = uuid4()
            self._DATABASE[dataID] = _DataChain(data, dataID)
            return dataID
    
    def getDataChain(self, dataID):
        return self._DATABASE[dataID]
    
    def query(self, dataID, taskUUID: tuple[int, int]):
        # wID, tID = taskUUID
        datachain = self.getDataChain(dataID)
        return datachain.query(taskUUID)

GLOBAL_DB: Final = Database()

class Workbase:
    import Workflow

    def __init__(self):
        self._WORKBASE = dict()

    @property
    def WORKFLOWS(self):
        return self._WORKBASE
    
    def addWorkflow(self, workflow: Workflow.Workflow):
        wID = workflow.WORKFLOW_ID

        # create workflow key
        if wID in self.WORKFLOWS:
            # error, already exists
            return
        self._WORKBASE[wID] = dict()
        
        # populate subdict with tasks
        wDict = self._WORKBASE[wID]
        for task in workflow.tasks:
            wDict[task.uuID] = task


    def getTask(self, taskUUID: tuple[int, int], db: Database = GLOBAL_DB):
        wID, tID = taskUUID
        return self._WORKBASE[wID][tID]
    
    def runWorkflow(self, workflowID, dataSourceID, db: Database = GLOBAL_DB):
        workflow = self._WORKBASE[workflowID]
        tail = workflow.tailTask
         #TODO continue compute here


GLOBAL_WB: Final = Workbase()

class _DataChain:

    def __init__(self, data, dataID):
        self._RAW_DATA: Final = data
        self._DATA_ID: Final = dataID
        self._WORKFLOWS: Final = dict()



    def runComputes(self, taskUUID, dataSourceID, db: Database = GLOBAL_DB, wb: Workbase = GLOBAL_WB):
        from Tasks import TaskHead

        wID, tID = taskUUID
        
        # get or create locator dicts
        wDB = self._WORKFLOWS.get(wID, None)
        if wDB is None:
            self._WORKFLOWS[wID] = dict()
            wDB = self._WORKFLOWS[wID]
        
        tDB = self._WORKFLOWS.get(tID, None)
        if tDB is None:
            wDB[tID] = deque() # better time access for LIFO access
            tDB = wDB[tID]
        
       
        task = wb.getTask(taskUUID)
        datachain = db.getDataChain(dataSourceID)
        inputVars = dict()

        # subvert if head task
        if (isinstance(task, TaskHead)):
            return {taskUUID: self._RAW_DATA}


        # get data in vals, or computer them if they DNE
        for inpt in task.taskInputUUIDs:
            data = datachain.query(inpt.uuID)
            if data is None:
                self.runComputes(inpt.taskUUID, dataSourceID, db, wb)
                data = datachain.query(inpt.uuID)
            inputVars[inpt.uuID] = data
        
        # add this task's computation to datachain
        tDB.append(task.compute(inputVars))

    def query(self, taskUUID):
        wID, tID = taskUUID
        try:
            wDB = self._WORKFLOWS[wID]
            tDB = wDB[tID]
            return tDB.peek()
        except NameError:
            return None
        
    def __contains__(self, item):
        import Tasks
        taskUUID: tuple[int, int]
        if isinstance(item, Tasks.Task):
            taskUUID = item.uuID
        elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(e, int) for e in item):
            taskUUID = item
        else:
            return False
        
        wID, tID = taskUUID

        wf = self._WORKFLOWS.get(wID, None)
        if wf is not None:
            task = wf.get(tID, None)
            return task is not None
        return False