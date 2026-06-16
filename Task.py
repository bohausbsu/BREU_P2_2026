import random

import Workflow
from ProvenanceRecord import Entry

# Represents a single task in a workflow.
# Has unique taskID within the workflow,
# a list of tasks that produce data inputs,
# and a validity flag (not yet implemented)
# A task should be uniquely identifiable through the combination
# of its workflowID and taskID.
class Task:

    def __init__(self, workflowID, taskID, dataInput: list[Task] | Task | None):
        self._workflowID = workflowID
        self._taskID = taskID
        self._dataInput = []
        self._dataInput.append(dataInput)
    
    @property
    def workflowID(self):
        return self._workflowID
    
    @property
    def taskID(self):
        return self._taskID
    
    @property
    def dataInput(self):
        return self._dataInput
    
    def addDataInput(self, inputTask):
        self._dataInput.append(inputTask)

    def compute(self, dataProductsIn: set[Entry]):



        
class TaskHead(Task):

    def __init__(self, workflowID):
        self._workflowID = workflowID
        self._taskID = 0  

    