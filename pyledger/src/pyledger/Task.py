import random

import Workflow

# Represents a single task in a workflow.
# Has unique taskID within the workflow,
# a list of tasks that produce data inputs,
# and a validity flag (not yet implemented)
# A task should be uniquely identifiable through the combination
# of its workflowID and taskID.
class Task:

    def __init__(self, workflowID, taskID, dataInput: list[Task]):
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


def randLinearTaskGen(workflow: Workflow.Workflow):
    wID = workflow.WORKFLOW_ID
    tasks = workflow.tasks
    taskID = len(tasks)

    if taskID == 0: # if 0, this is head task
        task = TaskHead(wID)
    else:
        task = Task(wID, taskID, tasks[-1]) # link to previous task as parent

    yield task

        
class TaskHead(Task):

    def __init__(self, workflowID):
        self._workflowID = workflowID
        self._taskID = 0  