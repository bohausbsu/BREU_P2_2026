import random

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







WORKFLOW_LEN_MIN = 1
WORKFLOW_LEN_MAX = 10
WORKFLOW_IDS = set([])
WORKFLOWS: dict[int, list[Task]] = {}

def randWorkflowLen(): 
    yield random.randint(WORKFLOW_LEN_MIN, WORKFLOW_LEN_MAX)

def genUniqWorkflowID():
    availableIDs = set(range(1,100)).difference(WORKFLOW_IDS)
    if not availableIDs:
        raise Exception("No more available workflow IDs.")
    else:
        wID = random.choice(list(availableIDs))
        WORKFLOW_IDS.add(wID)
        WORKFLOWS[wID] = [list[Task]]
        yield wID

def randTask(workflowID):
    taskID = len(WORKFLOWS[workflowID]) + 1
    task = Task(workflowID, taskID, True, max(0, taskID - 1)) # link to previous task as parent or none if first
    WORKFLOWS[workflowID].append(task)
    yield task


def randWorkflow():
    wID = genUniqWorkflowID()
    for _ in range(randWorkflowLen()):
        WORKFLOWS[wID].append(randTask(wID))
    yield (wID, WORKFLOWS[wID])
        
class TaskHead(Task):

    def __init__(self, workflowID):
        self._workflowID = workflowID
        self._taskID = 0
        self._isValid = True
    
    @property
    def workflowID(self):
        return self._workflowID
    
    @property
    def taskID(self):
        return self._taskID
    
    @property
    def isValid(self):
        return self._isValid