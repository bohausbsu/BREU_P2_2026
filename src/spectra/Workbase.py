
import src.spectra.Tasks as Tasks
from typing import Final

class Workbase:
    """
    This class defines a workbase object-- a database specializing in storing workflow design. It holds no actual instantiated workflows (i.e. provenance records).
    Rather, it maintains a record of defined tasks and workflows in order to provide referencing capability for other tasks and workflows. 

    TODO: 
        - port and refactor relevant code from Data.py's Workbase class, then deprecate that one
        - establish a singleton to serve as the central workbase-- i.e. the global workbase
    """

    def __init__(self):
        self._WORKBASE: dict = dict()

    def addWorkflow(self, workflowID):
        """
        adds an empty referenceable workflow to the workbase. The workflowID can be an integer or string-- but must be unique. Attempting to add a non-unique workflow will raise an exception.
        """
        if workflowID in self._WORKBASE:
            raise Exception(f"Workflow {workflowID} already exists in workbase")
        self._WORKBASE[workflowID] = dict()

    def addTask(self, task: Tasks.Task):
        """
        adds a task to the workbase under the defined workflow. The task's input tasks are checked for existence within the workbase. A warning is printed if the input task does not yet exist.
        """
        workflowID = task.workflowID
        if workflowID not in self._WORKBASE:
            raise Exception(f"Workflow {workflowID} doesn't exist in workbase")
        
        for inputUUID in task.inputTaskLocators:
            inWID, inTID = inputUUID
            wf = self._WORKBASE.get(inWID, None)
            if wf is None or inTID not in wf:
                print(f"Warning: input task {inputUUID} for task {task.locator} is not yet registered")
        self._WORKBASE[workflowID][task.taskID] = task

    def getTask(self, locator: tuple) -> Tasks.Task:
        """
        Returns the task referenced by the locator.
        
        Args:
            locator (tuple(workflowID, taskID)): the unique identifier of a task
        """
        wID, tID = locator # the locator tuple should be of the form (workflowID, taskID)
        try:
            return self._WORKBASE[wID][tID]
        except KeyError:
            return None
        
GLOBAL_WB: Final = Workbase()