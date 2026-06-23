
import src.spectra.Tasks as Tasks

class Workbase:
    """
    This class defines a workbase object-- a database specializing in storing workflow design. It holds no actual instantiated workflows (i.e. provenance records).
    Rather, it maintains a record of defined tasks and workflows in order to provide referencing capability for other tasks and workflows. 
    TODO: 
        - port and refactor relevant code from Data.py's Workbase class, then deprecate that one
        - establish a singleton to serve as the central workbase-- i.e. the global workbase
    """

    def __init__(self):
        pass

    def addWorkflow(self, workflowID):
        """
        adds an empty referenceable workflow to the workbase. The workflowID can be an integer or string-- but must be unique. Attempting to add a non-unique workflow will raise an exception.
        """
        pass

    def addTask(self, workflowID, task: Tasks.Task):
        """
        adds a task to the workbase under the defined workflow. The task's input tasks are checked for existence within the workbase. A warning is printed if the input task does not yet exist.
        """

    def getTask(self, locator: tuple):
        """
        Returns the task referenced by the locator.
        
        Args:
            locator (tuple(workflowID, taskID)): the unique identifier of a task
        """
        wID, tID = locator # the locator tuple should be of the form (workflowID, taskID)