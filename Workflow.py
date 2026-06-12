

from typing import Final
import Task
import random


MAX_WORFKLOWS: Final = 100
MIN_LEN_WORKFLOW: Final = 2
MAX_LEN_WORKFLOW: Final = 10



class Workflow:
    AVAILABLE_WIDS = set(range(1, MAX_WORFKLOWS + 1))
    
    # creates a workflow with unique ID and adds head task with taskID 0.
    def __init__(self):
        if not Workflow.AVAILABLE_WIDS:
            raise Exception("No more available workflow IDs.")
        else:
            self.WORKFLOW_ID: Final = Workflow._getUniqWorkflowID()
            self._tasks = []
            self._CURR_TIDS = []
            self._initializeTaskList()

    @classmethod
    def randomLinearWorkflow(cls):
        workflow = cls()
        workflowLen = random.randint(MIN_LEN_WORKFLOW, MAX_LEN_WORKFLOW)
        for _ in range(workflowLen):
            taskID = len(workflow.tasks)
            dataInput = tasks[-1]
            newTask = Task.Task(workflow.WORKFLOW_ID, taskID, dataInput)
            workflow.addTask(newTask)
        return workflow

    def addTask(self, task: Task.Task):
        if (task.taskID in self.CURR_TIDS and task.workflowID == self.WORKFLOW_ID):
            raise Exception(f"Task ID {task.taskID} already exists in workflow {self.WORKFLOW_ID}.")
        self.tasks.append(task)

    @property
    def tasks(self):
        return self._tasks
    
    def _getUniqWorkflowID():
        if not Workflow.AVAILABLE_WIDS:
            raise Exception("No more available workflow IDs.")
        else:
            wID = random.choice(list(Workflow.AVAILABLE_WIDS))
            Workflow.AVAILABLE_WIDS.remove(wID)
            return wID
    
    def _initializeTaskList(self):
        if len(self._tasks) > 0:
            raise Exception("Workflow already has tasks.")
        else:
            self._tasks.append(Task.TaskHead(self.WORKFLOW_ID))
            self._CURR_TIDS = list(self._tasks[0].taskID)

    def __str__(self):
        return f"W:{self.WORKFLOW_ID} Tasks: {[str(task.taskID) for task in self.tasks]}"
    
            

