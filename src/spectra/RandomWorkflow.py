from src.spectra.Tasks import Task, TaskHead
import src.spectra.Workflow as Workflow
import random
from typing import Final

MIN_LEN_WORKFLOW: Final = 2
MAX_LEN_WORKFLOW: Final = 10

# class RandomWorkflow(Workflow.Workflow):

#     @classmethod
def randomLinearWorkflow():
    workflow = Workflow.Workflow()
    workflowLen = random.randint(MIN_LEN_WORKFLOW, MAX_LEN_WORKFLOW)
    for _ in range(workflowLen):
        taskID = len(workflow.tasks)
        dataInput = workflow.tasks[-1]
        newTask = Task(workflow.WORKFLOW_ID, taskID, dataInput)
        workflow.addTask(newTask)
    return workflow

    
def randomNonlinearWorkflow():
    workflow = Workflow.Workflow()
    workflowLen = random.randint(MIN_LEN_WORKFLOW, MAX_LEN_WORKFLOW)
    pass

def randLinearTaskGen(workflow: Workflow.Workflow):
    wID = workflow.WORKFLOW_ID
    tasks = workflow.tasks
    taskID = len(tasks)

    if taskID == 0: # if 0, this is head task
        task = TaskHead(wID)
    else:
        task = Task(wID, taskID, tasks[-1]) # link to previous task as parent

    yield task