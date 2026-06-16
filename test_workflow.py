import Workflow, Task


# Proven methods
def createTask(workflowID, taskID, dataInput):
    return Task.Task(workflowID, taskID, dataInput)

def createNextTask(task):
    return Task.Task(task.workflowID, task.taskID + 1, task)

# Task Specific

def test_task_creation():
    wID = 12
    tID = 1
    task = createTask(wID, tID, list([]))
    assert task.workflowID == wID
    assert task.taskID == tID
    assert task.dataInput == [[]]

def test_task_chaining():
    task1 = createTask(1, 1, list([]))
    task2 = createNextTask(task1)
    assert task2.workflowID == task1.workflowID
    assert task2.taskID == task1.taskID + 1
    assert task2.dataInput == [task1]

