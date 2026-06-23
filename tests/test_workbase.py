import src.spectra.Tasks as Tasks
import src.spectra.Workbase as WorkBase
import pytest

def create_workbase():
    return WorkBase.Workbase()

def test_workbase_creation():
    WorkBase = create_workbase()
    assert WorkBase._WORKBASE == {}

def test_add_workflow():
    WorkBase = create_workbase()
    WorkBase.addWorkflow(0)
    assert WorkBase._WORKBASE[0] == {}
    with pytest.raises(Exception, match="already exists"):
        WorkBase.addWorkflow(0)

def test_add_task():
    WorkBase = create_workbase()
    WorkBase.addWorkflow(0)
    task1 = Tasks.Task(workflowID=0, taskID=1, dataInput=None)
    WorkBase.addTask(task1)
    assert WorkBase._WORKBASE[0][1] == task1
    with pytest.raises(Exception, match="doesn't exist"):
        task2 = Tasks.Task(workflowID=1, taskID=1, dataInput=None)
        WorkBase.addTask(task2)
    with pytest.raises(Exception, match="Task already exists"):
        task3 = Tasks.Task(workflowID=0, taskID=1, dataInput=None)
        WorkBase.addTask(task3)

def test_get_task():
    WorkBase = create_workbase()
    WorkBase.addWorkflow(0)
    task1 = Tasks.Task(workflowID=0, taskID=1, dataInput=None)
    WorkBase.addTask(task1)
    holder = WorkBase.getTask((0,1))
    assert task1 == holder

