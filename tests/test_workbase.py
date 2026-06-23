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
    with pytest.raises(Exception, match = "already exists"):
        WorkBase.addWorkflow(0)

def test_add_task():
    WorkBase = create_workbase()
    WorkBase.addWorkflow(0)
    WorkBase.addTask(0)
    assert WorkBase._WORKBASE[0][0] == {}

