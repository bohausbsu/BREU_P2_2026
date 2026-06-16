import Tasks

# Task Section

# Proven methods
def createTask(workflowID, taskID, inputTaskUUIDs):
    return Tasks.Task(workflowID, taskID, inputTaskUUIDs)

def createNextTask(task):
    return Tasks.Task(task.workflowID, task.taskID + 1, task)

# Task Tests

def test_task_creation():
    wID = 12
    tID = 1
    task = createTask(wID, tID, None)
    assert task.workflowID == wID
    assert task.taskID == tID
    assert task.inputTaskUUIDs == []

def test_task_chaining():
    task1 = createTask(1, 1, None)
    task2 = createNextTask(task1)
    assert task2.workflowID == task1.workflowID
    assert task2.taskID == task1.taskID + 1
    assert task2.inputTaskUUIDs == [task1.uuID]


# Workflow Section
import Workflow

# Proven methods

def createLinearWorkflow(length: int):
    # constr auto creates the head
    workflow = Workflow.Workflow()
    prev = workflow.headTask
    for _ in range(length-1):
        task = createNextTask(prev)
        prev = task
        workflow.addTask(task)
    return workflow

# Workflow Tests

def test_linear_workflow():

    LENGTH = 10
    workflow = createLinearWorkflow(LENGTH)

    tasks = iter(workflow.tasks)

    # valid head
    prev = next(tasks)
    assert prev == workflow.headTask

    while True:
        try:
            task = next(tasks)
            
            # only one data input
            assert len(task.inputTaskUUIDs) == 1

            # ensure prev lead to this one
            assert prev.uuID in task.inputTaskUUIDs

            prev = task
        except StopIteration:
            break

def test_tuple_workflow():
    
    edges = [
        (0,1),
        (0,2),
        (1,2),
        (2,3),
        (3,4),
        (1,5),
        (4,5)
    ]

    workflow = Workflow.Workflow.workflowFromTuples(edges)
    tasks = iter(workflow.tasks)
    
    t0 = next(tasks)
    t1 = next(tasks)
    t2 = next(tasks)
    t3 = next(tasks)
    t4 = next(tasks)
    t5 = next(tasks)

    # t0
    assert t0 == workflow.headTask
    assert t0.inputTaskUUIDs == []

    #t1
    assert t0.uuID in t1.inputTaskUUIDs
    assert len(t1.inputTaskUUIDs) == 1

    #t2
    assert t1.uuID in t2.inputTaskUUIDs
    assert t0.uuID in t2.inputTaskUUIDs
    assert len(t2.inputTaskUUIDs) == 2

    #t3
    assert t2.uuID in t3.inputTaskUUIDs
    assert len(t3.inputTaskUUIDs) == 1

    #t4
    assert t3.uuID in t4.inputTaskUUIDs
    assert len(t4.inputTaskUUIDs) == 1
    
    #t5
    assert t4.uuID in t5.inputTaskUUIDs
    assert t1.uuID in t5.inputTaskUUIDs
    assert len(t5.inputTaskUUIDs) == 2
