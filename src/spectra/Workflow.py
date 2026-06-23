

from typing import Final
import src.spectra.Tasks as Tasks
import random
# Task = Tasks.Task
# TaskHead = Tasks.TaskHead

MAX_WORFKLOWS: Final = 100
MIN_LEN_WORKFLOW: Final = 2
MAX_LEN_WORKFLOW: Final = 10



class Workflow:
    AVAILABLE_WIDS = set(range(1, MAX_WORFKLOWS + 1))

    @property
    def tasks(self):
        return self._tasks
    
    # creates a workflow with unique ID and adds head task with taskID 0.
    def __init__(self):
        if not Workflow.AVAILABLE_WIDS:
            raise Exception("No more available workflow IDs.")
        else:
            self.WORKFLOW_ID: Final = self._getUniqWorkflowID()
            self._tasks: list[Tasks.Task] = []
            self._CURR_TIDS: list[int] = []
            self._initializeTaskList()

    @classmethod
    def randomLinearWorkflow(cls):
        workflow = cls()
        workflowLen = random.randint(MIN_LEN_WORKFLOW, MAX_LEN_WORKFLOW)
        for _ in range(workflowLen):
            taskID = len(workflow.tasks)
            dataInput = workflow._tasks[-1]
            newTask = Tasks.Task(workflow.WORKFLOW_ID, taskID, dataInput)
            workflow.addTask(newTask)
        return workflow

    @classmethod
    def workflowFromTuples(cls, tasks: list[tuple[int, int]]):
        workflow = cls()
        seenIDs = set()

        # # create head task with id 0
        # workflow.addTask(Task.TaskHead(workflow.WORKFLOW_ID)) !! Handled by _initializeTaskList in constr
        seenIDs.add(0) # must still account for initialized head

        for inputID, taskID in tasks:
            
            # local cache for quick existence checking
            if inputID not in seenIDs:
                workflow.addTask(inputID)
                seenIDs.add(inputID)
            if taskID not in seenIDs:
                workflow.addTask(taskID)
                seenIDs.add(taskID)

            # workflow should have task after going through check, so no adtl check needed

            target = workflow.tasks[taskID]
            inpt = workflow.tasks[inputID]
            target.addDataInput(inpt)
        
        return workflow
    
    @property
    def headTask(self):
        return self._tasks[0]
                
    @property
    def tailTask(self):
        return self._tasks[-1]


    def addTask(self, task: Tasks.Task | int):
        if (self.hasTask(task)):
            raise Exception(f"Task ID {task.taskID if isinstance(task, Tasks.Task) else task} already exists in workflow {self.WORKFLOW_ID}.")
        else:
            if isinstance(task, int):
                task = Tasks.Task(self.WORKFLOW_ID, task, None)
            self._tasks.append(task)
    



    def hasTask(self, task: Tasks.Task | int):
        if isinstance(task, Tasks.Task):
            tId = task.taskID
        else:
            tId = task
        return tId in self._CURR_TIDS


    
    def _getUniqWorkflowID(self):
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
            self._tasks.append(Tasks.TaskHead(self.WORKFLOW_ID))
            self._CURR_TIDS.append(self._tasks[0].taskID)

    def __str__(self):
        return f"W:{self.WORKFLOW_ID} Tasks: {[str(task.taskID) for task in self.tasks]}"
    
            
    def visualize(self):
        repr = f"""---\ntitle: Workflow {self.WORKFLOW_ID}\n---\n
        flowchart RL

        """
        for task in self.tasks:
            # establish all nodes 
            repr += f"\t{task.taskID}\n"

        for task in self.tasks:
            # now draw all dependencies
            if task.inputTaskUUIDs is not None:
                for inpt in task.inputTaskUUIDs:
                    if inpt is not None:
                        repr += f"\t{task.uuID[1]} --> {inpt[1]}\n"
        return repr
    

