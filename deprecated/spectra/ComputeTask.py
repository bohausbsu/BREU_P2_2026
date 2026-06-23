from abc import ABC, abstractmethod

class AbstractTask(ABC):
    """
    Wraps a runnable action alongside data pointers to prerequisite tasks or data entry points.

    Tasks are able to chain together by directing one's outputs to another's inputs.
    These chains are referred to as workflows. 
    """

class _HeadTask(AbstractTask):
    """
    The HeadTask acts as the head of a workflow. A HeadTask need not have only one correlated workflow, and neither does a workflow need only one head task.
    
    The HeadTask interacts with an external data source. That is, if a workflow
    takes input from any non-Task data source, it must be a HeadTask. It is not required that a head task
    interact with an exterior data source-- it may very well source from another task. However, it is the only
    type of task capable of taking in external data. The HeadTask may only take input from one external data source. Multiple
    internal data sources may be used, however. This limitation is due to external sources not being compliant to internal mapping requirements.
    Furthermore, the HeadTask is meant to denote the start of a distinct workflow. 
    """

class _ComputeTask(AbstractTask):
    """
    The ComputeTask is used for internal data processing within a workflow. It may take data inputs from other Tasks. 
    """

def createWorkflowHeadTask(dataInput):
    """Creates a HeadTask"""