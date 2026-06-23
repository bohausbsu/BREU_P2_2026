
"""
The Controllers module defines controllers to interact with the various components of spectra.
"""

# """
# NOT CURRENT FUNCTIONALITY! IGNORE COMMENT! VVVVVV
# This file provides factory "architect" methods for creating tasks in a workbase.
# An architect is instantiated with a database and workbase, or defaults to the global singletons for each:
#     GLOBAL_DB and GLOBAL_WB respectively
# The architect provides methods to create tasks in reference to other tasks without needing to point to the exact Task objects in memory.
# This is useful, as requiring all task objects to be in memory will quickly become expensive.

# """

import src.spectra.Workbase as wb

class WorkbaseController:
    """
    The WorkbaseController assists in the CRUD cycle of tasks and workflows. It attaches to a workbase and allows simple task reference to other tasks within the same reference workbase.

    TODO:
        - implement some factory methods for tasks and workflows based on the "factory" design pattern
        - enable CRUD functionality for workflows and tasks
        - establish a querying system that doesn't require direct object reference
            - i.e. setting the data input for some task should take a tuple (wID, tID) rather than requiring the task object itself
        - add unit tests in the test directory relating to the workbase architect
    """

    def __init__(self, workbase: wb.Workbase):
        
        pass

