
"""
This file provides factory "architect" methods for creating tasks in a workbase.
An architect is instantiated with a database and workbase, or defaults to the global singletons for each:
    GLOBAL_DB and GLOBAL_WB respectively
The architect provides methods to create tasks in reference to other tasks without needing to point to the exact Task objects in memory.
This is useful, as requiring all task objects to be in memory will quickly become expensive.

"""

