import Data
from test_workflow import createLinearWorkflow

def test_db_global_init():
    assert Data.GLOBAL_DB

def test_add_source():
    s1 = Data.GLOBAL_DB.addSource("d1")

def test_linear_dataflow():
    rawSrc = "raw data"
    wf = createLinearWorkflow(3)
    src = Data.GLOBAL_DB.addSource(rawSrc)
    Data.GLOBAL_WB.addWorkflow(wf)
    chain = Data.GLOBAL_DB.getDataChain(src)
    return chain

