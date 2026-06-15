import pyledger
# from pyledger.simple import access2, access3

def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4

def test_access():
    assert pyledger.access() == "Accessing the simple module"

def test_access2():
    assert pyledger.access2() == "2"

def test_access3():
    assert pyledger.access3() == "3"
    