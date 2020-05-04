import utility_functions

def test_resize_dict():
    d = {2:5, 7:5, 1:4, 5:2}
    r = utility_functions.resize_dict(d, 3)
    assert d == {2: 5, 7: 5, 1: 4, 5: 2}
    assert r == {2: 5, 7: 5, 1: 4}

    r = utility_functions.resize_dict(d, 3)
    assert d == {2: 5, 7: 5, 1: 4, 5: 2}
    assert r == {2: 5, 7: 5, 1: 4}

    r = utility_functions.resize_dict(d, 0)
    assert d == {2: 5, 7: 5, 1: 4, 5: 2}
    assert r == {}
