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


def test_chunks():
    d = {i: i for i in range(10)}
    maxsize = 3
    count = 0
    for index, portion in enumerate(utility_functions.chunks(d, maxsize)):
        for j in range(maxsize):
            if j < len(portion):
                assert portion[index * maxsize + j] == index * maxsize + j
                count += 1

    assert count == 10


