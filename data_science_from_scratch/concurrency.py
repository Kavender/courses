from functools import partial
from collections import defaultdict

## MapReduce
def map_reduce(inputs, mapper, reducer):
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output for key, value in collector.items()
            for output in reducer(key, value)]


# generic reducer function
def reduce_value_using(aggregation_fn, key, value):
    """reduce a key-value pair by applying aggregation_fn to the value"""
    yield (key, aggregation_fn(value))

def value_reducer(aggregation_fn):
    """turns a function (value -> output) into a reducer that maps (key, values) - > (key, output)
    `sum_reducer = value_reducer(sum)
     max_reducer = value_reducer(max)
     count_distinct_reducer = value_reducer(lambda values: len(set(values)))``
    """
    return partial(reduce_value_using, aggregation_fn)
