"""Candidate groups and bounded batches for indexed BAM extraction."""
from itertools import islice


def candidate_groups(sites, gap=10000, max_fetch_span=10000000):
    """Split sorted centers at large gaps or the fetch span limit, across strands."""
    if not isinstance(gap, int) or isinstance(gap, bool) or gap < 1:
        raise ValueError('Candidate group gap must be a positive integer')
    if type(max_fetch_span) is not int or max_fetch_span < 1:
        raise ValueError('max_fetch_span must be a positive integer')
    groups = []
    first = 0
    for index in range(1, len(sites)):
        distance = sites[index]['start'] - sites[index - 1]['start']
        if distance < 0:
            raise ValueError('Candidate sites must be sorted within a chromosome')
        span = sites[index]['start'] - sites[first]['start'] + 1
        if distance >= gap or span > max_fetch_span:
            groups.append((first, index))
            first = index
    if sites:
        groups.append((first, len(sites)))
    return groups


def candidate_group_batches(sites, groups, size, max_fetch_span):
    """Yield (first, stop, group_count) for one bounded, contiguous BAM fetch."""
    first = stop = count = 0
    for group_first, group_stop in groups:
        if count and (count == size or
                      sites[group_stop - 1]['start'] - sites[first]['start'] + 1 > max_fetch_span):
            yield first, stop, count
            count = 0
        if count == 0:
            first = group_first
        stop = group_stop
        count += 1
    if count:
        yield first, stop, count


def read_batches(reads, size):
    """Read at most size BAM records together, then release that batch."""
    iterator = iter(reads)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch
