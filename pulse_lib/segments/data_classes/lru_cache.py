
class LruCache:
    '''
    Least recently used cache.

    Args:
        max_size (int): maximum number of entries to cache.
    '''
    def __init__(self, max_size):
        self.max_size = max_size
        # all items in the cache
        self.items = dict()
        # linked list with least recently used entry at the first position.
        self.first = None
        self.last = None


    def __getitem__(self, key):
        '''
        Returns the cached item, or an empty cache entry when the item is not yet cached.
        '''
        if key in self.items:
            entry = self.items[key]
            # remove from linked list
            prev = entry.prev
            nxt = entry.nxt
            self._link(prev, nxt)
            # add to entry end
            self._append(entry)
        else:
            entry = _LruEntry(key)
            self.items[key] = entry
            self._append(entry)
            self._check_size()

        return entry


    def _link(self, prev, nxt):
        if prev is None:
            self.first = nxt
        else:
            prev.nxt = nxt

        if nxt is None:
            self.last = prev
        else:
            nxt.prev = prev


    def _append(self, entry):
        entry.nxt = None
        self._link(self.last, entry)
        self.last = entry


    def _check_size(self):
        if len(self.items) <= self.max_size:
            return
        # remove first entry
        first = self.first
        self.items.pop(first.key)
        self._link(None, first.nxt)


class _LruEntry:
    '''
    Entry in last recently used cache.
    '''
    def __init__(self, key):
        self.prev = None
        self.nxt = None
        self.key = key
        self.data = None
