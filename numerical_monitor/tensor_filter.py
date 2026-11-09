"""Extensible tensor filters, independent of any training framework."""

from collections import defaultdict


class FilterChain:
    """Filters return True to exclude a tensor from numerical statistics."""

    def __init__(self, filters=()):
        self.filters = list(filters)

    def should_skip(self, position, tensor, module, identity):
        for predicate in self.filters:
            if hasattr(predicate, "should_skip"):
                skip = predicate.should_skip(position, tensor, module, identity)
            else:
                skip = predicate(position, tensor, module, identity)
            if skip:
                return True
        return False


class FilterRegistry:
    """Process-local filter registry; group=None registers a global filter."""

    _filters = defaultdict(list)

    @classmethod
    def register(cls, predicate, group=None):
        cls._filters[group].append(predicate)
        return predicate

    @classmethod
    def get_chain(cls, group):
        return FilterChain(cls._filters[None] + cls._filters[group])

    @classmethod
    def clear(cls):
        cls._filters.clear()
