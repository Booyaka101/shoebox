"""No-op stand-in for basicsr's arch registry (shoebox shim).

Vendored arch files keep their `@ARCH_REGISTRY.register()` decorators; shoebox
instantiates the classes directly, so registration only needs to pass the
class through unchanged.
"""


class ArchRegistry:
    def register(self):
        def _wrap(cls):
            return cls
        return _wrap


ARCH_REGISTRY = ArchRegistry()
