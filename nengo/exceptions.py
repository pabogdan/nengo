class NengoException(Exception):
    """Base class for Nengo exceptions.

    NengoException instances should not be created; this base class exists so
    that all exceptions raised by Nengo can be caught in a try / except block.
    """
    pass


class ObsoleteError(NengoException):
    """A feature that has been removed in a backwards-incompatible way."""

    def __init__(self, msg, since=None, url=None):
        self.since = since
        self.url = url
        super(ObsoleteError, self).__init__(msg)

    def __str__(self):
        return "Obsolete%s: %s%s" % (
            "" if self.since is None else " since %s" % self.since,
            super(ObsoleteError, self).__str__(),
            "\nFor more information, please visit %s" % self.url
            if self.url is not None else "")


class ValidationError(NengoException, ValueError):
    """A ValueError encountered during validation of a parameter."""

    def __init__(self, klass, attr, *args):
        self.klass = klass
        self.attr = attr
        super(ValidationError, self).__init__(*args)

    def ___str__(self):
        return "Validation error when setting %s.%s: %s" % (
            self.klass.__name__,
            self.attr,
            super(ValidationError, self).__str__())
