
class ContextUtils(object):

    _context_map = {}

    @staticmethod
    def register(name, filed, value):
        if name in ContextUtils._context_map:
            raise TypeError("Duplicate operation named {}".format(name))

        ContextUtils._context_map[name] = value

    @staticmethod
    def contains(name):
        return name in ContextUtils._context_map

    @staticmethod
    def list():
        return list(ContextUtils._context_map.keys())


    @staticmethod
    def get(name):
        if name in ContextUtils._context_map:
            return ContextUtils._context_map[name]
        else:
            raise ValueError("context not contain named {}".format(name))
