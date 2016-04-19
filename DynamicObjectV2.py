#!/usr/bin/python

import collections
import copy

# TODO: fix behaviour of update
# TODO: Support DynamicObjectV1
class Class(dict):
    def __init__(self, *args):
        self.__data__ = collections.OrderedDict()

        self.extendInternal(args)

    def __getattr__ (self, attr, special=False):
        if (special): return self.get(attr)
        if (attr[:2] == "__"): return self.get(attr)
        else: return self.getData(attr)

    def __setattr__(self, key, value):
        if (key[:2] == "__"):
            updObj = {}
            updObj[key] = value
            getattr(self, 'update', True)(updObj)
            return
        self.setData(key, value)

    def __getitem__ (self, attr, special=False):
        return self.getData(attr)

    def __setitem__(self, key, value):
        self.setData(key, value)

    def __isDynamicObjectV2__():
        pass

    def extend (self, *args):
        self.extendInternal(args)

    def extendInternal (self, args):
        if (len(args) == 1):
            try:
                isDynamicObjectV2 = getattr(args[0], '__isDynamicObjectV2__')
                if (isDynamicObjectV2): self.__data__.update(args[0].__data__)
                pass
            except AttributeError:
                self.__data__.update(args[0]) # assume dictionary
        else:
            toggle = False
            key = None
            for arg in args:
                if (not toggle): key = arg
                else: self.__data__[key] = arg
                toggle = not toggle

    def getData (self, item):
        try:
            value = self.__data__[item]
            return value
            pass
        except KeyError:
            pass

    def setData (self, item, value):
        self.__data__[item] = value

    def __delattr__(self, attr):
        try:
            if (attr[:2] == "__"): return self.__delitem__(attr)
            else: del self.__data__[attr]
            pass
        except AttributeError:
            pass

    def __delitem__(self, item):
        try:
            del self.__data__[item]
            pass
        except AttributeError:
            pass

    def __str__ (self):
        result = "{"
        for item in self.__data__.items():
            item = item[0]
            result += str(item) + ": " + str(self.__data__[item]) + ", "

        if (len(self.__data__.items()) > 0): result = result[:-2]
        result += "}"
        return str(result)

    def __to_json__(self):
        result = "{"
        for item in self.__data__.items():
            item = item[0]
            result += "\"" + str(item) + "\": " + str(self.__data__[item]) + ", "

        if (len(self.__data__.items()) > 0): result = result[:-2]
        result += "}"
        return str(result)

    def __reduce_ex__ (self, x):
        return self.__data__.__reduce_ex__(x)

    def __vars__(self):
        arr = []
        for field in self.__data__:
            arr.append(field)
        return arr