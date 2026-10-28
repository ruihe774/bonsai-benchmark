import gguf, sys
print(gguf.__file__)
try:
    print([x for x in dir(gguf) if 'PTQ' in x.upper() or 'TYPE' in x.upper()])
except Exception as e:
    print(e)
for name in dir(gguf):
    obj = getattr(gguf, name)
    if 'GGMLQuantizationType' == name:
        print(list(obj))
