# import bm25s

# print(bm25s.tokenize("what is rag?"))

def micro(func):
    def fun():
        return func() * 2
    return fun

@micro
def func():
    return (2)

res = func()
print(res)