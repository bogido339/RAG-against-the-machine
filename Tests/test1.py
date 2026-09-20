import fire


# def hello(name='world!'):
#     return "hello %s! " % name

# if __name__ == "__main__":
#     fire.Fire(hello)


class SumpleCalc:
    def double(self, number, counter):
        return number * counter
    
    def miltupel(self, number, counter):
        if counter > 0:
            return number / counter
        return "math error"
    
if __name__ == "__main__":
    fire.Fire(SumpleCalc)
