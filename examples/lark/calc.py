import lark, math

class EvalExpr(lark.Transformer):

    def __init__(self):
        self.vars = {}
        self.fncts = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp}

    def assignment(self,args):
        self.vars[args[0]]=args[1]
        return args[1]

    def var(self,args):
        try:
            return self.vars[args[0]]
        except KeyError:
            raise Exception("Variable %s not defined" % args[0])

    def function(self,args):
        try:
            return self.fncts[args[0]](args[1])
        # fncts is a dictionary. thus with self.fncts[args[0]] is a call to the dictionary with the key is args[0].
        # then args[1] is the number which as to be calculated using the function
        except KeyError:
            raise Exception("Function %s not defined" % args[0])

    NUMBER = float
    def sum(self,args):
        return args[0]+args[1]
    def sub(self,args):
        return args[0]-args[1]
    def mul(self,args):
        return args[0]*args[1]
    def div(self,args):
        return args[0]/args[1]
    def neg(self,args):
        return -args[0]

calc_parser = lark.Lark.open("calc.lark",rel_to=__file__,start="stmt",parser='lalr',transformer=EvalExpr())

while True:
    try:
         line = input('> ')
    except EOFError:
        break
    try:
        print(calc_parser.parse(line))
    except lark.UnexpectedToken as error:
        print(error.get_context(line),end='')
        print("Unexpected token %s " % error.token)
    except lark.UnexpectedInput as error:
        print("Parser error")
        print(error.get_context(line))
    except Exception as error:
        print(error)
