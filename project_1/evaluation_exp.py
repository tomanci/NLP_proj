import lark

class FuncionDefinition(lark.Transformer):
    def __init__(self):
        self.param_list = []
        self.oper_list = [] 

    def fun_definition(self,operation):
        self.param_list = [operation]
        self.oper_list = [operation]
        return self.oper_list

    #NUMBER = float

    def sum(self,*args):
        print(args)
        return args[0]
    
    def mul(self,args):
        return args[0] * args[1]
    
    
source = """sum(x,y)
            {return x + y;}"""


print(">> Input string: ",source)

function_parser = lark.Lark.open("functions_rule.lark", rel_to = __file__, start = "function_name",
                             parser = 'lalr', transformer = FuncionDefinition())
result = function_parser.parse(source)
print("\n*** Parse tree pretty print\n", result.pretty())

"""
def main():
    source = input("> ")
    function_parser = lark.Lark.open("functions_rule.lark", rel_to = __file__, start = "function_name",
                             parser = 'lalr', transformer = FuncionDefinition())
    result = function_parser.parse(source)
    print("\n*** Parse tree pretty print\n", result.pretty())


if __name__ == '__main__':
    main()

"""