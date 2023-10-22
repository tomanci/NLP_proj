import lark

class FuncionDefinition(lark.Transformer):
    
    def __init__(self):
        self.param_list = []
        self.oper_list = [] 
    """
    def fun_definition(self, parameter, operation):
        self.param_list = [parameter]
        self.oper_list = [operation]
        x = self.oper_list[0]
        for i in x:
            print(i) 
        return self.oper_list
    """

    def fun_definition(self,parameter):
        self.param_list = [parameter]
        return self.param_list
    
    def oper_definition(self, operation):
        self.oper_list = [operation]
        print(self.oper_list[0])
        return self.oper_list
    #NUMBER = float  
    
    def __check_number_parameters(self):
        if len(self.param_list[0]) != len(self.oper_list[0]):
            raise ValueError ("the number of parameters does not correspond!!!")

    def sum(self,*args):
        print(args)
        return args[0]
    
    def mul(self,args):
        return args[0] * args[1]
    
    
source = """f_n(x,y)
            {return x_1 + y_1*z;}"""

#source = "f_n(x,y)"
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