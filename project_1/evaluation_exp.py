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

    def sum(self,args):
        self.oper_list = ["+",args]
        return self.oper_list#("+", args[0] + args[1])
    
    def mul(self,args):
        self.oper_list = ["*",args]
        return self.oper_list#("*",args[0] * args[1])
    
    
source = """f_n(a,b)
            {return x * y + z*w;}"""

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
def lazy_evaluate(operation, operand1, operand2):
    if operation == "*":
        return str(operand1 + "*" + operand2)
    else:
        return str(operand1 + "+" + operand2)

def evaluation(expression):

    operation = expression[0]
    operand1 = expression[1]
    operand2 = expression[2]

    if isinstance(operand1,list):
        operand1 = evaluation(operand1)

    if isinstance(operand2,list):
        operand2 = evaluation(operand2)

    return lazy_evaluate(operation, operand1, operand2)

nested_list = ["+",["*","x","y"],["*","z","w"]]
#nested_list = ["+", "x",["*","z","w"]]

result = evaluation(nested_list)
print(result)
