import lark

class FuncionDefinition(lark.Transformer):
    
    def __init__(self):
        self.param_list = []
        self.oper_list = []
        self.parm_call_list = []
        self.def_fun_name = ""
        self.call_fun_name = ""
   
    def fun_initialization_def(self,values):
        self.def_fun_name = values[0]
        self.param_list = values[1]
        self.oper_list = values[2]
        self._check_name(self.oper_list,self.param_list)
        print(self.oper_list)
        
    def fun_initialization_call(self,values):
        self.call_fun_name = values[0]
        self.parm_call_list = values[1]
        self._check_number_parameters(self.parm_call_list, self.param_list, self.oper_list)
        self._check_name_function(self.def_fun_name, self.call_fun_name)
        self.evaluation(self.oper_list, self.param_list ,self.parm_call_list)
        #operation = values[1][0]
        operand1 = values[1][0][0]
        operand2 = values[1][0][1]
        print(int(operand1)+int(operand2))
        return operand1+operand2

    def fun_definition(self,parameter):
        self.param_list = [parameter]
        return self.param_list
    
    def fun_call(self,parameter):
        self.parm_call_list = [parameter]
        return self.parm_call_list
    
    def _check_number_parameters(self,list_param_def,list_parm_call, oper_list):

        flat_oper_list = self.flat(oper_list) 
        if len(list_param_def[0]) != len(list_parm_call[0]):
            raise ValueError ("the number of input parameters does not correspond!!!")
        
        if len(list_parm_call[0]) != len(flat_oper_list) or len(list_param_def[0]) != len(flat_oper_list):
            raise ValueError ("the number of INPUT paramters and OUTPUT parameters do not correspond!!!")
               
    def _check_name_function(self, string_1, string_2):
        if string_1 != string_2:
            raise ValueError ("the names of the function do not correspond!!!")
        
    def sum(self,args):
        self.oper_list = ["+",args]
        return self.oper_list#("+", args[0] + args[1])
    
    def mul(self,args):
        self.oper_list = ["*",args]
        return self.oper_list#("*",args[0] * args[1])
    
    def flat(self,l):
        final_list = []
        for item in l:
            if isinstance(item,list):
                final_list.extend(self.flat(item))
            elif item not in ["*","+"]:
                final_list.append(item)

        return final_list
    
    def evaluation(self, expression, param_list, parm_call_list):
        operation = expression[0]
    
        if len(expression[1:]) == 1:
            operands = expression[1]
        else:
            operands = expression[1:]

        if isinstance(operands, list):
            operands = [evaluation(operand, param_list ,parm_call_list) 
                        if isinstance(operand, list) else operand for operand in operands]

        return self.lazy_evaluate(operation, operands, self.param_list ,self.parm_call_list)

    def lazy_evaluate(self, operation, operands, param_list ,parm_call_list):

        operand1 = self._check_correspondence(operands[0], param_list, parm_call_list)
        operand2 = self._check_correspondence(operands[1], param_list, parm_call_list)

        if operation == "*":
            return operand1 * operand2
        else:
            return operand1 + operand2
    
    def _check_correspondence(self, operand, param_list, param_call_list):
        counter = 0
        for item in param_list:
            counter = counter + 1 
            if item == operand:
                return param_call_list[counter]
                
    def _check_name(self, oper_list, param_list):

        for item in param_list:
            bol = False
            for itm in oper_list:
                if item == itm:
                    bol = True
            if not bol:
                raise KeyError("Value names and parameters do not correspond!!!") 
                break                  
            
    
def_function = """f_n(x,y)
            {return x + y;}"""
            
call_function = """f_n(1,3)"""

#source = "f_n(x,y)"
#print(">> Input string: ",def_function)

function_parser = lark.Lark.open("functions_rule.lark", rel_to = __file__, start = "function_name",
                             parser = 'lalr', transformer = FuncionDefinition())

result_function = function_parser.parse(def_function)
result_function_call = function_parser.parse(call_function)

#print("\n*** Parse tree pretty print\n", result_function.pretty())
#print("\n*** Parse tree pretty print\n", result_function_call.pretty())


def lazy_evaluate(operation, operands):
    operand1 = operands[0]
    operand2 = operands[1]

    if operation == "*":
        return str(operand1 + "*" + operand2)
    else:
        return str(operand1 + "+" + operand2)

def evaluation(expression):
    operation = expression[0]
    
    if len(expression[1:]) == 1:
        operands = expression[1]
    else:
        operands = expression[1:]

    if isinstance(operands, list):
        operands = [evaluation(operand) if isinstance(operand, list) else operand for operand in operands]

    return lazy_evaluate(operation, operands)

            
nested_list = [ "+",["*", ["x","y"] ],[ "*", ["z","w"] ] ] #ok
#nested_list = ["+", ["x",["*",["z","w"]]]] #ok
#nested_list = ["+",["x","y"]] #ok
#nested_list = ["+",["x",["*",["y","z"]]]] #ok
result = evaluation(nested_list)
print(result)
