import lark

source = 'struct data {\
   int a;\
   float b;\
   char c;\
   double d;\
   int e;\
} x;\
'

@lark.v_args(inline=True)
class Struct_init(lark.Transformer):
    def __init__(self):
        self.struct = {} #dictionary because you have the key and the value

    def struct_init(self, struct_type, *args):
        self.struct[args[-1]] = 'struct'
        return self.struct[args[-1]], args[-1]
    
    def declaration(self, type_var, var):
        self.struct[var] = [type_var, 'NULL']
        return self.struct[var][0]
    
    def variable(self, s_var, var):
        if(s_var == var):
            print("Error in variable name: {}".format(var))
        elif (s_var in self.struct.keys()) & (var in self.struct.keys()):
            return self.struct[var][1]
        elif (s_var not in self.struct.keys()):
            print("Error in struct name")  
        else:
            print("Variable {} not defined".format(var))
    
    def assignment(self, s_var, var, value):
        if (s_var in self.struct.keys()) & (var in self.struct.keys()):
            self.struct[var][1]=value                
            return value
        elif (s_var not in self.struct.keys()):
            print("Error in struct name or struct not defined")
        else:
            print("Variable {} not defined".format(var))

    '''
    def assignment(self, s_var, var, value):
        if (s_var in self.struct.keys()) & (var in self.struct.keys()):
            print(value, type(value))
            if self.struct[var][0] == "int":
                assert type(value) == int, "Type error"
                self.struct[var][1]=value
            elif self.struct[var][0] == "float":
                assert type(value) == float, "Type error"
                self.struct[var][1]=value
            elif self.struct[var][0] == "char":
                assert type(value) == str, "Type error"
                self.struct[var][1]=value  
            elif self.struct[var][0] == "double":
                assert type(value) == float, "Type error"
                self.struct[var][1]=value                
            return value
    '''

    def output(self, s_var):
        if s_var in self.struct.keys():
            print('output:', "struct", s_var)
            for el in self.struct.keys():
                if el == s_var:
                    continue
                else:
                    print(el, "type", self.struct[el][0], "value", self.struct[el][1])
        else:
            print("Error in struct name")



# the call to the parser provides directly the result - 2nd method
struct_parser = lark.Lark.open("struct_complete_test.lark",rel_to=__file__,start="stmt",parser='lalr',transformer=Struct_init())

print("Define your struct:")

while True:
    try:
         line = input('> ')
    except EOFError:
        break
    try:
        res = struct_parser.parse(line) #print the result if I have no exceptions
        if res != None:
            print(res)
    # the error are from the more specific to the more general!!!
    except lark.UnexpectedToken as error: #example : you have written terminal symbols not present in the grammar
        print(error.get_context(line),end='') #print the context of the error
        print("Unexpected token %s " % error.token) #print the character generating the expression
    except lark.UnexpectedInput as error: #like 2**3
        print("Parser error")
        print(error.get_context(line))
        #general exception
    except Exception as error:
        print(error)