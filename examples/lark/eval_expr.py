import lark

#@lark.v_args(inline=True)
#class EvalExpr(lark.Transformer):
#    NUMBER = float
#    def sum(self,expr,term):
#        return expr+term
#    def sub(self,expr,term):
#        return expr-term
#    def mul(self,term,factor):
#        return term*factor
#    def div(self,term,factor):
#        return term/factor
#    def neg(self, factor):
#       return factor


class EvalExpr(lark.Transformer):
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


source = '-3*2+5+(2+3)*2'
print(">> Input string: ",source)

json_parser = lark.Lark.open("expressions.lark",rel_to=__file__,start="expr",parser='lalr')
result = json_parser.parse(source)
print("\n*** Parse tree pretty print\n", result.pretty())

# print tree to PDF file
#graph = lark.tree.pydot__tree_to_graph(result, "TB")
#graph.write_pdf("expr_tree.pdf")

eval_tree = EvalExpr().transform(result)
print(">> Result: ", eval_tree)

print("\n*** Creating parser with embedded Transformer")
json_parser = lark.Lark.open("expressions.lark",rel_to=__file__,start="expr",parser='lalr',transformer=EvalExpr())
result = json_parser.parse(source)
print(">> Result:", result)