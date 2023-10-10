import lark

class EvalExpr(lark.Transformer):
    NUMBER = float

    def sum(self,args):
        return args[0]+args[1]
    
    def mul(self,args):
        return args[0]*args[1]
    
    
source = '2*((2+1)*2)+(2+3)*2'

print(">> Input string: ",source)

json_parser = lark.Lark.open("expressions.lark",rel_to=__file__,start="expr",parser='lalr')
result = json_parser.parse(source)
print("\n*** Parse tree pretty print\n", result.pretty())

# print tree to PDF file
#graph = lark.tree.pydot__tree_to_graph(result, "TB")
#graph.write_pdf("expr_tree.pdf")

eval_tree = EvalExpr().transform(result)
print(">> Result: ", eval_tree)