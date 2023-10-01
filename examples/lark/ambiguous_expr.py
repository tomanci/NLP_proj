import lark,pydot

#
# Grammar for expressions
# ambiguous version
#

exp_grammar = r"""
    expr: expr MULT expr
        | expr PLUS expr
        | "(" expr ")"
        | NUMBER
    
    PLUS: "+"
    MULT: "*"    
    %import common.SIGNED_NUMBER -> NUMBER
    %import common.WS
    %ignore WS
"""

e = '3*4+2'
print("**** Grammar ****\n",exp_grammar)
print(">> Input string: ",e)

exp_parser = lark.Lark(exp_grammar,start="expr",ambiguity="explicit")
#import logging
#lark.logger.setLevel(logging.DEBUG)
#exp_parser = lark.Lark(exp_grammar,start="expr",parser="lalr",debug="True")
#exp_parser = lark.Lark(exp_grammar,start="expr")

result = exp_parser.parse(e)
print("\n*** Parse tree ***\n",result)

print("\n*** Parse tree pretty print\n", result.pretty())

print("\n saving PDF version of tree")
# you need to install the pydot package
# and graphviz https://graphviz.org/
graph = lark.tree.pydot__tree_to_graph(result, "TB")
#graph.write_svg("ambigous_parse_tree.svg")
graph.write_pdf("ambiguous_parse_tree.pdf")
