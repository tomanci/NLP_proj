import lark

source = '{"a":1, "b":[1,2,3], "c": {"d": true}, "e":[[1,2,3],[4,5,6]], "f": "hello" }'
print(">> Input string: ",source)

json_parser = lark.Lark.open("json.lark",rel_to=__file__,start="value")
result = json_parser.parse(source)
print("\n*** Parse tree ***\n",result)
print("\n*** Parse tree pretty print\n", result.pretty())

print("\n saving PDF version of tree")
graph = lark.tree.pydot__tree_to_graph(result, "TB")
graph.write_pdf("json_parse_tree.pdf")

print(30*"-")
print("Shaping the tree")
source = '{"a": true, "b": [true,false]}'
print(">> Input string: ",source)
result = json_parser.parse(source)

json_parser_shape = lark.Lark.open("json_shaped.lark",rel_to=__file__,start="value")
result_shaped = json_parser_shape.parse(source)
print("\n saving PDF version of trees")
graph = lark.tree.pydot__tree_to_graph(result, "TB")
graph.write_pdf("json_noshaped_parse_tree.pdf")

graph = lark.tree.pydot__tree_to_graph(result_shaped, "TB")
graph.write_pdf("json_shaped_parse_tree.pdf")
