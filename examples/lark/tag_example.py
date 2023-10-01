import lark

#
# Grammar for open tag in Markup Languages
# <tag attr="val" attr="p: v; p: v" attr>
#

tag_grammar = r"""
    open_tag: "<" CNAME (attr_name ["=" attr_val])* ">"
    attr_name: CNAME
    attr_val: "\"" (CNAME | val_list) "\""
    val_list:   (CNAME ":" STRING ";")* [CNAME ":" STRING]

    STRING: (LETTER|DIGIT)+      
    %import common.CNAME
    %import common.LETTER
    %import common.DIGIT
    %import common.WS
    %ignore WS
"""

source = '<tag att1="val1" attr2 attr3="p1: v1; p2: v2;" attr4="p1: v1; p2: v2">'
print("**** Grammar ****\n",tag_grammar)
print(">> Input string: ",source)

tag_parser = lark.Lark(tag_grammar,start="open_tag")
result = tag_parser.parse(source)
print("\n*** Parse tree ***\n",result)

print("\n*** Parse tree pretty print\n", result.pretty())

#
# alternative with recursive definition of the list of values
#
tag_grammar1 = r"""
    open_tag: "<" CNAME (attr_name ["=" attr_val])* ">"
    attr_name: CNAME
    attr_val: "\"" (CNAME | val_list) "\""
    val_list:   CNAME ":" STRING ";" val_list
              | CNAME ":" STRING ";"
              | CNAME ":" STRING

    STRING: (LETTER|DIGIT)+      
    %import common.CNAME
    %import common.LETTER
    %import common.DIGIT
    %import common.WS
    %ignore WS
"""
