import tree

def B(cursor,input):
    """Function to expand non terminal B
       B -> (B)B | e
    """
    if cursor<len(input) and input[cursor]=='(':    # match (
        cursor=cursor+1
        cursor, B1 = B(cursor,input)                # match B recursively
        if input[cursor]==')' and B1 is not None:   # match )
            cursor = cursor+1
            cursor, B2 = B(cursor,input)            # match B recursively
            if B2 is None:
                return cursor, None
            else:
                # match of B->(B)B : return node of parse tree
                return cursor, tree.Node('B', [tree.Node('(',[]),B1,tree.Node(')',[]),B2])
        else:
            return cursor, None
    else:
        # match of B->e : return node of parse tree
        return cursor, tree.Node('B',[tree.Node('e',[])])

w = "(())()"
cursor, parseTree = B(0,w)

if parseTree is not None and cursor==len(w):
    print('Parse success')
    parseTree.visit()
else:
    print('Parse failed')