import tree

# recursive parse function
# each function corresponds to the production rules of a non terminal

# E -> TE1
def E(cursor,input):
    cursor, t = T(cursor,input)                     # match T
    if t is None:
        return cursor, None
    else:
        cursor, e1 = E1(cursor,input)               # match E1
        if e1 is None:
            return cursor, None
        else:
            # create node for E -> TE1
            return cursor, tree.Node('E',[t,e1])

# E1 -> +TE1 | e
def E1(cursor,input):
    if cursor<len(input) and input[cursor]=='+':    # match +
        cursor=cursor+1
        cursor, t = T(cursor,input)                 # match T
        if t is None:
            return cursor, None
        else:
            cursor, e1 = E1(cursor,input)           # match E1
            if e1 is None:
                return cursor, None
            else:
                # create node for E1 -> +TE1
                return cursor, tree.Node('E1',[tree.Node('+',[]),t,e1])
    else:
        # create node for E1->e
        return cursor, tree.Node('E1',[tree.Node('e',[])])

# T -> FT1
def T(cursor,input):
    cursor, f = F(cursor,input)         # match F
    if f is None:
        return cursor, None
    else:
        cursor, t1 = T1(cursor,input)   # match T1
        if t1 is None:
            return cursor, None
        else:
            # create node for T->FT1
            return cursor, tree.Node('T',[f,t1])

# T1 -> *FT1 | e
def T1(cursor,input):
    if cursor<len(input) and input[cursor]=='*':                # match *
        cursor=cursor+1
        cursor, f = F(cursor,input)                             # match F
        if f is None:
            return cursor, None
        else:
            cursor, t1 = T1(cursor,input)                       # match T1
            if t1 is None:
                return cursor, None
            else:
                # create node T1->*FT1
                return cursor, tree.Node('T1',[tree.Node('*',[]),f,t1])
    else:
        # create node T1->e
        return cursor, tree.Node('T1',[tree.Node('e',[])])

# F -> (E) | n
def F(cursor,input):
    if cursor<len(input) and input[cursor]=='(':            # match (
        cursor = cursor+1
        cursor, e = E(cursor,input)                         # match E
        if e is None:
            return cursor, None
        elif cursor<len(input) and input[cursor]==')':      # match )
            cursor = cursor+1
            # create node F->(E)
            return cursor, tree.Node('F',[tree.Node('(',[]),e,tree.Node(')',[])])
        else:
            return cursor, None
    elif cursor<len(input) and input[cursor]=='n':          # match n
        cursor = cursor+1
        # create node F->n
        return cursor, tree.Node('F',[tree.Node('n',[])])
    else:
        return cursor, None

w = "n+n*n"
cursor, parseTree = E(0,w)

print('Input:', w)
if parseTree is not None and cursor==len(w):
    print('Parse success')
    parseTree.visit()
else:
    print('Parse failed at',cursor)