class Node:
    """ A node of a tree
    """
    # node label
    label = ""
    # list of child nodes
    children = []

    # constructor
    def __init__(self,label,children):
        self.label = label
        self.children = children

    # recursive visit of node and descendants
    def visit(self):
        print(self.label, end='')
        if len(self.children) > 0:
            # visit children
            print('{',end='')
            for node in self.children:
                node.visit()
            print('}',end='')

