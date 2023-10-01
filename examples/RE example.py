import re

################################
# Example: use of search()
################################

print("Search example\n")

s = "var1 = 234+_v2"
# pattern for identifiers
p = r"[a-zA-Z_][a-zA-Z_0-9]*"

print("Input: ",s)
print("Pattern: ",p)

# compile RE into object re_id
re_id = re.compile(p)

# search matches on input string
result = re_id.search(s)
while result:
    # iterate until no matches (result is None)
    print("match: ",result)
    # search string for match starting after the previous one
    result = re_id.search(s,result.end())

################################
# Example: use of split()
################################
print(60*"-")
print("Split example\n")

s = "Tom, Ben, Bob are together. They walk; they sing."
# p defines the separator
p = r"[\.,; ]+"
result = re.split(p,s);
print("Input: ",s)
print("Pattern: ",p)
print("split result: ", result)

print()

# Alternative solution
p = r"\W+"
result = re.split(p,s);
print("Alternative pattern: ",p)
print("split result: ", result)

################################
# Example: use of findall
################################
print(60*"-")
print("findall example")

s = "123 is a number. 324 and 5678 are two numbers."
p = r"\d+"
# p = r"[0-9]+"
print("Input string: ",s)
print("Pattern: ",p)

result = re.findall(p,s)
print("findall result: ", result)

################################
# Example: escape with backslash
################################
print(60*"-")
print("Backslash escape example")

s = "name=file.doc"
p = r"[a-zA-Z0-9]+\.[a-zA-Z0-9]{3}"
print("Input string: ",s)
print("Pattern: ",p)

result = re.findall(p,s)
print("findall result: ", result)

################################
# Examples: use of groups
################################
print(60*"-")
print("Group examples")

### example 1
print("\n-> Find var: val pairs")
s = "<tag id=\"p1: val1;p2: val2; p3: val3;\">"
# pattern for var: val pairs
p = r"\s*([a-z0-9]+):\s*([a-z0-9]+);\s*"

print("Input string: ",s)
print("Pattern: ",p)

result = re.findall(p,s)
print("findall result: ", result)

### example 2
print("\n-> Find repeated words")
s = "I like the the red car car"
# \1 matches the string matched by the first group
p = r"(\w+)\W+\1"

print("Input string: ",s)
print("Pattern: ",p)
result = re.findall(p,s)
print("findall result: ", result)

### example 3
print("\n-> Find parts of URL")
s = r"http://www.google.com/file.html"
p = r"([a-z]+)://([a-z0-9\.]+)/([a-z0-9\.]+)"

print("Input string: ",s)
print("Pattern: ",p)
result = re.findall(p,s)
print("findall result: ", result)

p = r"(?:[a-z]+)://(?:[a-z0-9\.]+)/([a-z0-9\.]+)"

print("Input string: ",s)
print("Pattern with not capturing groups: ",p)
result = re.findall(p,s)
print("findall result: ", result)

################################
# Examples: greedy vs non greedy match
################################
print(60*"-")
print("Greedy vs non-greedy match")

s = "<tag>text</tag>"
p_greedy = r"<.*>"
p_nogreedy = r"<.*?>"

print("Input string: ",s)
print("Greedy pattern: ",p_greedy)
result = re.findall(p_greedy,s)
print("Greedy result: ", result)

print("Non greedy pattern: ",p_nogreedy)
result = re.findall(p_nogreedy,s)
print("Non greedy result: ", result)

################################
# Examples: lookahead match
################################
print(60*"-")
print("Group Lookahead")

s = "100 apples 53 peaches 103 pears"
p= r"[1-9][0-9]*\s+(?=apples|pears)"

print("Input string: ",s)
print("Pattern: ",p)
result = re.findall(p,s)
print("Result: ", result)

################################
# Examples: tokenizer
################################
print(60*"-")
print("Tokenizer")

s = "int v = 0; float a1; if(v) {v++;}; while(1){}"
types = r"(?P<type>int|float)"
keywords = r"(?P<keyword>if|while)"
id = r"(?P<id>[a-zA-Z_][a-zA-Z0-9_]*)"
num = r"(?P<number>[0-9]+)"
char = r"(\S)"

p = types+"|"+keywords+"|"+id+"|"+num+"|"+char
## wrong order
#p = id+"|"+keywords+"|"+types+"|"+num+"|"+char

print("Input string: ",s)
print("Pattern: ",p)

for match in re.finditer(p,s):
    print(match.lastgroup,":",match.group(), "["+str(match.start())+","+str(match.end())+"]")
    #print(match.groupdict())

################################
# Examples: email address
################################

s ="mailto:john.doe77@mail.x-server1.com"
user = r"(?:\w|\d)+(?:\.(?:\w|d)+)*"
host = r"(?:\w|\d)+(?:-(?:\w|\d)+)*(?:\.(?:\w|d)+(?:-(?:\w|\d)+)*)*"
p = r"(?P<user>"+user+")@(?P<server>"+host+")"
print("Input string: ",s)
print("Pattern: ",p)

result = re.findall(p,s)
print("Result: ", result)
