# import regex as re
# # generic example, not your special tokens or your text
# parts = re.split(r"<SEP>", "aaa<SEP>bbb<SEP>ccc")
# print(parts)  # ['aaa', 'bbb', 'ccc']

# # what happens with no content between two delimiters?
# parts2 = re.split(r"<SEP>", "aaa<SEP><SEP>ccc")
# print(parts2)  # try it — what do you see at the empty slot?

# # re.escape on a string containing a regex metacharacter
# print(re.escape("a.b|c"))  # note how '.' and '|' get backslashed

# a = (1,2)
# print(a[1])

# d = {'a' : 3}

# d['a'] = d.get('a', 2)+2
# print(d)

# a = [1,2,3,4,5,6]
# del a[2:4]
# print(a)

# d = {("a", "b"): 1}
# for k, v in d.items():
#     del d[k]

# def merge_word(word, e1, e2):
#     word = list(word)
#     char = bytes(e1 + e2)
#     i=0
#     while True : 
#         if i >= len(word)-1: 
#             return word
#         if word[i] == e1 and word [i+1]==e2:
#             word[i] = char

#             word.pop(i+1)
#             i += 1
#         else : 
#             i += 1

# word = (b'l', b'o', b'l', b'o', b'l')
# e1 = b'l'
# e2 =b'o'
# print(merge_word(word,e1,e2))

print((b' ', b't') in (b' ', b't', b'h', b'e'))