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

#print((b' ', b't') in (b' ', b't', b'h', b'e'))

# import os
# print(os.cpu_count())

import torch

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)
# #print(x_data)

# x = [1,2,3,4,5,6,7]
# print(x[1::2])


def softmax(tens, dim_i): 
    """
    Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension 𝑖, and apply softmax to the 𝑖-th dimension of the
    input tensor. The output tensor should have the same shape as the input tensor, but its 𝑖-th
    dimension will now have a normalized probability distribution. Use the trick of subtracting the
    maximum value in the 𝑖-th dimension from all elements of the 𝑖-th dimension to avoid numerical
    stability issues.
    """
    mx = torch.amax(tens, dim = dim_i, keepdim=True) #torch.max returns (values, indices), torch.amax only return values
    print(f"mx : {mx}")
    vect = tens-mx
    print(f"vect : {vect}")
    s = torch.sum(torch.exp(vect), dim = dim_i, keepdim=True)
    print(f"s : {s}")
    vect = torch.exp(vect)/s
    return vect

tens = torch.rand(3,4)
#print(tens.shape[1])
#print(softmax(tens,0))

m = torch.tril(torch.ones(3,4))
m = m.where(1.0,True)
print(m)
