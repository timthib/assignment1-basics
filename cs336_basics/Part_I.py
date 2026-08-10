"""
UNICODE 1 : 

A)
>>> chr(0)
'\x00'
>>> print(chr(0))

B)
so chr(0) is the empty (not even space) character, like the 0 of character.
It's representation repr() is '\x00'
The printed values is nothing.

C)
>>> "this is a test" + chr(0) + "hello"
'this is a test\x00hello'
>>> print( "this is a test" + chr(0) + "hello")
this is a testhello
Printing lends nothing, running show the encoding.

ord(): character => unicode integer
chr(): unicode integer => character

Can't train model on unicode points because of vocab size = 150k and sparse vocab (undertrained on rare characters)
"""

sent = "hello comment ça va ? "
l = sent.encode('utf-8')

#print(l)
#b'hello comment \xc3\xa7a va ? '

#print(list(l))
#[104, 101, 108, 108, 111, 32, 99, 111, 109, 109, 101, 110, 116, 32, 195, 167, 97, 32, 118, 97, 32, 63, 32]


##################################################################################################################

"""
UNICODE 2

"""

def q2A():
    test = "salut 123 comment ça va !?"
    t8 = test.encode('utf-8')
    t16 = test.encode('utf-16')
    t32 = test.encode('utf-32')
    print(t8) #b'salut 123 comment \xc3\xa7a va !?'
    print(t16) #b'\xff\xfes\x00a\x00l\x00u\x00t\x00 \x001\x002\x003\x00 \x00c\x00o\x00m\x00m\x00e\x00n\x00t\x00 \x00\xe7\x00a\x00 \x00v\x00a\x00 \x00!\x00?\x00'
    print(t32) #b'\xff\xfe\x00\x00s\x00\x00\x00a\x00\x00\x00l\x00\x00\x00u\x00\x00\x00t\x00\x00\x00 \x00\x00\x001\x00\x00\x002\x00\x00\x003\x00\x00\x00 \x00\x00\x00c\x00\x00\x00o\x00\x00\x00m\x00\x00\x00m\x00\x00\x00e\x00\x00\x00n\x00\x00\x00t\x00\x00\x00 \x00\x00\x00\xe7\x00\x00\x00a\x00\x00\x00 \x00\x00\x00v\x00\x00\x00a\x00\x00\x00 \x00\x00\x00!\x00\x00\x00?\x00\x00\x00'

def q2B():
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    print(decode_utf8_bytes_to_str_wrong("ç".encode('utf-8')))
    #UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data

    # utf-8 encode some characters with 2 or more bytes. For eg "ç".encode('utf-8') = \xc3\xa7a
    # But here the function isolates them : "for b in bytestring"
    # So the function ask to decode \xc3 alone, which is impossible since \xc is a special marker that means the character is composed of 2 bytes. 
    # Hence \xc3 alone can't be decoded, which throw an error. 
    # The position 0 comes from the isolation of bytes, each bytes is at position 0 in the sequence of bytes selected. 

#print(q2B())

def q2C(): 
    return "ç"



##################################################################################################################

"""
TOKENIZER

Vocab initialisation 
Pre tokenisation
compute BPE merges


"""

import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool


def vocab_init(spectial_tokens):   
    vocab = {i : bytes([i]) for i in range(256)}
    for i,tok in enumerate(spectial_tokens):
        vocab[i+256] = tok.encode('utf-8')
    return vocab


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def get_words(input_path, special_tokens, num_processes=4):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        results = pool.starmap(process_chunk, chunk_args)  # list of dicts, one per chunk

    # merge (reduce step) — happens back in the main process
    d: dict[tuple[bytes, ...], int] = {}
    for local_counts in results:
        for word_bytes, count in local_counts.items():
            d[word_bytes] = d.get(word_bytes, 0) + count

    return d





def process_chunk(input_path, start, end, special_tokens):
    """Runs in a worker process. Reads its byte range, pre-tokenizes,
    returns a LOCAL word-count dict. Nothing shared with other workers."""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    split_pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
    pieces = re.split(split_pattern, chunk) if special_tokens else [chunk]
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    local_counts: dict[tuple[bytes, ...], int] = {}
    for piece in pieces:
        for match in re.finditer(PAT, piece):
            word_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            local_counts[word_bytes] = local_counts.get(word_bytes, 0) + 1
    return local_counts

def get_pairs(word_dict):
    pairs = {}
    for word,count in word_dict.items() :
        for i in range (len(word)-1):
            pairs[(word[i],word[i+1])] = pairs.get((word[i],word[i+1]),0) + count
    return pairs

def merge_word(word, e1, e2):
    word = list(word)
    char = bytes(e1 + e2)
    i=0
    while True : 
        if i >= len(word)-1: 
            return word
        if word[i] == e1 and word [i+1]==e2:
            word[i] = char

            word.pop(i+1)
            i += 1
        else : 
            i += 1


#def pair_update(pairs, pair, words_dict,pair_words):
    #nouvelle paire 
    #pour tous les mots contenant la paire :
        #update mot 
        #update words_dict : paires nouveau mot => count ancien mot
        #get paires nouveau mot + count
        #get paire ancien mot + count
        #si nouvelle paire: 
            #ajoute à pairs + count paire dans nouveau mot * ancien mot count
            #ajoute à pairs_words + nouveau mot 
        #si paires communes: 
            #si count ≠ : 
              #update pairs count
            #update pair_words: 
                #delete ancien mot, add nouveau mot 
        #si paires disparues : 
            #update pairs: -count dans ancien mot * count ancien mot 
            #si count = 0 => del 
            #update pair_words: del ancien mot 
            # si plus de mot => del pair

    


def pairs_update(pairs,pair,words_dict,pair_words):
    #nouvelle paire 
    #update mot 
    #update words_dict : paires nouveau mot => count ancien mot
    e1,e2 = pair[0],pair[1]
    for word in list(pair_words[(e1,e2)]):
        word_updated = merge_word(word, e1, e2)
        count_word = words_dict[word]
        words_dict[tuple(word_updated)] = count_word
        del words_dict[word]
        
        #get paires nouveau mot + count
        word_updated_pairs = {}
        if len(word_updated) > 1:
            new_word_pairs =  [(word_updated[i],word_updated[i+1]) for i in range(len(word_updated)-1)]
            for p in new_word_pairs :
                word_updated_pairs[p] = word_updated_pairs.get(p,0)+1

        #get paire ancien mot + count
        old_word_pairs = {}
        if len(word) >1 : 
            word_pairs = [(word[i],word[i+1]) for i in range(len(word)-1)]
            for p in word_pairs :
                old_word_pairs[p] = old_word_pairs.get(p,0)+1
        
        #si nouvelle paire
        #ajoute à pairs + count paire dans nouveau mot * ancien mot count
        #ajoute à pairs_words + nouveau mot 
        for p in word_updated_pairs.keys() : 
            if p not in old_word_pairs.keys():
                pairs[p] = pairs.get(p,0)+ word_updated_pairs[p] * count_word
                pair_words[p] = pair_words.get(p,[]) + [tuple(word_updated)]
            
            #si paires communes: 
            else : 
                #si count ≠ : 
                #update pairs count
                #update pair_words: 
                    #delete ancien mot, add nouveau mot
                if word_updated_pairs[p] != old_word_pairs[p]:
                    dif = old_word_pairs[p] - word_updated_pairs[p]
                    pairs[p] = pairs.get(p,0) - dif * count_word
                pair_words[p] = pair_words.get(p,[])
                pair_words[p].remove(word)
                pair_words[p] = pair_words.get(p,[]) + [tuple(word_updated)]
                

        #si paires disparues : 
            #update pairs: -count dans ancien mot * count ancien mot 
            #si count = 0 => del 
            #update pair_words: del ancien mot 
            # si plus de mot => del pair
        for p in old_word_pairs.keys():
            if p not in word_updated_pairs.keys():
                pairs[p] = pairs.get(p,0) - old_word_pairs[p]*count_word
                if pairs[p]==0 : 
                    del pairs[p]
                pair_words[p] = pair_words.get(p,[])
                pair_words[p].remove(word)
                if pair_words[p] == []:
                    del pair_words[p]
    
    pair_words.pop((e1,e2), None)

    return pairs,words_dict,pair_words

def get_pair_words(words_dict):
    pair_words = {}
    for word, count_word in list(words_dict.items()):
        pairs = [(word[i],word[i+1]) for i in range(len(word)-1)]
        for p in pairs :
            pair_words[p] = pair_words.get(p,[]) 
            if word not in pair_words[p]: 
                pair_words[p].append(word)
    return pair_words



def train_bpe(input_path, vocab_size, special_tokens):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    vocab = {}
    merges = []

    vocab = vocab_init(special_tokens)
    words_dict= get_words(input_path, special_tokens) #{(b'u',): 1, (b' ', b'd', b'o', b'n'): 1, 
    pairs = get_pairs(words_dict) #{(b' ', b'd'): 26, (b'd', b'o'): 5, 
    size = len(vocab)
    pair_words = get_pair_words(words_dict)
    while size < vocab_size :
        pair = max(pairs, key = lambda x: (pairs[x],x))
        merges.append(pair)
        vocab[len(vocab)]=bytes(pair[0]+pair[1])
        pairs,words_dict,pair_words = pairs_update(pairs,pair,words_dict,pair_words)

        size +=1

    return vocab, merges

if __name__ == "__main__":
    train_bpe('data/test_data.txt',350,['<|endoftext|>'])

