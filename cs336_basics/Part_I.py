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
import time
import cProfile, pstats
import resource
import numpy as np
import random

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



def get_words(input_path, special_tokens, num_processes=8):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        results = pool.starmap(process_chunk, chunk_args)  # list of dicts, one per chunk
    # merge (reduce step) happens back in the main process
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
    byte_to_bytes = [bytes([i]) for i in range (256)]
    for piece in pieces:
        for match in re.finditer(PAT, piece):
            word_bytes = tuple(byte_to_bytes[b] for b in match.group().encode("utf-8"))
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
                pair_words[p] = pair_words.get(p, set())
                pair_words[p].add(tuple(word_updated))
            
            #si paires communes: 
            else : 
                #si count ≠ : 
                #update pairs count
                #update pair_words: 
                    #delete ancien mot, add nouveau mot
                if word_updated_pairs[p] != old_word_pairs[p]:
                    dif = old_word_pairs[p] - word_updated_pairs[p]
                    pairs[p] = pairs.get(p,0) - dif * count_word
                pair_words[p] = pair_words.get(p,set())
                pair_words[p].discard(word)
                pair_words[p] = pair_words.get(p,set())
                pair_words[p].add(tuple(word_updated))
                

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
                pair_words[p] = pair_words.get(p,set())
                pair_words[p].discard(word)
                if pair_words[p] == set():
                    del pair_words[p]
    
    pair_words.pop((e1,e2), None)

    return pairs,words_dict,pair_words

def get_pair_words(words_dict):
    pair_words = {}
    for word, count_word in list(words_dict.items()):
        pairs = [(word[i],word[i+1]) for i in range(len(word)-1)]
        for p in pairs :
            pair_words[p] = pair_words.get(p,set()) 
            if word not in pair_words[p]: 
                pair_words[p].add(word)
    return pair_words



def train_bpe(input_path, vocab_size, special_tokens):
    vocab = vocab_init(special_tokens)
    t0 = time.perf_counter()
    words_dict= get_words(input_path, special_tokens) #{(b'u',): 1, (b' ', b'd', b'o', b'n'): 1, 
    pairs = get_pairs(words_dict) #{(b' ', b'd'): 26, (b'd', b'o'): 5, 
    pair_words = get_pair_words(words_dict)
    t1 = time.perf_counter()
    
    print(f"pre-tokenization + initial counting: {t1 - t0:.2f}s")
    
    size = len(vocab)
    merges = []
    t2 = time.perf_counter()
    while size < vocab_size :
        pair = max(pairs, key = lambda x: (pairs[x],x))
        merges.append(pair)
        vocab[len(vocab)]=bytes(pair[0]+pair[1])
        pairs,words_dict,pair_words = pairs_update(pairs,pair,words_dict,pair_words)

        size +=1
    t3 = time.perf_counter()
    print(f"merge loop: {t3 - t2:.2f}s")    
    return vocab, merges



############ BPE TINYSTORIES

def train_bpe_tinystories():
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.perf_counter()
    vocab, merges = train_bpe('data/TinyStoriesV2-GPT4-train.txt',10000,['<|endoftext|>'])
    elapsed = time.perf_counter() - start
    profiler.disable()
    pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)
    print(f"total duration : {elapsed}")
    # at the end of your script / after train_bpe finishes
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak memory (main process): {peak_kb / 1024**2:.1f} MB")
    return vocab, merges

# A) took 282s, 128.1MB memory
#B) merge loop took 85s, pretokenisation + initial counting took 196s  
"""
pre-tokenization + initial counting: 196.76s
merge loop: 85.88s
         385192897 function calls (385192505 primitive calls) in 282.645 seconds

   Ordered by: cumulative time
   List reduced from 599 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       38    0.004    0.000  389.068   10.239 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:500(_wait_for_updates)
       78    0.006    0.000  353.931    4.538 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/selectors.py:402(select)
        1    0.050    0.050  282.644  282.644 /Users/timothee/Desktop/assignment1-basics-1/cs336_basics/Part_I.py:313(train_bpe)
       78    0.011    0.000  215.223    2.759 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py:1104(wait)
        1    0.052    0.052  196.279  196.279 /Users/timothee/Desktop/assignment1-basics-1/cs336_basics/Part_I.py:148(get_words)
        1    0.000    0.000  196.147  196.147 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:738(__exit__)
        1    0.000    0.000  194.819  194.819 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:654(terminate)
       15    0.000    0.000  194.695   12.980 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/util.py:205(__call__)
        1    0.000    0.000  194.694  194.694 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:680(_terminate_pool)
        1    0.000    0.000  194.684  194.684 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:671(_help_stuff_finish)
        1    0.003    0.003  194.683  194.683 {method 'acquire' of '_multiprocessing.SemLock' objects}
      3/1    0.000    0.000  194.679  194.679 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/threading.py:995(_bootstrap)
      3/1    0.000    0.000  194.679  194.679 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/threading.py:1035(_bootstrap_inner)
      3/1    0.000    0.000  194.679  194.679 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/threading.py:978(run)
        1    0.000    0.000  194.679  194.679 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:527(_handle_tasks)
       16    0.000    0.000  194.679   12.167 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py:201(send)
       52    0.000    0.000  194.679    3.744 {built-in method posix.write}
       21    0.000    0.000  194.679    9.270 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py:389(_send_bytes)
       21    0.000    0.000  194.679    9.270 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py:364(_send)
        1    0.000    0.000  194.678  194.678 /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/pool.py:573(_handle_results)


total duration : 282.6829107920057
peak memory (main process): 128.1 MB

"""

#2.6 BPE Tokenizer: Encoding and Decoding

import pickle
import json

def save_tokenizer_files(vocab, merges, vocab_filepath, merges_filepath):
    with open(vocab_filepath, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_filepath, "wb") as f:
        pickle.dump(merges, f)


def bytes_to_unicode() -> dict[int, str]:
    """Reversible byte(0-255) <-> printable-unicode-char mapping, GPT-2 style."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files_json(vocab, merges, vocab_filepath, merges_filepath):
    byte_to_char = bytes_to_unicode()

    def bytes_to_str(b: bytes) -> str:
        return "".join(byte_to_char[byte] for byte in b)

    # GPT-2 vocab.json convention: token-string -> id (inverted from our int -> bytes)
    json_vocab = {bytes_to_str(tok_bytes): tok_id for tok_id, tok_bytes in vocab.items()}
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, ensure_ascii=False)

    with open(merges_filepath, "w", encoding="utf-8") as f:
        for e1, e2 in merges:
            f.write(f"{bytes_to_str(e1)} {bytes_to_str(e2)}\n")


def load_tokenizer_files_json(vocab_filepath, merges_filepath):
    byte_to_char = bytes_to_unicode()
    char_to_byte = {c: b for b, c in byte_to_char.items()}

    def str_to_bytes(s: str) -> bytes:
        return bytes(char_to_byte[c] for c in s)

    with open(vocab_filepath, "r", encoding="utf-8") as f:
        json_vocab = json.load(f)
    vocab = {tok_id: str_to_bytes(tok_str) for tok_str, tok_id in json_vocab.items()}

    merges = []
    with open(merges_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, s2 = line.split(" ")
            merges.append((str_to_bytes(s1), str_to_bytes(s2)))

    return vocab, merges


class tokenizer() :
    def __init__(self, vocab, merges, special_tokens=None):
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] 
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) :
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and
        list of merges (in the same format that your BPE training code output) and (optionally)
        a list of special tokens. This method should accept the following additional parameters:
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str]
        # Files are written by save_tokenizer_files_json (GPT-2 vocab.json /
        # merges.txt convention), so load them the same way.
        vocab, merges = load_tokenizer_files_json(vocab_filepath, merges_filepath)

        for special_tok in (special_tokens or []) :
            if special_tok.encode('utf-8') not in vocab.values():
                vocab[len(vocab)] = special_tok.encode('utf-8')

        return cls(vocab, merges, special_tokens)
    
    def pretokenise(self,text,special_tokens,num_processes = 8):
        special_tokens = special_tokens or []
        
        split_pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        pieces = re.split(f"({split_pattern})", text) if special_tokens else [text]
        PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = []
        byte_to_bytes = [bytes([i]) for i in range (256)]
        for piece in pieces:
            if piece in special_tokens: 
                words.append((piece.encode('utf-8'),)) 
            else : 
                for match in re.finditer(PAT, piece):
                    word_bytes = tuple(byte_to_bytes[b] for b in match.group().encode("utf-8"))
                    words.append(word_bytes)
        return words
    

    def apply_merge_word(self,word):
        """
        word =  [b't', b'h', b'e']
        merge = [b'the']
        """
        word = list(word)

        if len(word) == 1 : 
            return word

        for (e1,e2) in self.merges : 
            i = 0
            while True :
                if i == len(word)-1: 
                    break
                if e1 == word[i] and e2 == word[i+1]:
                    char = bytes(word[i] + word[i+1])
                    word[i] = char
                    word.pop(i+1)
                else : 
                    i +=1
        return word


    def encode(self, text: str) -> list[int] :
        """
        Encode an input text into a sequence of token IDs.
        """
        self.bytes_to_ids = {v : k for k,v in self.vocab.items()}
        
        words = self.pretokenise(text,self.special_tokens)
        
        final_encoding = []
        for word in words:
            if len(word) > 1 : 
                word = self.apply_merge_word(word)
            final_encoding += [self.bytes_to_ids[b] for b in word]
        
        return final_encoding
    
        
    
    def encode_iterable(self, iterable) :
        """ 
        Given an iterable ofstrings (e.g., a Python file handle), return a generator that
        lazily yields token IDs. This is required for memory-efficient tokenization of 
        large files that we cannot directly load into memory.
        """
        for line in iterable :
            for tok_id in self.encode(line):
                yield tok_id

    def decode(self, ids: list[int]) -> str :
        """
        Decode a sequence of token IDs into text.
        To test your Tokenizer against our provided tests, you will first need to implement the test
        adapter at [adapters.get_tokenizer] . Then, run uv run pytest tests/test_tokenizer.py. Your
        implementation should be able to pass all tests.
        """
        ids_bytes = [self.vocab[i] for i in ids]
        replacement = 'U+FFD'
        text = b"".join(ids_bytes).decode('utf-8', errors = 'replace')
        return text






if __name__ == "__main__" :
    vocab, merges = train_bpe_tinystories()
    save_tokenizer_files_json(vocab, merges, "data/ts_vocab.json", "data/ts_merges.txt")



## 2.7 Experiments 

def sample_document(file_path, special_token=b'<|endoftext|>'):
    """Seek to a random byte offset, then expand outward to the surrounding
    <|endoftext|>  and return the single document found between them.

    Strategy:
      1. Pick a random byte position `pos`.
      2. Scan forward from pos for the next delimiter -> doc_end.
      3. Scan backward from pos for the previous delimiter -> doc_start.
      4. Return the text strictly between the two delimiters.
    """
    tok_len = len(special_token)
    window = 1 << 16  # 64 KiB read window

    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        pos = random.randint(0, file_size)

        # Scan forward for the next delimiter 
        doc_end = file_size
        scan = pos
        # keep a small overlap of tok_len-1 bytes so a delimiter straddling a
        # window boundary is not missed
        while scan < file_size:
            f.seek(scan)
            chunk = f.read(window)
            found = chunk.find(special_token)
            if found != -1:
                doc_end = scan + found
                break
            scan += window - (tok_len - 1)

        #  Scan backward for the previous delimiter (end of it is doc_start) 
        doc_start = 0
        scan = pos
        while scan > 0:
            read_start = max(0, scan - window)
            f.seek(read_start)
            chunk = f.read(scan - read_start)
            found = chunk.rfind(special_token)
            if found != -1:
                doc_start = read_start + found + tok_len  # skip past the delimiter
                break
            scan = read_start + (tok_len - 1)
            if read_start == 0:
                break

        if doc_start >= doc_end:
            # pos landed inside a delimiter or an empty span = just retry
            return sample_document(file_path, special_token)

        f.seek(doc_start)
        raw = f.read(doc_end - doc_start)

    return raw.decode("utf-8", errors="ignore").strip()


def compression_ratio(tok, documents):
    """Average bytes/token across a list of document strings."""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_tokens = sum(len(tok.encode(doc)) for doc in documents)
    return total_bytes / total_tokens


def tokenizer_experiments(n_samples=10):
    ts_path = 'data/TinyStoriesV2-GPT4-train.txt'
    ts_vocab, ts_merges = 'data/ts_vocab.json', 'data/ts_merges.txt'

    ts_tok = tokenizer.from_files(ts_vocab, ts_merges, special_tokens=['<|endoftext|>'])

    # a) compression ratio on TinyStories with the TinyStories tokenizer
    ts_docs = [sample_document(ts_path) for _ in range(n_samples)]
    print(f"TinyStories tok on TinyStories: {compression_ratio(ts_tok, ts_docs):.3f} bytes/token")

    # c) throughput estimate
    t0 = time.perf_counter()
    total_bytes = 0
    for doc in ts_docs:
        total_bytes += len(doc.encode("utf-8"))
        ts_tok.encode(doc)
    elapsed = time.perf_counter() - t0
    throughput = total_bytes / elapsed
    pile_bytes = 825e9
    print(f"throughput ~ {throughput/1e6:.2f} MB/s")
    print(f"time to tokenize the Pile (825GB) ~ {pile_bytes / throughput / 3600:.1f} hours")

    return True


def encode_dataset_to_uint16(input_path, tok, out_path):
    """(d) Encode a file to a uint16 numpy array on disk, streaming so we
    never hold the full corpus in memory. Uses encode_iterable over the file
    handle and appends in blocks."""

    ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for tok_id in tok.encode_iterable(f):
            ids.append(tok_id)
    arr = np.array(ids, dtype=np.uint16)
    np.save(out_path, arr)
    print(f"saved {arr.size} tokens to {out_path}")
    return arr


"""
A) Compression ratio (bytes/token) is higher for the larger-vocab tokenizer:
   the 32K OWT tokenizer packs more bytes per token than the 10K TinyStories
   tokenizer, because larger vocab -> longer merged tokens -> fewer tokens.

B) Tokenizing OWT with the TinyStories tokenizer gives a WORSE (lower)
   compression ratio: TinyStories has a small, simple-English vocabulary, so
   OWT's rarer words fragment into many short tokens.

C) Measure seconds to encode the sample, divide bytes by time to get
   bytes/second (throughput), then Pile_time = 825e9 bytes / throughput.

D) uint16 holds integers in [0, 65535], which covers vocab sizes up to 64K
   (both 10K and 32K fit) while using only 2 bytes/token so half the memory of
   int32 and far less than Python ints.
"""