"""
Utility functions for handling RNA secondary structures.
"""


def len_motif(motif: str) -> int:
    """
    Calculate the length of a motif, excluding the 5' and 3' ends.
    """
    return len([c for c in motif if c != '*' and c != '5' and c != '3'])


def extract_pairs(ss):
    pairs = list(range(len(ss)))
    stack = []
    for i, c in enumerate(ss):
        if c=='.':
            pass
        elif c=="(":
            stack.append(i)
        elif c==")":
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
        else:
            raise ValueError(f"wrong structure at position {i}: {c}")
    return pairs


def extract_pairs_list(ss):
    """Extract pairs from a secondary structure string.
    Returns a list of tuples where each tuple contains the indices of paired positions.
    """
    pairs = []
    stack = []
    for i, c in enumerate(ss):
        if c == ".":
            pass
        elif c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            pairs.append((j, i))
        else:
            raise ValueError(f"wrong structure at position {i}: {c}")
    return pairs


def pairs_match(ss): # find the pairs in a secondary structure, return a dictionary
    assert len(ss) > 2
    pairs = dict()
    stack = []
    for i, s in enumerate(ss):
        if s==".":
            pass
        elif s=="(":
            stack.append(i)
        elif s==")":
            j = stack.pop()
            assert j < i
            pairs[j] = i
            pairs[i] = j
        else:
            raise ValueError(f'the value of structure at position: {i} is not right: {s}!')
    return pairs


def struct_dist(s1, s2):
    assert len(s1) == len(s2), f"len(s1)={len(s1)}, len(s2)={len(s2)}, s1: {s1}, s2: {s2}"
    pairs_1 = pairs_match(s1)
    pairs_2 = pairs_match(s2)
    union = len(pairs_1.keys()|pairs_2.keys())
    overlap = len(s1) - union
    for k in pairs_1:
        if k in pairs_2 and pairs_1[k]==pairs_2[k]:
            overlap += 1
    return len(s1) - overlap


# convert a motif to a target structure by filling the boundaries with dots
def dotbracket2target(line: str) -> str:
    return ''.join('...' if x == '*' else x for x in line)


# convert a motif to a constraint structure by filling the boundaries with dots, other characters are replaced with '?'
def dotbracket2constraint(line: str) -> str:
    length = len(line)
    constraint = ['?'] * length

    if length > 0:
        constraint[0] = '('
        constraint[-1] = ')'

    for i, x in enumerate(line):
        if x == '*':
            constraint[i] = '...'
            if i > 0:
                constraint[i - 1] = '('
            if i + 1 < length:
                constraint[i + 1] = ')'

    return ''.join(constraint)


# viennaRNA uses a different format for constraints, where ""
def dotbracket2constraint_vienna(line: str) -> str:
    has_external = False
    # if starting with 5 and ending with 3, it is an external motif
    if line[0] == '5' and line[len(line) - 1] == '3':
        has_external = True

    length = len(line) if not has_external else len(line) - 2
    constraint = ['.'] * length  # no constraints

    line_trimed = line

    if not has_external and length: # outermost brackets
        constraint[0] = '('
        constraint[-1] = ')'
    else:
        line_trimed = line[1:-1]

    for i, x in enumerate(line_trimed):
        if x == '*':
            constraint[i] = 'xxx'  # fill with dots, unpaired
            if i > 0: 
                constraint[i - 1] = '('  # left bracket before the dot
            if i + 1 < length:
                constraint[i + 1] = ')'  # right bracket after the dot

    return ''.join(constraint)
  

if __name__ == '__main__':
    ss = "..........((((....))))((((....))))((((...))))"
    #     012345678901234567890123456789012345678901234
    pairs = extract_pairs(ss)
    print('structure:', ss)
    print('pairs:', pairs)
    pairs_list = extract_pairs_list(ss)
    print("pairs_list: ", pairs_list)

    y1 = "(..(((..((((..((....((..((((....))....))..))))..))....))..))((..((..((..((((..((....((..((((..((....((....))))..))....))..))))..))((..((..((....))((..((((..((....((..((((....))....))..))))..))....))..))))..))..))..))((..((....((..((((..((....((....))))..))....))..))))..))))..))..)..)"
    y2 = "..((((..((..........((..((((....))....))..))..........))..))))......((..((((..((....((..((((..((....((....))))..))....))..))))..))((..((..((....))((..((((..((....((..((((....))....))..))))..))....))..))))..))..))..))((..((....((..((((..((....((....))))..))....))..))))..))............"
    dist = struct_dist(y1, y2)
    print(f'structural distance: {dist}')  # expect 21

    y1 = "....((((((((.(....)).).).)))))...."
    y2 = "....(((((((..(....)..).).)))))...."
    dist = struct_dist(y1, y2)
    print(f'structural distance: {dist}')  # expect 2
