import re
import argparse

from utils.structure import extract_pairs_list


PATTERN = re.compile(r"^[\(]+\*[\)]+$")


class PairNode:

    def __init__(self, pair, type='unknown'):
        self.pair = pair
        self.type = type  # no external loop type
        self.unpaired_bases = []
        self.children = []
        self.child_id = None
        self.parent = None
        self._dotbracket = None
        self._len = None
        self.motif_child = None  # point to another motif, used for motif hierarchy by structure decomposition

    @staticmethod
    def string2node(structure, root_type='E'):
        """
        Convert dotbracket representation of structure or motif to a tree of PairNodes.
        root_type: The type of the root node ('E' for external loop of structure, 'p' for leaf / root node of motif)
        """
        map_of_nodes = dict()  # To keep track of nodes by their pairs
        if root_type == 'E':
            root = PairNode((-1, len(structure)), type=root_type)
        elif root_type == 'p':
            root = PairNode((-1, -1), type=root_type)
        map_of_nodes[root.pair] = root
        stack = [root]
        for i, c in enumerate(structure):
            if c == '(':
                stack.append(PairNode((i, -1)))
            elif c == ')':
                pair_node = stack.pop()
                pair_node.pair = (pair_node.pair[0], i)
                assert pair_node.pair not in map_of_nodes, f"Duplicate pair found: {pair_node.pair}"
                map_of_nodes[pair_node.pair] = pair_node
                stack[-1].children.append(pair_node)
                pair_node.parent = stack[-1]
                pair_node.child_id = len(stack[-1].children) - 1
                assert pair_node.pair[0] >= 0
                if pair_node.pair[1] - pair_node.pair[0] == 2:  # leaf node
                    pair_node.type = 'p'
                    # print("Leaf node:", pair_node.pair)
                    pair_node.unpaired_bases = []
                elif len(pair_node.children) == 0:  # hairpin
                    pair_node.type = 'H'
                    pair_node.unpaired_bases = [pair_node.pair[1] - pair_node.pair[0] - 1]
                elif len(pair_node.children) == 1:  # internal loop
                    pair_node.type = 'I'
                    pair_node.unpaired_bases = [
                        pair_node.children[0].pair[0] - pair_node.pair[0] - 1,
                        pair_node.pair[1] - pair_node.children[0].pair[1] - 1,
                    ]
                else:  # multi-loop
                    pair_node.type = 'M'
                    pair_node.unpaired_bases = [pair_node.children[0].pair[0] - pair_node.pair[0] - 1]
                    for j in range(len(pair_node.children) - 1):
                        pair_node.unpaired_bases.append(
                            pair_node.children[j + 1].pair[0] - pair_node.children[j].pair[1] - 1
                        )
                    pair_node.unpaired_bases.append(
                        pair_node.pair[1] - pair_node.children[-1].pair[1] - 1
                    )
        assert len(stack) == 1
        # process root node
        if not root.children:  # no children means there is no base pairs
            root.unpaired_bases = [len(structure)]
        else:
            root.unpaired_bases = [root.children[0].pair[0] - root.pair[0] - 1]
            for j in range(len(root.children) - 1):
                root.unpaired_bases.append(
                    root.children[j + 1].pair[0] - root.children[j].pair[1] - 1
                )
            root.unpaired_bases.append(root.pair[1] - root.children[-1].pair[1] - 1)
        return root, map_of_nodes

    @property
    def dotbracket(self):
        # if self._dotbracket is None:
        self._dotbracket = self.to_dotbracket()
        return self._dotbracket

    def to_dotbracket(self):
        result = ""
        if self.type == 'p':
            if self.child_id is None:
                # assert len(self.children) == 1, str(len(self.children))
                result = self.children[0].dotbracket
            else:
                result = "(*)"
        elif self.type == 'H':
            result = "(" + "." * self.unpaired_bases[0] + ")"
        else:
            result = ""
            if self.type != "E":
                result += "("
            result += "." * self.unpaired_bases[0]
            for i, child in enumerate(self.children):
                result += child.to_dotbracket()
                assert i + 1 < len(self.unpaired_bases)
                result += "." * self.unpaired_bases[i + 1]
            if self.type != "E":
                result += ")"
        return result

    def __str__(self):
        return self.dotbracket

    def __len__(self):
        def compute_length(node):
            if node.type == 'p':
                if node.child_id is not None:  # leaf node
                    self._len += 2  # no children
                    return
                else:  # (virtual) root node for motif which does not exist actually
                    self._len += 0
            elif node.type == 'E':
                self._len += sum(node.unpaired_bases)
            elif node.type in ['H', 'I', 'M']:
                self._len += sum(node.unpaired_bases) + 2
            else:
                assert False, f"Unknown node type: {node.type}"

            for child in node.children:
                compute_length(child)

        # if self._len is None:
        self._len = 0
        compute_length(self)

        return self._len

    def len_verbose(self):
        def compute_length(node):
            print(f"Computing length for node {node.pair} of type {node.type}")
            for child in node.children:
                compute_length(child)
            if node.type == 'p':
                print('p node:', node.child_id, node.children)
                if node.child_id is not None:  # leaf node
                    self._len += 2  # no children
                    return
                else:  # (virtual) root node for motif which does not exist actually
                    self._len += 0
            elif node.type == 'E':
                self._len += sum(node.unpaired_bases)
            elif node.type in ['H', 'I', 'M']:
                self._len += sum(node.unpaired_bases) + 2
            else:
                assert False, f"Unknown node type: {node.type}"
            print(f"Current length for node {node.pair} is {self._len}")

        # if self._len is None:
        self._len = 0
        compute_length(self)

        return self._len

    def preorder(self):
        """Yield nodes in pre-order (excluding 'E' and 'p' types)."""
        def traverse(node):
            if node.type != 'E' and node.type != 'p':
                yield node
            for child in node.children:
                yield from traverse(child)
        return traverse(self)

    @property
    def postordered_motifs(self):
        """Yield motifs in post-order by tracking self.motif_child"""
        def traverse(node):
            if node.child_id is None:
                yield node
            if node.motif_child is not None:
                # yield node.motif_child
                yield from node.motif_child.postordered_motifs
            else:
                for child in node.children:
                    yield from traverse(child)
        return traverse(self)

    def has_stack(self):
        def traverse(node):
            if node.type == 'I' and node.unpaired_bases == [0, 0]:
                return True
            for child in node.children:
                if traverse(child):
                    return True
            return False

        return traverse(self)

    @property
    def loop_count(self):
        def traverse(node):
            total = 0
            if node.type != 'p':
                total += 1
            for child in node.children:
                total += traverse(child)
            return total
        return  traverse(self)
    

class MotifNode:
    "each node in the motif tree"
    def __init__(self, motif, parent=None):
        self.motif = motif
        self.children = []
        self.parent = parent

    def postorder(self):
        """Yield nodes in post-order."""
        for child in self.children:
            yield from child.postorder()
        yield self

    def preorder(self):
        """Yield nodes in pre-order."""
        yield self
        for child in self.children:
            yield from child.preorder()

    def to_dotbracket(self):
        if not self.children:  # leaf node
            return self.motif.to_dotbracket()
        dotbracket_self = self.motif.to_dotbracket()
        dotbracket_children = [child.to_dotbracket() for child in self.children]
        # each "*" in dotbracket_self corresponds to a disjoint child, replace each "*" with a child's dotbracket
        for child_dotbracket in dotbracket_children:
            dotbracket_self = dotbracket_self.replace("*", child_dotbracket, 1)
        return dotbracket_self

    def __len__(self):
        return len(self.motif)


def build_motiftree(pair_node: PairNode) -> MotifNode:
    """Build the motif tree, similar to postordered_motifs but output a tree"""
    current_node = MotifNode(pair_node) if pair_node.child_id is None else None
    def get_motif_children(current, parent):
        if current.motif_child is not None:
            new_node = MotifNode(current.motif_child, parent)
            parent.children.append(new_node)
            new_node.parent = parent
            get_motif_children(current.motif_child, new_node)
        else:
            for child in current.children:
                get_motif_children(child, parent)

    if current_node:
        # for child in self.children:
        get_motif_children(pair_node, current_node)
    return current_node


def find_stack_from_boundary(boundary_nodes):  # bpairs: boundary pairs
    root_node = boundary_nodes[0]
    leaf_bpairs = set(node.pair for node in boundary_nodes[1:])
    stack_boundary = []
    stack_internal = []
    def traverse(node):
        if node.pair in leaf_bpairs:
            return
        if node.type == 'I' and node.unpaired_bases == [0, 0]:
            adjacent_boundary = False
            if node == root_node or node.children[0].pair in leaf_bpairs:
                adjacent_boundary = True
            if adjacent_boundary:
                stack_boundary.append(node)
            else:
                stack_internal.append(node)
        for child in node.children:
            traverse(child)
    traverse(root_node)
    return stack_boundary, stack_internal


def find_stack_from_motif(motif_root):
    # assert len(motif_root.children) == 1, f"Motif root should have exactly one child: {motif_root}, {motif_root.children}"
    start_node = motif_root.children[0]
    stack_boundary = []
    stack_internal = []
    def traverse(node):
        # print('node type:', node.type)
        if node.type == 'p':
            return
        if node.type == 'I' and node.unpaired_bases == [0, 0]:
            adjacent_boundary = False
            if node == start_node or node.children[0].type == 'p':
                adjacent_boundary = True
            if adjacent_boundary:
                stack_boundary.append(node)
            else:
                stack_internal.append(node)
        for child in node.children:
            traverse(child)
    traverse(start_node)
    return stack_boundary, stack_internal


def find_stack_from_match(match):
    # assert len(motif_root.children) == 1, f"Motif root should have exactly one child: {motif_root}, {motif_root.children}"
    start_node = match[0]
    stack_boundary = []
    stack_internal = []
    def traverse(node):
        # print('node type:', node.type)
        if node in match[1:]:  # reach to the bottom
            return
        if node.type == 'I' and node.unpaired_bases == [0, 0]:
            adjacent_boundary = False
            if node == start_node or node.children[0] in match[1:]:
                adjacent_boundary = True
            if adjacent_boundary:
                stack_boundary.append(node)
            else:
                stack_internal.append(node)
        for child in node.children:
            traverse(child)
    traverse(start_node)
    return stack_boundary, stack_internal


def match_search(motif_small, motif_large):
    """Find the match for a small motif within a larger motif."""
    match_list = []

    if len(motif_small) > len(motif_large):
        return match_list

    # print('start to search match')

    def match_recursive(node_small, node_large):
        if node_small.type == 'p':
            nodes_matched.append(node_large)
            return True  # A leaf node matches any node
        if node_small.type != node_large.type:
            return False
        if node_small.unpaired_bases != node_large.unpaired_bases:
            return False
        for child_small, child_large in zip(node_small.children, node_large.children):
            if not match_recursive(child_small, child_large):
                return False
        return True

    if motif_small.type == 'E':
        # print('start to search E')
        nodes_matched = [motif_large]
        if match_recursive(motif_small, motif_large):
            match_list.append(nodes_matched)

    if motif_small.type == 'p':  # try to match every pairnode within motif_large to motif_small
        # print('start to search p')
        # assert len(motif_small.children) == 1, "Small motif should have exactly one child"
        for node in motif_large.preorder():
            # print("Trying to match pair:", node.pair)
            nodes_matched = [node]
            if match_recursive(motif_small.children[0], node):
                match_list.append(nodes_matched)

    return match_list


def split_motif_at_stack(motif_root, stack_node):
    """Split a motif into its constituent parts."""
    len_before_split = len(motif_root)
    # remove the child node from the stack
    child_of_stack = stack_node.children[0]
    stack_node.type = 'p'  # convert stack to leaf
    stack_node.children = []  # remove children from stack node
    # print("stack_node.child_id:", stack_node.child_id)
    # print("stack_node.pair:", stack_node.pair)
    # print("stack_node.parent.pair:", stack_node.parent.pair)

    # make a new root node for the child
    new_root = PairNode((-1, -1), 'p')
    new_root.children.append(child_of_stack)
    stack_node.motif_child = new_root
    assert len_before_split == len(motif_root) + len(new_root), f"Length mismatch after split: {len_before_split} != {len(motif_root)} + {len(new_root)}"
    return motif_root, new_root


def hierarchical_decompose(structure_root, motif_candidates, motifs_split=None):
    """Decompose a structure into its hierarchical motifs."""
    if motifs_split is None:
        motifs_split = []
    # first match, then find stack in the matched motifs, finally split. 
    for i, motif in enumerate(motif_candidates):
        match_list = match_search(motif, structure_root)
        if not match_list:
            continue
        whether_split = False
        for matched_motif in match_list:
            print("Found match for motif:", motif.dotbracket)
            # break
            # first_match = match_list[0]
            print("Motif to split:", matched_motif[0].type, [match.pair for match in matched_motif])
            stack_boundary, stack_internal = find_stack_from_match(matched_motif)
            print("boundary stacks:", [node.pair for node in stack_boundary])
            print("internal stacks:", [node.pair for node in stack_internal])
            # if stack_internal or stack_boundary:
                # take the middle one instead of the first one
                # stack_node = stack_internal[len(stack_internal) // 2] if stack_internal else stack_boundary[len(stack_boundary) // 2]
            if stack_internal:
                stack_node = stack_internal[(len(stack_internal) - 1) // 2]
                len_before_split = len(structure_root)
                len_lower_after_split = len(stack_node.children[0])
                len_upper_after_split = len_before_split - len_lower_after_split
                loop_count_before_split = structure_root.loop_count
                loop_count_lower = stack_node.children[0].loop_count
                loop_count_upper = loop_count_before_split - loop_count_lower - 1
                whether_split = True
                if min(len_lower_after_split, len_upper_after_split) < 5:
                    print("Skipping split, not enough size:", min(len_lower_after_split, len_upper_after_split))
                    whether_split = False
                if min(loop_count_lower, loop_count_upper) < 2:
                    print("Skipping split, not enough loop count:", min(loop_count_lower, loop_count_upper))
                    whether_split = False

                if whether_split:
                    # add boundary pairs of matched motif to the list of split motifs
                    bpairs = [match.pair for match in matched_motif]
                    motifs_split.append(bpairs)
                    print("Split motif at stack:", stack_node.pair)
                    print()
                    structure_root, new_root = split_motif_at_stack(structure_root, stack_node)
                    assert len(structure_root) == len_upper_after_split, f"Length mismatch after split: {len(structure_root)} != {len_upper_after_split}"
                    assert len(new_root) == len_lower_after_split, f"Length mismatch after split: {len(new_root)} != {len_lower_after_split}"
                    assert structure_root.loop_count == loop_count_upper, f"Loop count mismatch after split: {structure_root.loop_count} != {loop_count_upper}"
                    assert new_root.loop_count == loop_count_lower, f"Loop count mismatch after split: {new_root.loop_count} != {loop_count_lower}"
            #         # recursively split, search from the current motif that can match
                    if len(new_root) > 14:
                        hierarchical_decompose(new_root, motif_candidates[i:], motifs_split)
                    if len(structure_root) > 14:
                        hierarchical_decompose(structure_root, motif_candidates[i:], motifs_split)
                    return structure_root, motifs_split
    return structure_root, motifs_split


def test_motifs():
    with open('data/easy_motifs.txt') as f:
        for line in f:
            motif = line.strip()
            if not line:
                continue
            print("Processing motif:", motif)
            if motif.startswith('5'):
                root, _ = PairNode.string2node(motif[1: -1], 'E')
            else:
                root, _ = PairNode.string2node(motif, 'p')
            dot_bracket = root.dotbracket
            if motif.startswith('5'):
                dot_bracket = '5' + dot_bracket + '3'
            # print("Dot-bracket string:", dot_bracket)
            assert dot_bracket == motif, f"Dot-bracket mismatch for {motif}: {dot_bracket}"


def test_structures():
    with open('data/eterna100.txt') as f:
        for line in f:
            structure = line.strip()
            if not structure:
                continue
            print("Processing structure:", structure)
            root, _ = PairNode.string2node(structure, 'E')
            dot_bracket = root.dotbracket
            # print("Dot-bracket string:", dot_bracket)
            assert dot_bracket == structure, f"Dot-bracket mismatch for {structure}"


def simple_verification():
    y = "............((....))...((...((....))..((...((....))...))..((....))...))..((....))........................"
    root, _ = PairNode.string2node(y)
    dot_bracket = root.dotbracket
    print("Dot-bracket string:", dot_bracket)
    assert len(root) == len(y)

    m = "(((*))..(*)..(*))"
    root, _ = PairNode.string2node(m)
    dot_bracket = root.dotbracket
    print("Dot-bracket string:", dot_bracket)
    assert dot_bracket == m


def test_match():
    motif_list = ["(((*))..(*)..(*))", "((*)..((*))..(*))", "((*)..(*)..((*)))", "(((*)..(*)..(*)))"]
    # motif = "(((*))..(*)..(*))"
    y_star = "........(((((((((((....))))..((((....))))..((((....)))))))))))..(((((((...))))..((((...))))..((((...)))))))........."

    for motif in motif_list:
        print("Processing motif:", motif)
        motif_root, _ = PairNode.string2node(motif, 'p')
        y_star_root, _ = PairNode.string2node(y_star, 'E')

        match_list = match_search(motif_root, y_star_root)
        for nodes_matched in match_list:
            print("Found match:", [node.pair for node in nodes_matched])
        print()


# choose motifs to split at, based on some criteria:
# 1. has stack
# 2. has at least three loops
def select_motifs(motifs, min_loop_count=3):
    selected = []
    for motif in motifs:
        if motif.startswith('5'):
            motif_root, _ = PairNode.string2node(motif, 'E')
        else:
            motif_root, _ = PairNode.string2node(motif, 'p')
        if motif_root.has_stack() and motif_root.loop_count >= min_loop_count:
            selected.append(motif_root)
    return selected


def is_helix(dotbracket):
    return True if PATTERN.match(dotbracket) else False


def get_selected_motifs(path_motifs, min_loop_count=3):
    motif_list = []
    with open(path_motifs) as f:
        for line in f:
            motif = line.strip()
            if not line:
                continue
            motif_list.append(motif)
    selected_motifs = select_motifs(motif_list, min_loop_count)
    # sort according to length
    selected_motifs.sort(key=lambda x: len(x), reverse=True)
    helix_motifs = [motif for motif in selected_motifs if is_helix(motif.dotbracket)]
    non_helix_motifs = [motif for motif in selected_motifs if not is_helix(motif.dotbracket)]
    assert len(helix_motifs) + len(non_helix_motifs) == len(selected_motifs)
    print("count of helix motifs:", len(helix_motifs))
    print("count of non-helix motifs:", len(non_helix_motifs))
    selected_motifs = helix_motifs + non_helix_motifs
    # selected_motifs = helix_motifs # + non_helix_motifs
    print("count of selected motifs:", len(selected_motifs))
    return selected_motifs


def decompose(structure, selected_motifs):
    y_root, _ = PairNode.string2node(structure, 'E')
    root_decomposed, _ = hierarchical_decompose(y_root, selected_motifs)

    motif_tree = build_motiftree(root_decomposed)
    # motifs_from_tree = list(motif_tree.preorder())
    dotbracket_recovered = motif_tree.to_dotbracket()
    assert dotbracket_recovered == structure, f"Dot-bracket mismatch: {dotbracket_recovered} != {structure}"
    return motif_tree


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RNA Motif Design")
    parser.add_argument("--motifs", type=str, default="data/easy_motifs.txt", help="Path to the motifs file")
    parser.add_argument("--min_loop_count", type=int, default=3, help="Minimum loop count")
    parser.add_argument("-i", "--index_inspect", type=int, default=None, help="Index to inspect")
    args = parser.parse_args()

    test_motifs()
    test_structures()
    test_match()

    import time
    start_time = time.time()
    selected_motifs = get_selected_motifs(args.motifs, args.min_loop_count)
    print("count of selected motifs:", len(selected_motifs))
    end_time = time.time()
    print(f"Time taken to select motifs: {end_time - start_time:.2f} seconds")