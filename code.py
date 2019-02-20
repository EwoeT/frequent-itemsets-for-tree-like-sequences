#!/usr/bin/env python
# coding: utf-8

# In[178]:


from ete3 import Tree
import time

#Enter file path
dataset = 'file path'
#Enter minimum support count
min_sup_count = 10

# load data
def import_data(file):
    with open(file) as f:
        lines = f.readlines()
    data = []
    dat = []
    for line in lines:
        dat.append(line)

    for d in dat:
        try:
            data.append(Tree(d))
        except:
            None
    return data
#print tree
imported_data = import_data(dataset)
for data in imported_data:
    print(data)


# In[179]:


# convert nodes to lists
def to_list(file):
    imported_data = import_data(file)
    dd = []
    for data in imported_data:    
        for node in data:
            temp_data = ""
            for v in node.name:
                if v == "|":
                    temp_data = temp_data + ","
                else:
                    temp_data = temp_data + v
                    temp_data = temp_data.strip()
            dd.append(temp_data.split(","))
    return dd

to_list(dataset)     


# In[180]:


to_list1 = to_list(dataset)
to_list2 = []
for i in to_list1:
    if i not in to_list2:
        to_list2.append(i)
list_items_index = []
for index, item in enumerate(to_list2):
    tup = (item, index)
    list_items_index.append(tup)
    
list_items_index
# print(to_list2)


# In[181]:


items1 = []
for data in imported_data:
    for dat in data:
        items1.append(dat.name)

item2 = []
for i in items1:
    if i not in item2:
        item2.append(i)
        
items_index = []
for index, item in enumerate(item2):
    tup = (item, index)
    items_index.append(tup)
    
items_index


# In[182]:


#print first n trees
def print_trees(tree, n):
    for i in range(n):
        try:
            print(tree[i])
        except:
            print("no more trees to print")


# In[183]:


for tree in imported_data:
    for node in tree:
        for ii in items_index:
            if node.name in ii[0]:
                node.name = str(ii[1])
            
print_trees(imported_data, 20)


# In[184]:


# convert tree to sequence
def trees_to_sequence(data):
    forest2 = []
    for t in imported_data:
        temp = []
        tt1 = (t.write(format=9))
        tt1 = list(tt1)
        for k,v in enumerate(tt1[:-1]):
            if v=="," and tt1[k+1]=="(":
                tt1[k] = ''

        for k,v in enumerate(tt1):
            if v==")":
                tt1[k] = '$'

        for k,v in enumerate(tt1):
            if v=="(" or v==")":
                tt1[k] = ''

    #         for k,v in enumerate(tt1):
    #             if v!="(" or v!=")" or v != '$':
    #                 tt1[k] = int(v)

        for k,v in enumerate(tt1):
            if v==",":
                tt1[k] = '$'
    #         tt1 = "".join(tt1)
        temp.append(tt1[k])
        forest2.append(tt1[:-2])
        numbers = ['1','2', '3' ,'4' ,'5' ,'6', '7', '8', '9', '0']
        for fo in forest2:
            for k,v in enumerate(fo[:-1]):
                if (fo[k] in numbers) and (fo[k+1] in numbers)and (fo[k+2] in numbers):
                    fo[k] = fo[k] + fo[k+1] + fo[k+2]
                    fo[k+1] = ''
                    fo[k+2] = ''
                if (fo[k] in numbers) and (fo[k+1] in numbers):
                    fo[k] = fo[k] + fo[k+1]
                    fo[k+1] = ''
    forest3 = []
    for fo in forest2:
        fo = [x for x in fo if x != '']
        forest3.append(fo)
    return forest3

formated_sequence = trees_to_sequence(imported_data)
print(formated_sequence)


# In[185]:


#Gap_BIDE algorithm
'''
Python implementation of the Gap-Bide algorithm.
Based on
Chun Li,Jianyong Wang.
Efficiently Mining Closed Subsequences with Gap Constraints.
Siam SDM 2008.
Copyright (c) 2017 by Chun Li
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
sequence_list = []

class Gapbide:
    def __init__(self, sdb, sup, m, n):
        '''
        sdb: alist of sequences,
        sup: the minimum threshold of support,
        m,n: the gap [m,n]
        '''
        self.sdb = sdb
        self.sup = sup
        self.m = m
        self.n = n
        self.count_closed = 0
        self.count_non_closed = 0
        self.count_pruned = 0

    def run(self):
        l1_patterns = self.gen_l1_patterns()
        for pattern, sup, pdb in l1_patterns:
            self.span(pattern, sup, pdb)

    def output(self, pattern, sup, pdb):
        '''
        overide this function to output patterns to files.
        '''
        tup = (pattern, sup)
        sequence_list.append(tup)
#         print(sequence_list)


    def gen_l1_patterns(self):
        '''
        generate length-1 patterns
        '''
        pdb_dict = dict()
        for sid in range(len(self.sdb)):
            seq = self.sdb[sid]
            for pos in range(len(seq)):
                if seq[pos] in pdb_dict:
                    pdb_dict[seq[pos]].append((sid, pos, pos))
                else:
                    pdb_dict[seq[pos]] = [(sid, pos, pos)]
        patterns = []
        for item, pdb in pdb_dict.items():
            sup = len(set([i[0] for i in pdb]))
            if sup >= self.sup:
                patterns.append(([item], sup, pdb))
        return patterns

    def span(self, pattern, sup, pdb):
        (backward, prune) = self.backward_check(pattern, sup, pdb)
        if prune:
            self.count_pruned += 1
            return
        forward = self.forward_check(pattern, sup, pdb)
        if not(backward or forward):
            self.count_closed += 1
            self.output(pattern, sup, pdb)
        else:
            self.count_non_closed += 1
        pdb_dict = dict()
        for (sid, begin, end) in pdb:
            seq = self.sdb[sid]
            new_begin = end + 1 + self.m
            new_end = end + 2 + self.n
            if new_begin >= len(seq):
                continue
            if new_end > len(seq):
                new_end = len(seq)
            for pos in range(new_begin, new_end):
                if seq[pos] in pdb_dict:
                    pdb_dict[seq[pos]].append((sid, begin, pos))
                else:
                    pdb_dict[seq[pos]] = [(sid, begin, pos)]
        for item, new_pdb in pdb_dict.items():
            sup = len(set([i[0] for i in new_pdb]))
            if sup >= self.sup:
                # add new pattern
                new_pattern = pattern[:]
                new_pattern.append(item)
                self.span(new_pattern, sup, new_pdb)

    def forward_check(self, pattern, sup, pdb):
        sids = {}
        forward = False
        for (sid, begin, end) in pdb:
            seq = self.sdb[sid]
            new_begin = end + 1 + self.m
            new_end = end + 2 + self.n
            if new_begin >= len(seq):
                continue
            if new_end > len(seq):
                new_end = len(seq)
            for pos in range(new_begin, new_end):
                if seq[pos] in sids:
                    sids[seq[pos]].append(sid)
                else:
                    sids[seq[pos]] = [sid]
        for item, sidlist in sids.items():
            seq_sup = len(set(sidlist))
            if seq_sup == sup:
                forward = True
                break
        return forward

    def backward_check(self, pattern, sup, pdb):
        sids = {}
        backward = False
        prune = False
        for (sid, begin, end) in pdb:
            seq = self.sdb[sid]
            new_begin = begin - self.n - 1
            new_end = begin - self.m
            if new_end < 0:
                continue
            if new_begin < 0:
                new_begin = 0
            for pos in range(new_begin, new_end):
                if seq[pos] in sids:
                    sids[seq[pos]].append(sid)
                else:
                    sids[seq[pos]] = [sid]
        for item, sidlist in sids.items():
            seq_sup = len(set(sidlist))
            uni_sup = len(sidlist)
            if uni_sup == len(pdb):
                prune = True
            if seq_sup == sup:
                backward = True
            if backward and prune:
                break
        return (backward, prune)


# In[186]:


g = Gapbide(formated_sequence, min_sup_count, 0, 0)
p = g.run()
sequence_list


# In[187]:


# get list of all items
all_items = []
for lst in to_list1:
    for l in lst:
        all_items.append(l)
        
all_items
# count item frequency
from collections import Counter
counts = Counter(all_items)
co = {k:v for k,v in counts.items()}

#prune infrequent items from freq dictionary
freq_co = {k:v for k,v in co.items() if v >= min_sup_count}
freq_co

# make list of frequent items
freq_list = []
for item in freq_co:
    freq_list.append(item)
freq_co


# In[188]:


# __author__ = 'Eric Naeseth <eric@naeseth.com>'
# __copyright__ = 'Copyright © 2009 Eric Naeseth'
# __license__ = 'MIT License

from collections import defaultdict, namedtuple
class FPTree(object):
    """
    An FP tree.

    This object may only store transaction items that are hashable
    (i.e., all items must be valid as dictionary keys or set members).
    """

    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        # The root node of the tree.
        self._root = FPNode(self, None, None)

        # A dictionary mapping items to the head and tail of a path of
        # "neighbors" that will hit every node containing that item.
        self._routes = {}

    @property
    def root(self):
        """The root node of the tree."""
        return self._root

    def add(self, transaction):
        """Add a transaction to the tree."""
        point = self._root

        for item in transaction:
            next_point = point.search(item)
            if next_point:
                # There is already a node in this tree for the current
                # transaction item; reuse it.
                next_point.increment()
            else:
                # Create a new point and add it as a child of the point we're
                # currently looking at.
                next_point = FPNode(self, item)
                point.add(next_point)

                # Update the route of nodes that contain this item to include
                # our new node.
                self._update_route(next_point)

            point = next_point

    def _update_route(self, point):
        """Add the given node to the route through all nodes for its item."""
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point # route[1] is the tail
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            # First node for this item; start a new route.
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        """
        Generate one 2-tuples for each item represented in the tree. The first
        element of the tuple is the item itself, and the second element is a
        generator that will yield the nodes in the tree that belong to the item.
        """
        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        """
        Generate the sequence of nodes that contain the given item.
        """

        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        """Generate the prefix paths that end with the given item."""

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print('Tree:')
        self.root.inspect(1)
        
        print('\nRoutes:')
        for item, nodes in self.items():
            print('  %r' % item)
            for node in nodes:
                print('    %r' % node)

def conditional_tree_from_paths(paths):
    """Build a conditional FP-tree from the given prefix paths."""
    tree = FPTree()
    condition_item = None
    items = set()

    # Import the nodes in the paths into the new tree. Only the counts of the
    # leaf notes matter; the remaining counts will be reconstructed from the
    # leaf counts.
    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                # Add a new node to the tree.
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    # Calculate the counts of the non-leaf nodes.
    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    return tree

class FPNode(object):
    """A node in an FP tree."""

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        """Add the given FPNode `child` as a child of this node."""

        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        """
        Check whether this node contains a child node for the given item.
        If so, that node is returned; otherwise, `None` is returned.
        """
        try:
            return self._children[item]
        except KeyError:
            return None

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        """The tree in which this node appears."""
        return self._tree

    @property
    def item(self):
        """The item contained in this node."""
        return self._item

    @property
    def count(self):
        """The count associated with this node's item."""
        return self._count

    def increment(self):
        """Increment the count associated with this node's item."""
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        """True if this node is the root of a tree; false if otherwise."""
        return self._item is None and self._count is None

    @property
    def leaf(self):
        """True if this node is a leaf in the tree; false if otherwise."""
        return len(self._children) == 0

    @property
    def parent(self):
        """The node's parent"""
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        """
        The node's neighbor; the one with the same value that is "to the right"
        of it in the tree.
        """
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        """The nodes that are children of this node."""
        return tuple(self._children.values())

    def inspect(self, depth=0):
        print(('  ' * depth) + repr(self))
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)


# In[189]:


def find_with_suffix(tree, suffix):
    for item, nodes in tree.items():
        support = sum(n.count for n in nodes)
        if support >= min_sup_count and item not in suffix:
            # New winner!
            found_set = [item] + suffix
            yield (found_set, support) if True else found_set

            # Build a conditional tree and recursively search for frequent
            # itemsets within it.
            cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
            for s in find_with_suffix(cond_tree, found_set):
                yield s # pass along the good news to our caller

#     # Search for frequent itemsets, and yield the results we find.
#     for itemset in find_with_suffix(master, []):
#         yield itemset


# In[190]:


master = FPTree()      

# remove items below minsup from node
for idx, line in enumerate(to_list1[:]):
    for item in to_list1[idx][:]:
        if item not in freq_list:
            to_list1[idx].remove(item)
    to_list1[idx] = sorted(to_list1[idx], key=lambda x: freq_list.index(x))
    master.add(to_list1[idx])


# In[191]:


# Search for frequent itemsets with support count more than 
freq_itemset = []
for itemset in find_with_suffix(master, []):
    freq_itemset.append(itemset[0])


# In[192]:


freq_itemset


# In[193]:


# for freq in freq_itemset:
for item in list_items_index:
    print(item)


# In[194]:



for item in sequence_list:
    for k,v in enumerate(item[0]):
        for i in list_items_index:
            try :
                if int(v) == i[1]:
                    item[0][k] = i[0]
#                     print(v)
            except:
                None
seqq = sorted(sequence_list, key=lambda tup: tup[1], reverse=True)

for item in seqq:
    print(item)


# In[195]:


seqq

