import itertools
from queue import Queue
import collections
import sys, getopt

#parses file to list of edges per graph
def file_to_graphs(file):
    f = open(file, "r")
    a = -1
    edges = []
    graphs = []
    for line in f : 
        data = line.split()
        if data[0] == a:
            edges.append(data[1:])
        else:
            edges = []
            edges.append(data[1:])
            graphs.append(edges)
        a = data[0]
    return graphs

#print(file_to_graphs("/home/fatemeh/Bureau/Stage/graph.txt"))

#parses list of edges to adjacency matrix
def to_adjacency(edges,n):
    size = n
    res = [ [ 0 for i in range(size+1) ] for j in range(size+1) ] 
    for edge in edges:
        res[int(edge[0])][int(edge[1])] = 1 
        res[int(edge[1])][int(edge[0])] = 1
    return res

#print(to_adjacency([ [0,1], [0,6], [0,8], [1,4], [1,6], [1,9], [2,4], [2,6], [3,4], [3,5],
#[3,8], [4,5], [4,9], [7,8], [7,9] ]))

#parses list of edges to graphs in form of adjacency matrix
def to_list_of_matrices(graphs,n):
    matrices = []
    for graph in graphs:
        matrices.append(to_adjacency(graph,n-1))
    return matrices

#print(to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/target.txt"),5))

#checks if two graphs are equal
def graphsequal(g1,g2):
    for i in range (len(g1)) :
        for j in range (len(g1)) :
            if g1[i][j] != g2[i][j] :
                return False
    return True

#returns list of neighbours of a vertex in a graph
def neighbours(node,E):
    n = []
    for i in range(0,len(E[node])):
        if E[node][i]==1:
            n.append(i)
    return n

#BFS, un classique ! ;-)
def bfs(gprim,start):
    visited = []
    for x in range(0,len(gprim)):
        visited.append(0)
    queue = []
    all_paths = []

    queue.append([start])
    all_paths.append([start])
    
    i=0
    while queue:
        path = queue.pop(0)
        node = path[-1]
        visited[node]=1
        for adjacent in neighbours(node,gprim):
            #print(adjacent)
            if visited[adjacent]==0:
                i = i + 1 
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)
                all_paths.append(new_path)
    
    return all_paths

# returns cartesian product of mappings at each instant
def creat_all_mappings(E,Eprim,t):
    list_mappings = []
    for i in range (0,len(E)):
        list_mappings.append(creat_all_mappings_for_single_graph(Eprim[t+i],E[i]))
    
    return  list(itertools.product(*list_mappings))

def get_key(val,my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

#merges each mapping into one and checks if the mapping is valid by comparing it before and after for repeated keys and values
def mergeDict(dict1, dict2):
    dict3 = dict1.copy()
    dict3.update(dict2)
    for key in dict3.keys():
        if key in dict2.keys() and key in dict1.keys():
            if  dict1[key] != dict3[key]:
                return -1
        values = dict3.values()
        duplicates = [item for item, count in collections.Counter(values).items() if count > 1]
        if duplicates != []:
            return -1

    return dict3

#print(mergeDict({0:2, 1:1, 2:0},{2:0, 4:5, 3:5}))
#print(mergeDict(mergeDict({0:2, 1:1, 2:0},{2:0, 4:5, 3:6}),{4:5, 5:7}))

#generalizes the function before for all instances
def is_valid(mapping):
    dict3 = mapping[0].copy()
    for dict2 in enumerate(mapping):
        dict3 = mergeDict(dict3, dict2[1])
        if(dict3 == -1):
            return -1
    
    return dict3
        
#print(is_valid(({0: 2, 1: 1, 2: 0}, {2: 0, 3: 6, 4: 5}, {4: 5, 5: 7})))

#Calculates all possible isomorphism by finding paths of desired length all over the graph at the instance
def creat_all_mappings_for_single_graph(gprim,g):
    testing_gprim = []
    testing_g = []
    list_mappis = []
    paths_in_gprim = []

    for i in range(len(gprim)):
        for j in range(len(gprim)) :
            if gprim[i][j] == 1 and i not in testing_gprim:
                testing_gprim.append(i)
    
    for i in range(len(g)):
        for j in range(len(g)) :
            if g[i][j] == 1 and i not in testing_g:
                testing_g.append(i)
                if len(neighbours(i,g))==1:
                    start = i

    g_to_paths = bfs(g,start)
    g_to_path_side_1 = max((x) for x in g_to_paths)
    
    for v in testing_gprim:
       for chemin in (bfs(gprim,v)):
            paths_in_gprim.append(chemin)
    
    potential_paths = []
    for p in paths_in_gprim:
        if len(p) == len(testing_g):
            potential_paths.append(p)
    
    for p in potential_paths:
        mapi = dict()
        for i in range(len(p)):
            mapi[g_to_path_side_1[i]] = p[i]
        list_mappis.append(mapi)
    
    return list_mappis

#print(creat_all_mappings_for_single_graph([[0,1,1,0,0],[1,0,0,1,1],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0]],[[0,1,0],[1,0,1],[0,1,0]]))
#print(creat_all_mappings(example_pattern,example_target,0))

#Final algorithm takes pattern and target and returns possible isomorphism if available
def naive_algo(E,Eprim):
    for t in range (0,len(Eprim)):
        if (t + len(E)) > len(Eprim):
            print("No possible isomorphism")
            return -1
        mappings = creat_all_mappings(E,Eprim,t)
        for m in mappings:
            merged = is_valid(m)
            if merged!=-1:
                print("Found isomorphism with mapping " + str(merged) + " at instance " + str(t+1) + " of the pattern.")
                return (merged,t+1)
    
    print("No possible isomorphism")
    return -1

example_target_1 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target1.txt"),9)
example_pattern_1 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern1.txt"),6)

example_target_2 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target2.txt"),4)
example_pattern_2 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern2.txt"),4)

example_target_3 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target3.txt"),5)
example_pattern_3 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern3.txt"),5)

print(naive_algo(example_pattern_1,example_target_1))


'''
def computeLPSArray(E, T, lps): 
    len = 0 # length of the previous longest prefix suffix 
  
    lps[0] # lps[0] is always 0 
    i = 1
  
    # the loop calculates lps[i] for i = 1 to M-1 
    while i < T: 
        if graphsequal(E[i],E[len]): 
            len += 1
            lps[i] = len
            i += 1
        else: 
            # This is tricky. Consider the example. 
            # AAACAAAA and i = 7. The idea is similar  
            # to search step. 
            if len != 0: 
                len = lps[len-1] 
  
                # Also, note that we do not increment i here 
            else: 
                lps[i] = 0
                i += 1

def KMPSearch(E, Eprim,Vprim): 
    M = len(E) 
    N = len(Eprim) 
    result = []
    mapi = dict()
  
    # create lps[] that will hold the longest prefix suffix  
    # values for pattern 
    lps = [0]*M 
    j = 0 # index for pat[] 
  
    # Preprocess the pattern (calculate lps[] array) 
    computeLPSArray(E, M, lps) 
  
    i = 0 # index for txt[] 
    while i < N: 
        (j,mapi) = naive_algo(Eprim[i],E[j],i)
        if j == M: 
            result.append(((i-j),mapi))
            print(str(Eprim[i-j]))
            print "Found pattern at index " + str(i-j) 
            j = lps[j-1] 
            i = i + 1

        # mismatch after j matches 
        elif i < N and j < M: 
            # Do not match lps[0..lps[j-1]] characters, 
            # they will match anyway 
            if j != 0: 
                j = lps[j-1] 
            else: 
                i += 1
    
    return result
'''
