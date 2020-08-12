import itertools
from queue import Queue
import collections
import sys, getopt
import timeit
import numpy as np
import random
import os
import os.path
import matplotlib.pyplot as plt
import math
import copy

def make_random_path(nbr_vertices,length):
    l = random.sample(range(0,nbr_vertices), length)
    return l

def generate_uniform_pattern(nbr_length_per_instance,nbr_instance,nbr_vertices):
    f = open("/home/fatemeh/Bureau/Stage/pattern_"+ str(nbr_instance) +"inst_"+ str(nbr_length_per_instance)+"vert.txt", "w")
    for j in range(1,(nbr_instance+1)):
        l = make_random_path(nbr_vertices,nbr_length_per_instance)
        for i in range(len(l)):
            if i!=len(l)-1:
                f.write(str(j) + " " + str(l[i]) + " " + str(l[i+1]))
                f.write("\n")
    f.close()
    return f


def make_binary_tree_form_arbogen(trees,n,file_name):
    f2 = open(file_name, "w")
    for fname in trees:
        f = open(fname, "r")
        fi = f.readline()
        fi = f.readline()
        i = 0
        while fi[0]==" ":
            fi = f.readline()
            i=i+1
        line = fi.split()
        last = line[2]
        last1 = last[:len(last)-1]
        f2.write(str(trees.index(fname)+1) + " " + str(line[0]) + " " + str(last1))
        f2.write("\n") 
        for j in range(0,i-2):
            fi = f.readline()
            line = fi.split()
            last = line[2]
            last1 = last[:len(last)-1]
            f2.write(str(trees.index(fname)+1) + " " + str(line[0]) + " " + str(last1))
            f2.write("\n")    
    
    f2.close()
    return f2

def generate_random_target_stream(number_of_vertex_per_instance,number_of_instances):
    l = []
    for i in range(number_of_instances+1):
        while True:  
            os.system('arbogen -o ~/Bureau/Stage/arbre'+ str(number_of_vertex_per_instance) + '_' + str(i) +' -otype dot ~/arbogen-master/examples/unarybinary'+str(number_of_vertex_per_instance)+'.spec')
            f =  'arbre' + str(number_of_vertex_per_instance) + '_' + str(i) + '.dot'
            if(os.path.exists(os.path.join('/home/fatemeh/Bureau/Stage/', f))):  
               break  
        l.append('/home/fatemeh/Bureau/Stage/'+f)
    r = make_binary_tree_form_arbogen(l,number_of_instances,"/home/fatemeh/Bureau/Stage/target_"+ str(number_of_instances) +"inst_"+ str(number_of_vertex_per_instance)+"vert.txt")
    print("Done ! target stream created.")
    return r

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


def file_to_paths(file):
    f = open(file, "r")
    a = -1
    edges = []
    graphs = []
    i = 0
    for line in f :
        data = line.split()
        if data[0] == a:
            edges.append(data[2])
        else:
            edges = data[1:]
            graphs.append(edges)
        a = data[0]
    return graphs

print(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern3.txt"))
print(file_to_paths("/home/fatemeh/Bureau/Stage/example_pattern3.txt"))

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


def pathequal(p1,p2):
    for i in range (len(p1)) :
        if(p1[i] != p2[i]):
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
def bfs_k_length(l,gprim,start):
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
                #print(all_paths)
                i = i + 1 
                new_path = list(path)
                if (len(new_path)+1<=l):
                    new_path.append(adjacent)
                    queue.append(new_path)
                    if (len(new_path)==l):
                        all_paths.append(new_path)
                else:
                    return all_paths[1:]
    
    return all_paths[1:]
#print(bfs_k_length(2,[[0,1,1,0,0,0,0,0],[1,0,0,1,0,0,0,1],[1,0,0,1,1,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,1],[0,0,0,0,1,0,0,0],[0,1,0,0,1,0,0,0]],0))
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

def creat_all_mappings(E,Eprim,t):
    list_mappings = []
    for i in range (0,len(E)):
        list_mappings.append(creat_all_mappings_for_single_graph(Eprim[t+i],E[i]))
    
    #print("me voilaaaa" + str(list_mappings))
    #print("me voilaaaaaaaaa" + str(list(itertools.product(*list_mappings))))
    
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
def deep_list(x):
    """fully copies trees of tuples to a tree of lists.
       deep_list( (1,2,(3,4)) ) returns [1,2,[3,4]]"""
    if type(x)!=type( () ):
        return x
    return map(deep_list,x)



def is_valid(mapping):
    result = deep_list(mapping)
    dict3 = result[0].copy()
    for dict2 in result:
        dict3 = mergeDict(dict3, dict2)
        if(dict3 == -1):
            return -1
    return dict3

def clean_mappings(mappings):
    new_mappings = []
    for mapping in mappings:
        print(mapping)
        m = is_valid(mapping)
        print(m)
        if m != -1:
            new_mappings.append(m)
    return [new_mappings]

#print(is_valid(({0: 2, 1: 1, 2: 0}, {2: 0, 3: 6, 4: 5}, {4: 5, 5: 7})))

#Calculates all possible isomorphism by finding paths of desired length all over the graph at the instance
def creat_all_mappings_for_single_graph(gprim,g):
    testing_gprim = []
    testing_g = []
    list_mappis = []
    potential_paths = []

    paths_in_gprim = []


    #Trouver les sommets du tareget
    for i in range(len(gprim)):
        for j in range(len(gprim)) :
            if gprim[i][j] == 1 and i not in testing_gprim:
                testing_gprim.append(i)
    
    #Trouver le point de depart de BFS du pattern
    for i in range(len(g)):
        for j in range(len(g)) :
            if g[i][j] == 1 and i not in testing_g:
                testing_g.append(i)
                if len(neighbours(i,g))==1:
                    start = i

    g_to_paths = bfs(g,start)
    g_to_path_side_1 = max((x) for x in g_to_paths)
    '''
    for v in testing_gprim:
        c = bfs_k_length(len(testing_g),gprim,v)
        potential_paths = potential_paths+c'''
    for v in testing_gprim:
       for chemin in (bfs(gprim,v)):
            paths_in_gprim.append(chemin)
    
    print("POTENTIEL" + str(potential_paths))
    potential_paths = []
    for p in paths_in_gprim:
        #print(len(p))
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



def computeLPSArray(E, T, lps): 
    leni = 0 # length of the previous longest prefix suffix 
  
    lps[0] # lps[0] is always 0 
    i = 1
  
    # the loop calculates lps[i] for i = 1 to M-1 
    while i < T: 
        if graphsequal(E[i],E[leni]): 
        #if len(E[i])==len(E[leni]): 
            leni += 1
            lps[i] = leni
            i += 1
        else: 
            # This is tricky. Consider the example. 
            # AAACAAAA and i = 7. The idea is similar  
            # to search step. 
            if leni != 0: 
                leni = lps[leni-1] 
  
                # Also, note that we do not increment i here 
            else: 
                lps[i] = 0
                i += 1


def KMPSearch(E, Eprim): 
    M = len(E) 
    N = len(Eprim) 
    result = []
    all_mappings = []
    copies = dict()
    # create lps[] that will hold the longest prefix suffix  
    # values for pattern 
    lps = [0]*M 
    j = 0 # index for pat[] 
  
    # Preprocess the pattern (calculate lps[] array) 
    computeLPSArray(E, M, lps) 

    print("Tableau PI" + str(lps))
    print("all_mappings" + str(all_mappings))

    i = 0 # index for txt[] 
    while i < N: 
        print("voici i " + str(i))
        print("voici j " + str(j))
        mapping = creat_all_mappings_for_single_graph(Eprim[i],E[j])
      #  print("voici mapping" + str(mapping))
        if (i==0):
           # print("voici i " + str(i))
            all_mappings = [mapping]
            print("all_mappings" + str(all_mappings))
        if(i>0):
         #   print("voici i " + str(i))
         #   print("voici j " + str(j))
           # all_mappings_bef = copy.deepcopy(all_mappings)
            all_mappings.append(mapping)
            print("all_mappings" + str(all_mappings))
            print("MAPPING" + str(mapping))
            all_mappings = list(itertools.product(*all_mappings))
            all_mappings = clean_mappings(all_mappings)
            print("all_mappings 2" + str(all_mappings))
        if (all_mappings!=[[]]):
            print("il y a des isomorphismes restantes")
            if (lps[j]==0):
                print "Comme Pi[j] est 0 on va rajouter une copie de all_mappings de cet instant dans copies"
                copies[j] = copy.deepcopy(all_mappings)
                print("copies = " + str(copies))

            j = j + 1; 
            i = i + 1; 
           # print(j)
         
        if j == M: 
            result.append((((i-j)),all_mappings[0]))
           # print "Found pattern at index " + str(i-j) 
            j = lps[j-1]
           # print(j)
            i = i + 1

        # mismatch after j matches 
        elif i < N and all_mappings==[[]]: 
            # Do not match lps[0..lps[j-1]] characters, 
            # they will match anyway 
            if j != 0: 
                print("Mismatch 1 : il y a PAS d'isomorphisme")
                j = lps[j - 1]
                #all_mappings = copy.deepcopy(all_mappings_bef)
                if (j>0):
                    all_mappings = copy.deepcopy(copies.get(j))
                else: 
                    all_mappings = []
                    #all_mappings_bef = []
                #copies = dict()
            else: 
                print("Mismatch 2 : il y a PAS d'isomorphisme")
                i += 1
    
    return result

example_target_1 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target1.txt"),9)
example_pattern_1 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern1.txt"),6)

example_target_2 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target2.txt"),4)
example_pattern_2 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern2.txt"),4)

example_target_3 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target3.txt"),5)
example_pattern_3 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern3.txt"),5)
print(KMPSearch(example_pattern_3, example_target_3))

'''
somme = 1
n = 5
for i in range(n):
    generate_random_target_stream(50,50)
    generate_uniform_pattern(10,2,15)
    example_target_5 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/target_50inst_50vert.txt"),70)
    example_pattern_5 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/pattern_2inst_10vert.txt"),15)
    start = timeit.default_timer()
    print(KMPSearch(example_pattern_5,example_target_5))
    stop = timeit.default_timer()
    somme = somme + (stop - start)
    print('Time: ', stop - start)
print("temps moyenne " + str(somme/n))
'''

'''
def algo_back_track(G_i,G_prim_i):
    mappings = creat_all_mappings_for_single_graph(G_prim_i,G_i)
    if mappings != []:
        return mappings
   
all_mappings = []
mapping = algo_back_track([[0,1,0,0,0],[1,0,1,0,0],[0,1,0,0,0]],[[0,1,0,0,1],[1,0,1,1,0],[0,1,0,0,0],[0,1,0,0,0],[1,0,0,0,0]])
print(mapping)
all_mappings.append(mapping)
#all_mappings = list(itertools.product(*all_mappings))
print(all_mappings)
mapping = algo_back_track([[0,0,0,0,1],[0,0,0,0,1,0]],[[0,1,0,0,0],[0,0,1,1,0],[0,1,0,0,0],[0,1,0,0,0]])
print(mapping)
all_mappings.append(mapping)
all_mappings = list(itertools.product(*all_mappings))
print(all_mappings)
print(clean_mappings(all_mappings))

mapping = algo_back_track([[0,1,0],[1,0,1],[0,1,0]],[[0,1,0],[1,0,1],[0,1,0]])
all_mappings = [all_mappings]
all_mappings.append(mapping)
all_mappings = list(itertools.product(*all_mappings))
print(all_mappings)
print(clean_mappings(all_mappings))

'''