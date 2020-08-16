import itertools
from queue import Queue
import collections
import sys, getopt
import timeit
import numpy as np
import random
import gc
import os
import os.path
#import matplotlib.pyplot as plt
import math
import timeout_decorator
import resource



def make_random_path(nbr_vertices,length):
    l = random.sample(range(0,nbr_vertices), length)
    print(l)
    return l
    '''
    for i in range(len(l)):
        if i!=len(l)-1:
            print(str(l[i]) + " " + str(l[i+1]))'''

#make_random_path(4,4)

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

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

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

def clean_mappings(mappings):
    new_mappings = []
    for mapping in mappings:
        #print(mapping)
        m = is_valid(mapping)
        #print(m)
        if m != -1:
            new_mappings.append(m)
    return [new_mappings]

# returns cartesian product of mappings at each instant
def creat_all_mappings(E,Eprim,t,testing_gprim):
    all_mappings = []
    c = []

    all_mappings.append(creat_all_mappings_for_single_graph(Eprim[t],E[0],testing_gprim[t]))
    
    for i in range (1,len(E)):
        #print(str(i)+"/"+str(len(E)))
        m = creat_all_mappings_for_single_graph(Eprim[t+i],E[i],testing_gprim[t+i])
        all_mappings.append(m)
        all_mappings = list(itertools.product(*all_mappings))
        #print("avant clean")
        #print(all_mappings)
        all_mappings = clean_mappings(all_mappings)
        #print("apres clean")

        if (all_mappings == [[]]):
            return -1

    return all_mappings
        #list_mappings.append(creat_all_mappings_for_single_graph(Eprim[t+i],E[i],testing_gprim[t+i]))
    
    #print("me voilaaaa" + str(list_mappings))
    #print("me voilaaaaaaaaa" + str(list(itertools.product(*list_mappings))))
    '''
    for i in range (0,len(E)):
        #print(str(i)+"/"+str(len(E)))
        c = cartesian(creat_all_mappings_for_single_graph(Eprim[t+i],E[i],testing_gprim[t+i]),c)
        #list_mappings.append(creat_all_mappings_for_single_graph(Eprim[t+i],E[i],testing_gprim[t+i]))'''
    
    #return  list(itertools.product(*list_mappings))

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

#Calculates all possible isomorphism by finding paths of desired length all over the graph at the instance
def creat_all_mappings_for_single_graph(gprim,g,testing_gprim):
 #   testing_gprim = []
    list_mappis = []
    potential_paths = []
    
    #print(testing_gprim)

    for v in testing_gprim:
        c = bfs_k_length(len(g),gprim,v)
        potential_paths = potential_paths+c
    
    #print("POTENTIEL" + str(potential_paths))

    for p in potential_paths:
        mapi = dict()
        for i in range(len(p)):
            mapi[g[i]] = p[i]
        list_mappis.append(mapi)
    
    return list_mappis

#print(creat_all_mappings_for_single_graph([[0,1,1,0,0],[1,0,0,1,1],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0]],[[0,1,0],[1,0,1],[0,1,0]]))
#print(creat_all_mappings(example_pattern,example_target,0))

#Final algorithm takes pattern and target and returns possible isomorphism if available

def naive_algo(E,Eprim,testing_grpim):
    result = []
    for t in range (0,len(Eprim)):
        print(t)
        #print(len(E))
        #print(len(Eprim))
        if (t + len(E)) > len(Eprim):
            print("No possible isomorphism heeeree")
            return -1
        mappings = creat_all_mappings(E,Eprim,t,testing_grpim)
        #print(mappings)
        if (mappings != 0):
           # print("Found isomorphism with mapping " + str(mappings) + " at instance " + str(t+1) + " of the target.")
            result.append((t+1,mappings))
            return result
        '''for m in mappings:
            merged = is_valid(m)
            if merged!=-1:
                print("Found isomorphism with mapping " + str(merged) + " at instance " + str(t+1) + " of the target.")
                result.append((t+1,merged))
                return result'''

    print("No possible isomorphism")
    return result

'''
def naive_algo(E,Eprim,testing_grpim):
    result = []
    for t in range (0,len(Eprim)):
        print(t)
        #print(len(E))
        #print(len(Eprim))
        if (t + len(E)) > len(Eprim):
            print("No possible isomorphism heeeree")
            return -1
        mappings = creat_all_mappings(E,Eprim,t,testing_grpim)
        #print(mappings)
        for m in mappings:
            merged = is_valid(m)
            if merged!=-1:
                print("Found isomorphism with mapping " + str(merged) + " at instance " + str(t+1) + " of the target.")
                result.append((t+1,merged))
                return result
        gc.collect()

    
    print("No possible isomorphism")
    return result
'''
'''
x1 = [5, 10, 15, 20, 30]
y1 = [7.82, 307.81283,  144.733283043, 18.3932094,  6.54 ]
y2 = [10.92, 693.34, 304.47,36.03, 9.23]
y3 = [6.027, 1000,1000,1000, 177.64]


plt.plot(x1, y1, label = "Target with 50 instances")
plt.plot(x1, y2, label = "Target with 100 instances")
plt.plot(x1, y3, label = "Target with 1000 instances")
plt.xlabel('Pattern length')
plt.ylabel('Execution time (seconds)')

plt.ylim((25,800))
plt.legend()


plt.show()
'''
'''
x1 = [5, 10, 15, 20, 30]
y1 = [4.48, 1507.1912,  2200.033, 3000, 3000 ]
y2 = [24.14, 3000, 3000,3000, 3000]
y3 = [63.49, 6000,6000,6000, 6000]


plt.plot(x1, y1, label = "Target with 50 instances")
plt.plot(x1, y2, label = "Target with 100 instances")
plt.plot(x1, y3, label = "Target with 1000 instances")
plt.xlabel('Pattern length')
plt.ylabel('Execution time (seconds)')

plt.xlim((5,10))
plt.ylim((25,2300))
plt.legend()


plt.show()

example_target_1 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target1.txt"),9)
example_pattern_1 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern1.txt"),6)

example_target_2 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target2.txt"),4)
example_pattern_2 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern2.txt"),4)

example_target_3 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_target3.txt"),5)
example_pattern_3 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern3.txt"),5)

example_target_4 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/data/target_1000.txt"),2000)
example_pattern_4 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/example_pattern2.txt"),8)
'''

def file_to_graphs_present_vertices(file):
    f = open(file, "r")
    a = -1
    edges = []
    graphs = []
    present_v = []
    edges2 = []
    maxi = 0
    i = 0
    for line in f : 
        data = line.split()
        max2 = max(list(map(int, data[1:])))
        if max2 > maxi:
            maxi = max2

        if data[0] == a:
            edges2.append(data[1:])
            if(int(data[1]) not in edges):
                edges.append(int(data[1]))
            if(int(data[2]) not in edges):
                edges.append(int(data[2]))
            #if (edges != []):
                #print(edges)
        else:
            edges = []
            edges = edges + list(map(int, data[1:])) 
            present_v.append(edges)

            edges2 = []
            edges2.append(data[1:])
            graphs.append(edges2)
        a = data[0]
    return (graphs,present_v,maxi+1)

def file_to_paths(file):
    f = open(file, "r")
    a = -1
    edges = []
    graphs = []
    i = 0
    for line in f :
        data = line.split()
        if data[0] == a:
            edges.append(int(data[2]))
        else:
            test_list = list(map(int, data[1:])) 
            edges = test_list
            graphs.append(edges)
        a = data[0]
    #print(graphs)
    return graphs

@timeout_decorator.timeout(400,timeout_exception=OSError) 
def test_defined_pattern_defiend_target(tpath,ppath,target, pattern, number_of_try):
    somme = 0
    for i in range(number_of_try):
        (tomatrix, testing_grpim,maxi) = file_to_graphs_present_vertices(tpath+"/"+target)
        target = to_list_of_matrices(tomatrix,maxi)
        pattern = file_to_paths(ppath +"/"+ pattern)
        start2 = timeit.default_timer()
        r = naive_algo(pattern,target,testing_grpim)
        stop2 = timeit.default_timer()
        somme = somme + (stop2 - start2)
        #print('Time: ', stop - start)

    print("temps moyenne " + str(somme/number_of_try))
    return (somme/number_of_try,r)



def run_tests(d1, d2,results,n_o_t,fold,all_fold):
    patterns = []
    targets = []
    for filename in os.listdir(d1):
        patterns.append(filename)
    for filename in os.listdir(d2):
        targets.append(filename)
    
    fi = open(results,"w")
    alltests = len(targets)*len(patterns)
    i = 0

    for target in targets:
        for pattern in patterns:
            try:
                i = i + 1
                #print("test " + str(i) + "/" + str(alltests) + ":" + target + "    " + pattern)
                #print("folder" + str(fold) + "/" + str(all_fold))
                print("test {} / {} : {}    {}".format(i,alltests,target,pattern))
                print ("folder {} / {}".format(fold,all_fold))
                (t,r) = test_defined_pattern_defiend_target(d2,d1,target,pattern,n_o_t)
                print(r)
            except Exception as e:
                #print("ALERTE OUT OF TIME")
                print(e)
                fi.write("target {} {} OUT OF TIME \n".format(target,pattern))
                #fi.write(target + " " + pattern + " OUT OF TIME " + "\n")            
            else:
                if (r == []):
                    res = "no"
                    fi.write("target {} {} {} {} \n".format(target,pattern,t,res))
                    #fi.write(target + " " + pattern + " " + str(t) + " " + res + " " + "\n")
                else:
                    res = "yes"
                    (instant,mapi) = r[0]
                    fi.write("target {} {} {} {} {} \n".format(target,pattern,t,res,instant))
                    #fi.write(target + " " + pattern + " " + str(t) + " " + res + " " + str(instant) + "\n")
        gc.collect()
        print "\nGARBAGE OBJECTS:"
        for x in gc.garbage:
            s = str(x)
            if len(s) > 80: s = s[:77]+'...'
            print type(x),"\n  ", s

            
                


#example_target_5 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/target_1000inst_15vert.txt"),25)
#example_pattern_5 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/pattern_2inst_5vert.txt"),10)

#print(naive_algo(example_pattern_1,example_target_1))

def run_tests_on_folders(paths_folders,targtes_folders):

    subfolders_paths = [os.path.join(paths_folders, o) for o in os.listdir(paths_folders) 
                    if os.path.isdir(os.path.join(paths_folders,o))]

    subfolders_targets = [os.path.join(targtes_folders, o) for o in os.listdir(targtes_folders) 
                    if os.path.isdir(os.path.join(targtes_folders,o))]

    #subfolders_paths = [ f.path for f in os.scandir(paths_folders) if f.is_dir() ]
    #subfolders_targets = [ f.path for f in os.scandir(targtes_folders) if f.is_dir() ]
    i = 0
    for targtes_folder in subfolders_targets:
        i = i + 1
        r = "/home/fatemeh/Bureau/Stage/resultsNaif/T"+ targtes_folder[35:]
        for path_folder in subfolders_paths:
            run_tests(path_folder,targtes_folder,r,1,i,len(subfolders_targets))
    
#run_tests_on_folders("/home/fatemeh/Bureau/Stage/patterns/","/home/fatemeh/Bureau/Stage/targets/")


def main():
    gc.enable()
    gc.set_debug(gc.DEBUG_LEAK)

    run_tests_on_folders("/home/fatemeh/Bureau/Stage/patterns/","/home/fatemeh/Bureau/Stage/targets/")
    '''(tomatrix, testing_grpim,maxi) = file_to_graphs_present_vertices("/home/fatemeh/Bureau/Stage/targets/T100/target_100inst_15vert")
    print(tomatrix)
    print(testing_grpim)
    print("MAXI" + str(maxi))
    example_target_1 = to_list_of_matrices(tomatrix,maxi)

    example_pattern_1 = file_to_paths("/home/fatemeh/Bureau/Stage/patterns/example_pattern1")

    print(naive_algo(example_pattern_1, example_target_1,testing_grpim))'''



#test_defined_pattern_random_target(100, 15, "example_pattern2.txt" ,1)

#def test_random_pattern_random_target(T_G, V_G, T_P, V_P,number_of_try):

if __name__ == "__main__":
    #try:
    main()
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print ("Memory usage is: {0} KB".format(mem))


    #except IOError as (errno, strerror):
      #  print "I/O error({0}): {1}".format(errno, strerror)
    #except MemoryError as error:
        # Output expected MemoryErrors.
        #print(error)
    #except ValueError:
     #   print ("Could not convert data to an integer.")
    #except:
      #  print ("Unexpected error:", sys.exc_info()[0])
    


'''
somme = 1
n = 5
for i in range(n):
    generate_random_target_stream(100,100)
    generate_uniform_pattern(10,2,15)
    example_target_5 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/target_100inst_100vert.txt"),120)
    example_pattern_5 = to_list_of_matrices(file_to_graphs("/home/fatemeh/Bureau/Stage/pattern_2inst_10vert.txt"),15)
    start = timeit.default_timer()
    print(naive_algo(example_pattern_5,example_target_5))
    stop = timeit.default_timer()
    somme = somme + (stop - start)
    print('Time: ', stop - start)
print("temps moyenne " + str(somme/n))
'''
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