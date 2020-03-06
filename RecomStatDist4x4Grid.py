#Camryn Hollarsmith
#TransitionMatrix4x4.py
#Thesis Advisor: Sarah Cannon
#Thesis Fall 2019 - Spring 2020

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import numpy.linalg as npla 
import pickle 

fileObject = open('ListOf4x4Plans', 'rb') 
plans = pickle.load(fileObject)

np.set_printoptions(precision=10)

g = nx.grid_graph([4,4])

def countDistInCommon(p, q):
    """returns the number of districts in common
    input is two districting plans (dictionaries)
    output is the number of districts in common +
    what district it is
    """
    
    #want 4 vertices in common because that means that a district is in common
    numDistInCommon = 0 #like with for loops want to add to this number, set them both to 0 respectively 
    distlista = []
    distlistb = []
    #these for loops are summarizing all the different variations of count00, count01, etc. inthe 3x3 case
    for a in range(4):
        for b in range(4):
            countab = len([v for v in g.nodes() if (p[v] == a and q[v] == b)])
            if countab == 4:
                numDistInCommon += 1
                distlista.append(a) #adding a to commonlist
                distlistb.append(b)
    return [numDistInCommon, [distlista, distlistb]] #returning both the number of dists in common as well as which district it is.

def IsItSquare(plan, district):
    """should return True if district is a square in plan. 
    """

    #getting the vertices that are in the district    
    nodesindist = [v for v in plan if plan[v] == district]

    #taking a subgraph of the four vertices that were found by nodesindist
    s = g.subgraph(nodesindist)

    #count edges of s to get 4 for square and 3 for not
    if s.number_of_edges() == 4:
        return True
    else:
        return False 

def endpoints(a,b):
    """want to find if endpoints are NOT in the same district
    """
    numedgesbetween = 0
    for e in union.edges():
        if plans[j][e[0]] != plans[j][e[1]]:
            numedgesbetween += 1
    return numedgesbetween 

def numedgesbetween(color, listofedges):
    n = 0
    for e in g.edges():
        if color[e[0]] not in listofedges and color[e[1]] not in listofedges and color[e[0]] != color[e[1]]:
            n+=1
    return n 
            

#VERSION 1
s = (117, 117) 
#we set it to zero primarily because there are more empty values than filled and then throughout the for loops below i fill in the values for which the statements are true
A = np.zeros(s, dtype=float )
#set the word "plans" equal to the list of all 117 plans that were defined by ray
#this works such that when we call plan[i] or plan[j] it will cycle through all the plans 0-116
#want to cycle i and j through all 117 possibilities for the plans
for i in range(117):
    sum = 0
    for j in range(117): 
        #output from countDistInCommon of two plans from the list above, n is the number of districts in common and common list is which district the plans have in common
        [n, commonlist] = countDistInCommon(plans[i], plans[j])

        #if the number of districts in common is 3 then the plans are the same so there is no probability of going from one to the other
        # the zero will be addressed later as the diagonal entry but for now, this is correct.
        # technically this line does not have to be indicated because if we did not include it the diagonals would still be zero because the base matrix is all 0's
        if n == 4: # or n == 3: arbitrary to say 
            A[i,j] = 0 
        #this is what we are most interested in!!!:
        #if the two plans share one or two district in common, then what's the probability of going from one districting plan to the other?
        if n == 2:
            #want to know the list of vertices in plani where they are not in the first district in common between the two plans
            #this is what the step is of going from one plan to another. 
            #know there are two dist in common, then look at the other vertices of districts not in common
            LV1 = [v for v in g.nodes() if plans[i][v] not in commonlist[0]]
           
            # print ("Vertices of districts not in common between two plans:") 
            # print (LV1)
            
            #want to know which plans we are indicating between and what values we get from each plan
            
            # print ("i, j:")
            # print (i, j)
            
            #the commonlist is the vertices that are shared between the two plans
            
            # print ("This is the commonlist:")
            # print(commonlist)
            
            #taking the union of all of the vertices from LV1 because this is what the probability is! 
            #disrregarding the one district in common, this is taking the spanning tree of the rest of the vertices.
            union = g.subgraph(LV1) 
            # print ("union") 
            # print (union)
            #using this to find if endpoiints are not in the same district. if they are NOT in the same district then you add 1 for each vertex
            #the idea behind this is to count the number of choices of a tree and an edges so that the union happens - in 3x3 case is usually 3

            #counting the number of edges between the two districts that are not in the commonlist for plans[j] 
            neb = numedgesbetween(plans[j], commonlist[1])
            wayssameplan = neb 
            for k in range(4):
                if k not in commonlist:
                    if IsItSquare(plans[j], k) == True:
                        wayssameplan *= 4
            
            #test if isitsquare is true for both districts in the union: 
            #if yes, then prob is 4*4*2

            


            #taking the laplacian of the union
            # the diagonals of the laplacian give you the degrees of the vertices
            #so if the diagonal entry is 2 then that vertex has degree 2 meaning it is connected to 2 other vertices
            #we take the laplacian of the union because we want to know the degrees of the two other districts not in common (which is the union)
            Lapl = nx.laplacian_matrix(union)
            #turning it into a matrix that is viewable
            LaplArray = np.matrix(Lapl.toarray())
            
            #print ("This is the Laplacian array:")
            #print (LaplArray )
            #print ()
            
            #taking the minor of the laplacian. 
            #doing this allows us to then take the determinant which gives us the correct number of spanning trees 
            #taking the minor means we delete one row and one column. I decide to delete the last row and column because coding it is simple.
            #it is irrelevant which row/column we delete. 
            mLapl = LaplArray[:-1,:-1] #deleting last row and last column
            
            # print ("This is the mLaplacian array:")
            # print (mLapl) 
            # print ()
            
            #want to round the determinant of the minor of laplacian because we don't need the extra values
            det = round( npla.det(mLapl)) #determinant of minor of Laplacian
            
            # print("This is the determinant (number of spanning trees):")
            # print (det)
            # print () 
            
            #now multiplying all of the elements of our probability to find the probability
            #we have 1/det which is 1/the number of spanning trees
            #we have 1/6 = 1/(4 choose 2) = 0.1666666667 which is the probability of choosing one district out of the four
            #we have 1/7 = 0.1428571429 which is the number of edges 
            #finally we have wayssameplan which is the number of choices of a tree and an edges
            prob = ((1/det) * 0.1666666667 * 0.1428571429 * wayssameplan) 
           
            # print("This is the probability:")
            # print (round(prob, 4)) 
            # print()
            
            #fill in each value in the matrix with a correct value! 
            A[i,j] = prob 
            #adding up every value in the row so that we can find the diagonal entry
            sum = sum + prob #A[i,j]

    #fill in the diagonals with -sum because want rows to sum to zero.        
    A[i,i] = -sum 
    #print ("plans[",i, "]")
    #print (plans[i])
    #print ("plans[", j, "]")
    #print (plans[j])
    #print ("A[", i, "][", j, "]")
    #print (A[i,j])
   

# print ('VERSION 1 117x177 matrix with zeros where there is probability zero going from one plan to another and the correct value where there is a probability of going from one plan to another:')
print (A)
# print ()



#version 1 4x4, diagonal entries s.t. row sums to 0
A = np.array(A)
for i in range(117):
    A[i][116] = 1
# print ('A matrix from version 1 4x4 s.t. row sums to 0:')
# print (A) 
#to find stationary distribution:
X = np.zeros((1,117))
X[0,116] = 1 
X1 = X.T 
Y = np.linalg.lstsq(A.T, X1)
PI = Y[0].T 
print ('solution to version 1, pi:')
print (PI) 



#Drawing grid graphs 
#for i in range(117):
#   print ("plan", i)
#    plt.figure() 
#    nx.draw(g,pos = {x:x for x in g.nodes()}, node_color = [plans[i][x] for x in g.nodes()])
#    plt.show()
    

#plt.figure() # start a figure
#nx.draw(g,pos = {x:x for x in g.nodes()}, node_color = [plans[1][x] for x in g.nodes()])
#plt.show()

# plt.figure() # start a figure
# nx.draw(g,pos = {x:x for x in g.nodes()}, node_color = [plans[76][x] for x in g.nodes()])
# plt.show()

# plt.figure() # start a figure
# nx.draw(g,pos = {x:x for x in g.nodes()}, node_color = [plan2[x] for x in g.nodes()])
# plt.show()

# plt.figure() # start a figure
# nx.draw(g,pos = {x:x for x in g.nodes()}, node_color = [plan3[x] for x in g.nodes()])
# plt.show()


# np.set_printoptions(precision=10)