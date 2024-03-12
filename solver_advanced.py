from utils import *
import random
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt


class CustomPizzeria(Pizzeria):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demande_journaliere = []


    def calculer_demande(self):

        #On calcule la demande et on la renvoie si elle est positive
        demande = self.L+self.dailyConsumption-self.inventoryLevel
        
        return demande if demande > 0 else 0



def distance(point1,point2):
    _, x1, y1 = point1
    _, x2, y2 = point2

    distance = np.sqrt((x1-x2)**2+(y1-y2)**2)


    return distance


def graph_complet(points):


    G = nx.Graph()

    for x in range(len(points)-1):

        G.add_node(x)
        G.add_edge(x,x+1,weight = distance(points[x],points[x+1]))

    return G

def TSP(points,mineX,mineY):
    points = [(-1,mineX,mineY)]+points
    G = graph_complet(points)

    
    min = nx.minimum_spanning_tree(G) 


    impair_min = []

    for node in min.nodes: 
        if (min.degree(node) % 2 != 0):  

            impair_min.append(node * -1)

    impair = nx.complete_graph(impair_min)
    couplage = nx.max_weight_matching(impair, maxcardinality=True)

    liste_aretes = []

    for i in couplage:
        liste_aretes.append(i[0]*-1)
        liste_aretes.append(i[1]*-1)


    liste_ordre = []
    IterZip = zip(*[iter(liste_aretes)] * 2)
    for i in IterZip:
        liste_ordre.append(i)
    min.add_edges_from(liste_ordre)
    

    resultat = []
    for i in nx.eulerian_circuit(min):
        if i[0] not in resultat:
            resultat.append(i[0])
    
    return resultat[1:]

def solution_initiale(nos_pizzerias,N,T,M,Q):
     

    solution = np.zeros((T,M,N))
    pizzerias = nos_pizzerias.copy()


    for t in range(T):


        for pizzeria in pizzerias:
            demande_jour = pizzeria.calculer_demande()
            pizzeria.demande_journaliere.append(demande_jour)
        
        
         
        for truck_id in range(M):
            truck_capacity = Q
            
            for pizzeria in pizzerias:
                    if np.count_nonzero(solution[t,:,pizzeria.id-1]) == 0:

                        demande = pizzeria.demande_journaliere[t]

                        if demande:
                            livraison = demande if  truck_capacity > demande else 0

                            if livraison :
                                solution[t,truck_id,pizzeria.id-1] = 1
                                truck_capacity -= livraison
                                pizzeria.inventoryLevel += livraison
                        

                        


        for pizzeria in pizzerias:
             pizzeria.inventoryLevel -= pizzeria.dailyConsumption

    return solution


def generate_raw_solution(solution,pizzerias,M,N,T,mineX,mineY):

    raw_solution = []

    for t in range(T):
        timestep =[]
        for truck_id in range(M):
            truck_route = []
            truck_route_coordinate = []

            for id_pizz in range(N):

                if solution[t,truck_id,id_pizz]==1:
                    truck_route.append((pizzerias[id_pizz].id,pizzerias[id_pizz].demande_journaliere[t]))
                    truck_route_coordinate.append((id_pizz,pizzerias[id_pizz].x,pizzerias[id_pizz].y))

            if len(truck_route)>2:
                print(truck_route)
                print(truck_route_coordinate)
                ordered_indices = TSP(truck_route_coordinate,mineX,mineY)
                
                truck_route = [truck_route[i-1] for i in ordered_indices]
                print(truck_route)

            
            timestep.append(truck_route)
        raw_solution.append(timestep)

    return raw_solution


def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    Q, M, N, T = instance.Q, instance.M, instance.npizzerias, instance.T


    mineX, mineY = instance.mine.x, instance.mine.y
    #nos_pizzerias = {id: CustomPizzeria(p.id, p.x, p.y, p.maxInventory, p.minInventory, p.inventoryLevel, p.dailyConsumption, p.inventoryCost) for id, p in instance.pizzeria_dict.items()}
    
    

    nos_pizzerias = [CustomPizzeria(p.id, p.x, p.y,  p.maxInventory, p.minInventory, p.inventoryLevel, p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]
    
    pizzerias = nos_pizzerias.copy()
    random.shuffle(nos_pizzerias)

    
    sol_raw = generate_raw_solution(solution_initiale(nos_pizzerias,N,T,M,Q),pizzerias,M,N,T, mineX,mineY)

    


    
             
    return Solution(instance.npizzerias,sol_raw)


    
if __name__ == "__main__":

    points = [(2,0,0),(3,2,3),(4,3,2),(5,0,1),(6,5,12) ]

    puntos = []
    for i,x,y in points:
        puntos.append((i,float(x), float(y)))


    print(TSP(puntos))
