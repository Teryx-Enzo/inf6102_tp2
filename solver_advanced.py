from utils import *
import random
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import matplotlib.pyplot as plt

from copy import deepcopy
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
        #print(self.inventoryLevel,self.dailyConsumption,demande, self.L)
        if (demande > 0 and self.inventoryLevel-self.dailyConsumption+1*demande<=self.U) :
            return 1*demande 
        
        elif demande > 0:
            return demande
        else :
            return 0





def solution_initiale(nos_pizzerias,N,T,M,Q):
     

    solution = np.zeros((T,M,N))
    pizzerias = nos_pizzerias.copy()


    for t in range(T):


        for pizzeria in pizzerias:
            demande_jour = pizzeria.calculer_demande()
            pizzeria.demande_journaliere.append(demande_jour)
        
        
         
        for truck_id in range(M):
            #print(truck_id)
            truck_capacity = Q
            
            #On rempli chaque pizzeria pour petre sûr qu'il y aura assez de charbon pour la journée
            for pizzeria in pizzerias:
                    if np.count_nonzero(solution[t,:,pizzeria.id-1]) == 0:

                        demande = pizzeria.demande_journaliere[t]

                        if demande:
                            livraison = demande if  truck_capacity > demande else 0

                            if livraison :
                                solution[t,truck_id,pizzeria.id-1] = 1
                                truck_capacity -= livraison
                                pizzeria.inventoryLevel += livraison

            #S'il reste de la place dans le camion, on rempli un peu plus une ou plusieurs pizzeria pour éviter d'avoir à les livrer à nouveau le lendemain
            if truck_capacity > 0:
                
                for pizzeria in pizzerias:
                    
                    if solution[t,truck_id,pizzeria.id-1] == 1 :
                        

                        ajout = min([truck_capacity, pizzeria.U-pizzeria.i])

                        
                        pizzeria.inventoryLevel += ajout
                        pizzeria.demande_journaliere[t] += ajout
                        truck_capacity -= ajout
                        


        for pizzeria in pizzerias:
             pizzeria.inventoryLevel -= pizzeria.dailyConsumption



    return solution, pizzerias


def generate_raw_solution(solution,pizzerias,M,N,T,mineX,mineY,ordre_pizze):

    raw_solution = []
    

    for t in range(T):
        timestep =[]
        for truck_id in range(M):
            truck_route = []
            truck_route_coordinate = []

            for id_pizz in ordre_pizze:

                if solution[t,truck_id,id_pizz]==1:
                    truck_route.append((pizzerias[id_pizz].id,pizzerias[id_pizz].demande_journaliere[t]))
                    truck_route_coordinate.append((id_pizz,pizzerias[id_pizz].x,pizzerias[id_pizz].y))


            
            timestep.append(truck_route)
        raw_solution.append(timestep)

    return raw_solution


def deux_swap(pizzerias):
    """
    Génère une liste de voisins de la solution artpieces en échangeant pour chaque voisin 2 tableaux distincts.

    Args:
        artpieces (List): Liste d'items de dictionnaire (index, ArtPiece), solution actuelle.
    
    Returns:
        deux_swap_neigh (List[List]) : Liste de voisins de artpieces.
    """
    n = len(pizzerias)
    deux_swap_neigh = []
    for i in range(n):
        for j in range(i+1,n):
            neigh = deepcopy(pizzerias)
            neigh[i],neigh[j] = pizzerias[j], pizzerias[i]
            deux_swap_neigh.append(neigh)

    return deux_swap_neigh


def metric(pizzerias,instance,N,T,M,Q,mineX,mineY,nos_pizzerias):
    
    instance_copy = deepcopy(instance)
    voisin = []
    
    for pizzeria in pizzerias:
                pizzeria_copy = deepcopy(nos_pizzerias[pizzeria.id-1])
                pizzeria_copy.demande_journaliere = []
                voisin.append(pizzeria_copy)

    sol,pizz  = solution_initiale(voisin,N,T,M,Q)
    ordre_pizze = [pizzeria.id-1 for pizzeria in voisin]

    for pizzeria in pizz:
        nos_pizzerias[pizzeria.id-1].demande_journaliere = pizzeria.demande_journaliere
    sol_raw = generate_raw_solution(sol,nos_pizzerias,M,N,T, mineX,mineY,ordre_pizze)

    
    cost,validity = instance_copy.solution_cost_and_validity(Solution(instance_copy.npizzerias,sol_raw))


    return cost, sol_raw

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
    
    
    # Liste des pizzerias qui servent à la contruction des solutions (on doit conserver l'ordre initial)
    nos_pizzerias = [CustomPizzeria(p.id, p.x, p.y,p.inventoryLevel,  p.maxInventory, p.minInventory,  p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]
    
    # Copie des pizzerias et de l'instance pour pour la recherche de solution optimale
    pizzerias = deepcopy(nos_pizzerias)
    instance_copy = deepcopy(instance)

    # On créé une solution initiale avec l'ordre décrit dans le fichier d'instance
    sol,pizz  = solution_initiale(pizzerias,N,T,M,Q)
    ordre_pizze = [pizzeria.id-1 for pizzeria in pizzerias]
    for pizzeria in pizz:
        nos_pizzerias[pizzeria.id-1].demande_journaliere = pizzeria.demande_journaliere

    # On stocke la solution intiale et son score
    best_sol_raw = generate_raw_solution(sol,nos_pizzerias,M,N,T, mineX,mineY,ordre_pizze)
    best_cost,validity = instance_copy.solution_cost_and_validity(Solution(instance_copy.npizzerias,best_sol_raw))

    for _ in range(20):

        instance_temps = deepcopy(instance)
        pizzerias = deepcopy(nos_pizzerias)
        #restart aléatoire depuis la solution initiale
        random.shuffle(pizzerias)
        

        best_cost_restart, best_sol_raw_restart = metric(pizzerias,instance_temps,N,T,M,Q,mineX,mineY,nos_pizzerias)


        for _ in range(50):


            instance_temps = deepcopy(instance)

            # On génère la liste de tous les voisins obtenus à partir des deux swaps depuis la solution courante
            voisins = deux_swap(pizzerias)

            # On évalue tous les voisins
            metric_liste = [[voisin,*metric(voisin,instance_temps,N,T,M,Q,mineX,mineY,nos_pizzerias)] for voisin in voisins]

            # On trie selon le score obtenu
            voisins_sorted = sorted(metric_liste, key = lambda x:x[1])

            
            if voisins_sorted[0][1] < best_cost_restart:
                    print("Amelioration dans voisinnage",voisins_sorted[0][1])
                    best_cost_restart  = voisins_sorted[0][1]
                    best_sol_raw_restart = voisins_sorted[0][2]
                    
                    # On choisi comme solution courante le meilleur voisin
                    pizzerias = voisins_sorted[0][0]
                    
        if best_cost_restart < best_cost:
                    
                    print("Amelioration dans le restart",best_cost_restart)
                    best_cost  = best_cost_restart
                    best_sol_raw = best_sol_raw_restart
                       
             
    return Solution(instance.npizzerias,best_sol_raw)



