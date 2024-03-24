from utils import *
import random
import numpy as np
from copy import deepcopy
from time import time


class CustomPizzeria(Pizzeria):
    """ 
    Extension de la classe Pizzeria avec une liste de demandes journalières.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demande_journaliere = [] # TODO : changer par livraison ???

    def calculer_demande(self):
        # On calcule la demande et on la renvoie si elle est positive
        demande = max(0, self.L + self.dailyConsumption - self.inventoryLevel)

        return demande
        
        # if demande > 0:
        #     return demande
        # else :
        #     return 0


def solution_initiale(Pizzerias: List[CustomPizzeria], N: int, T: int, M: int, Q: int) -> np.ndarray:
    """
    Calcule une solution initiale gloutonne.

    Args:
        Pizzerias (List[CustomPizzeria]) : les pizzerias de l'instance.
        N (int) : le nombre de pizzerias de l'instance.
        T (int) : le nombre de jours de l'instance.
        M (int) : le nombre de camions de l'instance.
        Q (int) : la capacité de chaque camion de l'instance.
    
    Returns:
        solution (np.ndarray) : l'assignation gloutonne jour par jour des camions aux pizzerias.
    """ 

    solution = np.zeros((T,M,N))
    pizzerias = Pizzerias.copy()

    for t in range(T):

        for pizzeria in pizzerias:
            demande_jour = pizzeria.calculer_demande()
            pizzeria.demande_journaliere.append(demande_jour)
         
        for truck_id in range(M):
            truck_capacity = Q
            
            # On remplit chaque pizzeria dans l'ordre pour être sûr qu'il y aura assez de charbon pour la journée.
            for pizzeria in pizzerias:
                if np.count_nonzero(solution[t, :, pizzeria.id-1]) == 0: # Si la pizzeria n'est pas encore desservie
                    demande = pizzeria.demande_journaliere[t]

                    if demande:
                        livraison = demande if truck_capacity > demande else 0

                        if livraison :
                            solution[t, truck_id, pizzeria.id-1] = 1
                            truck_capacity -= livraison
                            pizzeria.inventoryLevel += livraison

            # S'il reste de la place dans le camion, on remplit un peu plus une ou plusieurs pizzeria(s) pour
            # éviter d'avoir à les livrer à nouveau le lendemain.
            if truck_capacity > 0:
                for pizzeria in pizzerias:
                    if solution[t, truck_id, pizzeria.id-1] == 1:
                        
                        ajout = min([truck_capacity, pizzeria.U-pizzeria.i])

                        pizzeria.inventoryLevel += ajout
                        pizzeria.demande_journaliere[t] += ajout
                        truck_capacity -= ajout
        
        for pizzeria in pizzerias:
             pizzeria.inventoryLevel -= pizzeria.dailyConsumption

    return solution, pizzerias


def generate_raw_solution(solution, pizzerias, M, N, T, mineX, mineY, ordre_pizze):
    """
    e
    
    Args:
        D
    
    Returns:
        D
    """

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
            neigh[i], neigh[j] = pizzerias[j], pizzerias[i]
            deux_swap_neigh.append(neigh)

    return deux_swap_neigh


def metric(pizzerias,instance,N,T,M,Q,mineX,mineY,nos_pizzerias):
    """
    e
    
    Args:
        D
    
    Returns:
        D
    """
    
    instance_copy = deepcopy(instance)
    voisin = []
    
    for pizzeria in pizzerias:
        pizzeria_copy = deepcopy(nos_pizzerias[pizzeria.id-1])
        pizzeria_copy.demande_journaliere = []
        voisin.append(pizzeria_copy)

    sol, pizz  = solution_initiale(voisin,N,T,M,Q)
    ordre_pizze = [pizzeria.id-1 for pizzeria in voisin]

    for pizzeria in pizz:
        nos_pizzerias[pizzeria.id-1].demande_journaliere = pizzeria.demande_journaliere

    sol_raw = generate_raw_solution(sol,nos_pizzerias,M,N,T, mineX,mineY,ordre_pizze)

    
    cost, _ = instance_copy.solution_cost_and_validity(Solution(instance_copy.npizzerias,sol_raw))


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

    list_costs = []


    mineX, mineY = instance.mine.x, instance.mine.y

    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0

    if ('instanceE' in instance.filepath) or ('instanceD' in instance.filepath):
        credit_temps = 600
    else:
        credit_temps = 300
    
    # Liste des pizzerias qui servent à la contruction des solutions (on doit conserver l'ordre initial)
    nos_pizzerias = [CustomPizzeria(p.id, p.x, p.y,p.inventoryLevel,  p.maxInventory, p.minInventory,  p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]
    
    # Copie des pizzerias et de l'instance pour pour la recherche de solution optimale
    pizzerias = deepcopy(nos_pizzerias)
    instance_copy = deepcopy(instance)

    # On créé une solution initiale avec l'ordre décrit dans le fichier d'instance
    sol, pizz  = solution_initiale(pizzerias,N,T,M,Q)
    ordre_pizze = [pizzeria.id-1 for pizzeria in pizzerias]
    for pizzeria in pizz:
        nos_pizzerias[pizzeria.id-1].demande_journaliere = pizzeria.demande_journaliere

    # On stocke la solution intiale et son score
    best_sol_raw = generate_raw_solution(sol,nos_pizzerias,M,N,T, mineX,mineY,ordre_pizze)
    best_cost,validity = instance_copy.solution_cost_and_validity(Solution(instance_copy.npizzerias,best_sol_raw))

    while ((time()-t0) + iteration_duration) < credit_temps-5:

        t1 = time()

        instance_temps = deepcopy(instance)
        pizzerias = deepcopy(nos_pizzerias)
        #restart aléatoire depuis la solution initiale
        random.shuffle(pizzerias)
        

        current_cost, current_sol_raw = metric(pizzerias,instance_temps,N,T,M,Q,mineX,mineY,nos_pizzerias)
        best_cost_restart, best_sol_raw_restart = current_cost, current_sol_raw

        t = 1000

        for _ in range(100):

            instance_temps = deepcopy(instance)

            # On génère la liste de tous les voisins obtenus à partir des deux swaps depuis la solution courante
            voisins = deux_swap(pizzerias)

            # On évalue tous les voisins
            #metric_liste = [[voisin,*metric(voisin,instance_temps,N,T,M,Q,mineX,mineY,nos_pizzerias)] for voisin in voisins]

            # On trie selon le score obtenu
            #voisins_sorted = sorted(metric_liste, key = lambda x:x[1])

            voisin = random.choice(voisins)

            cost, sol_row = metric(voisin, instance_temps, N, T, M, Q, mineX, mineY, nos_pizzerias)

            

            delta = cost - current_cost

            # if delta <= 0:
            #     print(delta)
            # else:
            #     print(delta, np.exp(-delta/t))

            if delta <= 0 : #or np.random.rand() < np.exp(-delta/t):
                current_cost, current_sol_raw = cost, sol_row
            
            list_costs.append(current_cost)

            if current_cost < best_cost_restart:
                #print("Amelioration dans voisinnage", current_cost)
                best_cost_restart    = current_cost
                best_sol_raw_restart = current_sol_raw
                
                # On choisi comme solution courante le meilleur voisin
                pizzerias = voisin
            
            t *= 0.99
                    
        if best_cost_restart < best_cost:
                    
                    print("Amelioration dans le restart",best_cost_restart)
                    best_cost  = best_cost_restart
                    best_sol_raw = best_sol_raw_restart
        iteration_duration = time() - t1
    
    # plt.figure()
    # plt.plot(list_costs)
    # plt.show()
             
    return Solution(instance.npizzerias,best_sol_raw)
