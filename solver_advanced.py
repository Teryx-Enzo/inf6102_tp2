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
        self.daily_deliveries = []

    def compute_demand(self):
        # On calcule la demande et on la renvoie si elle est positive
        demand = max(0, self.L + self.dailyConsumption - self.inventoryLevel)

        return demand


def greedy_assignment(Pizzerias: List[CustomPizzeria], N: int, T: int, M: int, Q: int) -> Tuple[np.ndarray, List[CustomPizzeria]]:
    """
    Calcule une assignation gloutonne des camions aux pizzerias. Met à jour le stock de chaque pizzeria selon les quantités livrées et consommées.

    Args:
        Pizzerias (List[CustomPizzeria]) : les pizzerias (étendues) de l'instance, dans un ordre quelconque.
        N (int) : le nombre de pizzerias de l'instance.
        T (int) : le nombre de jours de l'instance.
        M (int) : le nombre de camions de l'instance.
        Q (int) : la capacité de chaque camion de l'instance.
    
    Returns:
        assignment (np.ndarray) : l'assignation (binaire) gloutonne jour par jour des camions aux pizzerias.
        Updated_pizzerias (List[CustomPizzeria]) : les pizzerias de l'instance avec leur stock mis à jour selon les livraisons assignées.
    """
    assignment = np.zeros((T,M,N))
    Updated_pizzerias = Pizzerias.copy()

    for t in range(T):

        for pizzeria in Updated_pizzerias:
            daily_demand = pizzeria.compute_demand()
            pizzeria.daily_deliveries.append(daily_demand)
        
        for truck_id in range(M):
            truck_capacity = Q
            
            # On remplit chaque pizzeria dans l'ordre pour être sûr qu'il y aura assez de charbon pour la journée.
            for pizzeria in Updated_pizzerias:
                if np.count_nonzero(assignment[t, :, pizzeria.id-1]) == 0: # Si la pizzeria n'est pas encore desservie
                    demand = pizzeria.daily_deliveries[t]

                    if demand:
                        livraison = demand if truck_capacity > demand else 0

                        if livraison :
                            assignment[t, truck_id, pizzeria.id-1] = 1
                            truck_capacity -= livraison
                            pizzeria.inventoryLevel += livraison

            # S'il reste de la place dans le camion, on remplit un peu plus une ou plusieurs pizzeria(s) pour
            # éviter d'avoir à les livrer à nouveau le lendemain.
            if truck_capacity > 0:
                for pizzeria in Updated_pizzerias:
                    if assignment[t, truck_id, pizzeria.id-1] == 1:
                        
                        ajout = min([truck_capacity, pizzeria.U-pizzeria.i])

                        pizzeria.inventoryLevel += ajout
                        pizzeria.daily_deliveries[t] += ajout
                        truck_capacity -= ajout
        
        for pizzeria in Updated_pizzerias:
             pizzeria.inventoryLevel -= pizzeria.dailyConsumption

    return assignment, Updated_pizzerias


def generate_raw_solution(assignment, Instance_pizzerias, M, T, Order) -> List[List[List[Tuple[int,float]]]]:
    """
    Génère une solution brute étant données les assignations de camions et les quantités à livrer, dans l'ordre considéré.
    
    Args:
        assignment (np.ndarray) : l'assignation (binaire) gloutonne jour par jour des camions aux pizzerias.
        Instance_pizzerias (List[CustomPizzeria]) : les pizzerias (étendues) de l'instance, dans leur ordre initial.
        M (int) : le nombre de camions de l'instance.
        T (int) : le nombre de jours de l'instance.
        Order (List[int]) : l'ordre de livraison des pizzerias.
    
    Returns:
        raw_solution (raw_solution:List[List[List[Tuple[int,float]]]]) : liste de timesteps qui sont des listes de routes pour les camions utilisés. 
            Une route est une liste de couples (pizzeria, qté).
    """
    raw_solution = []

    for t in range(T):
        timestep =[]

        for truck_id in range(M):
            truck_route = []

            for id_pizz in Order:
                if assignment[t, truck_id, id_pizz] == 1:
                    truck_route.append((Instance_pizzerias[id_pizz].id, Instance_pizzerias[id_pizz].daily_deliveries[t]))
            
            timestep.append(truck_route)
        raw_solution.append(timestep)

    return raw_solution


def two_swap(Pizzerias: List[CustomPizzeria]) -> List[List[CustomPizzeria]]:
    """
    Génère une liste de voisins de la représentation ordonnée des pizzerias en échangeant pour chaque voisin 2 pizzerias distinctes.

    Args:
        Pizzerias (List[CustomPizzeria]) : les pizzerias (étendues) de l'instance.
    
    Returns:
        two_swap_neigh (List[List[CustomPizzeria]]) : liste de représentation voisines de Pizzerias.
    """
    n = len(Pizzerias)
    two_swap_neigh = []

    for i in range(n):
        for j in range(i+1, n):
            neigh = deepcopy(Pizzerias)
            neigh[i], neigh[j] = Pizzerias[j], Pizzerias[i]
            two_swap_neigh.append(neigh)

    return two_swap_neigh


def generate_and_evaluate(pizzerias, instance, N, T, M, Q, Instance_pizzerias) -> Tuple[float, List[List[List[Tuple[int,float]]]]]:
    """
    Calcule une solution gloutonne pour l'ordre dans lequel on présente les pizzerias, et génère la solution brute correspondante et son coût.
    
    Args:
        pizzerias (List[CustomPizzeria]) : les pizzerias étendues de l'instance, dans l'ordre dans lequel on les considère pour l'assignation gloutonne.
        instance (Instance) : l'instance du problème.
        N (int) : le nombre de pizzerias de l'instance.
        T (int) : le nombre de jours de l'instance.
        M (int) : le nombre de camions de l'instance.
        Q (int) : la capacité de chaque camion de l'instance.
        Instance_pizzerias (List[CustomPizzeria]) : les pizzerias étendues de l'instance, dans leur ordre initial.
    
    Returns:
        cost (float) : le coût total de la solution.
        raw_solution (raw_solution:List[List[List[Tuple[int,float]]]]) : liste de timesteps qui sont des listes de routes pour les camions utilisés. 
            Une route est une liste de couples (pizzeria, qté).
    """
    # On crée une nouvelle représentation des pizzerias dans l'ordre présenté dans pizzerias, et on fait l'assignation gloutonne
    neigh = []
    for pizzeria in pizzerias:
        pizzeria_copy = deepcopy(Instance_pizzerias[pizzeria.id-1])
        pizzeria_copy.daily_deliveries = []
        neigh.append(pizzeria_copy)

    assignment, Updated_pizzerias = greedy_assignment(neigh, N, T, M, Q)
    Order = [pizzeria.id-1 for pizzeria in neigh]

    # On génère la solution brute correspondante et on l'évalue
    for pizzeria in Updated_pizzerias:
        Instance_pizzerias[pizzeria.id-1].daily_deliveries = pizzeria.daily_deliveries

    raw_solution = generate_raw_solution(assignment, Instance_pizzerias, M, T, Order)
    cost, _ = instance.solution_cost_and_validity(Solution(instance.npizzerias, raw_solution))

    return cost, raw_solution


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

    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0

    if ('instanceA' in instance.filepath) or ('instanceB' in instance.filepath) or ('instanceC' in instance.filepath):
        time_credit = 300
    else: #instances D, E ou X
        time_credit = 600
    
    # Liste des pizzerias qui sert à la contruction des solutions (on doit conserver l'ordre initial)
    Instance_pizzerias = [CustomPizzeria(p.id, p.x, p.y,p.inventoryLevel,  p.maxInventory, p.minInventory,  p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]
    
    # Copie des pizzerias pour la recherche de solution optimale
    pizzerias = deepcopy(Instance_pizzerias)

    # On crée une solution initiale avec l'ordre décrit dans le fichier d'instance
    assignment, Updated_pizzerias = greedy_assignment(pizzerias, N, T, M, Q)
    Order = [pizzeria.id-1 for pizzeria in pizzerias]
    for pizzeria in Updated_pizzerias:
        Instance_pizzerias[pizzeria.id-1].daily_deliveries = pizzeria.daily_deliveries

    # On stocke la solution intiale et son score
    best_raw_solution = generate_raw_solution(assignment, Instance_pizzerias, M, T, Order)
    best_cost, _ = instance.solution_cost_and_validity(Solution(instance.npizzerias, best_raw_solution))


    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:

        t1 = time()

        pizzerias = deepcopy(Instance_pizzerias)
        #restart aléatoire depuis la solution initiale
        random.shuffle(pizzerias)

        current_cost, current_raw_solution = generate_and_evaluate(pizzerias, instance, N, T, M, Q, Instance_pizzerias)
        best_cost_restart, best_raw_sol_restart = current_cost, current_raw_solution

        temp = 200

        for _ in range(200):

            # On génère la liste de tous les voisins obtenus à partir des 2-swaps depuis la solution courante
            neighbors = two_swap(pizzerias)

            # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.
            neigh = random.choice(neighbors)

            cost, raw_solution = generate_and_evaluate(neigh, instance, N, T, M, Q, Instance_pizzerias)     
            delta = cost - current_cost

            if delta <= 0 or np.random.rand() < np.exp(-delta/temp):
                current_cost, current_raw_solution = cost, raw_solution

                # On choisit comme représentation courante le meilleur voisin
                pizzerias = neigh
            
            list_costs.append(current_cost)

            if current_cost < best_cost_restart:
                best_cost_restart    = current_cost
                best_raw_sol_restart = current_raw_solution
            
            temp *= 0.97

        if best_cost_restart < best_cost:
            best_cost  = best_cost_restart
            best_raw_solution = best_raw_sol_restart
        
        iteration_duration = time() - t1

    return Solution(instance.npizzerias, best_raw_solution)
