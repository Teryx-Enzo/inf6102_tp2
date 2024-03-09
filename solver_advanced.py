from utils import *
import random
import numpy as np


class CustomPizzeria(Pizzeria):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def calculer_demande(self):

        #On calcule la demande et on la renvoie si elle est positive
        demande = self.L+self.dailyConsumption-self.inventoryLevel
        
        return demande if demande > 0 else 0


def clustering(Q: int, n: int, Pizzerias: Dict[int, CustomPizzeria], To_deliver: List[int]) -> Dict[int, int]:
    """
    Génère n ensembles de pizzerias regroupées par distance, telle que la somme de leurs demandes soit inférieure à la quantité Q transportable.

    Args:
        Q (int) : la capacité des camions
        n (int) : le nombre de clusters souhaités
        Pizzerias (List[CustomPizzeria]) : le dictionnaire des pizzerias 
        To_deliver (List[int]) : la liste des indexs des pizzerias à regrouper

    Returns:
        Clusters (List[List[int]]) : une liste de n clusters d'indexs de pizzerias proches géographiquement, et dont la somme des demande n'excède pas Q.
    """
    Initial_pizzerias = random.sample(To_deliver, k=n) # Pizzerias avec lesquelles on initialise les clusters

    Clusters = dict(zip(range(n),
                        [[[idx], Pizzerias[idx].calculer_demande()] for idx in Initial_pizzerias]))

    for p_id in To_deliver:
        if p_id not in Initial_pizzerias:
            print(p_id)
            # On met les pizzerias restantes dans des clusters si c'est possible
            placed = False

            for c_id in Clusters:
                if Clusters[c_id][1]+Pizzerias[p_id].calculer_demande() < Q:
                    Clusters[c_id][0].append(p_id)
                    Clusters[c_id][1] += Pizzerias[p_id].calculer_demande()
                    placed = True
                    break
            
            if not placed:
                return False

    converged = True
    while not converged:
        pass

    return Clusters


def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """
    Q, M = instance.Q, instance.M

    nos_pizzerias = {id: CustomPizzeria(p.id, p.x, p.y, p.maxInventory, p.minInventory, p.inventoryLevel, p.dailyConsumption, p.inventoryCost)
                      for id, p in instance.pizzeria_dict.items()}
    
    print(clustering(Q, 2, nos_pizzerias, list(nos_pizzerias.keys())))

    nos_pizzerias = [CustomPizzeria(p.id, p.x, p.y,  p.maxInventory, p.minInventory, p.inventoryLevel, p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]



    sol_raw=[]
    Q = instance.Q


    for t in range(instance.T):
        timestep=[]
        pizzeria_a_livrer = [id 
                             for id, pizz in nos_pizzerias.items()
                             if pizz.calculer_demande() > 0] # liste d'indices 
        
        for truck_id in range(instance.M):
            truck_capacity = Q
            truck_route=[]
            temp_pizz = nos_pizzerias.copy()
            for pizzeria in nos_pizzerias:
                    demande = pizzeria.calculer_demande()
                    if demande:
                        livraison = demande if  truck_capacity > demande else 0
                        if livraison > 0:
                            truck_route.append((pizzeria.id,livraison))
                            truck_capacity -= livraison
                            pizzeria.inventoryLevel += livraison
                            temp_pizz.remove(pizzeria)
                    else :
                        temp_pizz.remove(pizzeria)
            timestep.append(truck_route)
        sol_raw.append(timestep)

        for pizzeria in nos_pizzerias:
             pizzeria.inventoryLevel -= pizzeria.dailyConsumption
             
    return Solution(instance.npizzerias,sol_raw)
    
