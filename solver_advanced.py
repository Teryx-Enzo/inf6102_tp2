from utils import *
import random
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

def solve(instance: Instance) -> Solution:
    """
    This function generates a solution where at each timestep
    the first truck goes through every pizzeria and delivers pizzeria.dailyConsumption

    Args:
        instance (Instance): a problem instance

    Returns:
        Solution: the generated solution
    """

    nos_pizzerias = [CustomPizzeria(p.id, p.x, p.y,  p.maxInventory, p.minInventory, p.inventoryLevel, p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]



    sol_raw=[]
    Q = instance.Q


    for t in range(instance.T):
        timestep=[]
        pizzeria_a_livrer = [pizz for pizz in nos_pizzerias if pizz.calculer_demande()]
        
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
    
