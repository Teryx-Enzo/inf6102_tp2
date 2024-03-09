from utils import *
import random
class CustomPizzeria(Pizzeria):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def calculer_demande(self):
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


    quelque_chose = instance.pizzeria_dict.items()

    nos_pizzerias = [CustomPizzeria(p.id, p.x, p.y,  p.maxInventory, p.minInventory, p.inventoryLevel, p.dailyConsumption ,p.inventoryCost ) for _,p in list(instance.pizzeria_dict.items())]



    #ca marche pas parce que je modifie l'instance
    sol_raw=[]
    
    for t in range(instance.T):
        timestep=[]
        Q = instance.Q
        for truck_id in range(instance.M):
            truck_capacity = Q
            truck_route=[]
            temp_pizz = nos_pizzerias.copy()
            for pizzeria in nos_pizzerias:
                    if pizzeria.inventoryLevel - pizzeria.dailyConsumption < pizzeria.L:

                        livraison = pizzeria.dailyConsumption if  truck_capacity > pizzeria.dailyConsumption else 0
                        if livraison > 0:
                            truck_route.append((pizzeria.id,livraison))
                            truck_capacity -= livraison
                            pizzeria.inventoryLevel += livraison
                            temp_pizz.remove(pizzeria)
            timestep.append(truck_route)
        sol_raw.append(timestep)

        for pizzeria in nos_pizzerias:
             pizzeria.inventoryLevel -= pizzeria.dailyConsumption
             
    return Solution(instance.npizzerias,sol_raw)
    
