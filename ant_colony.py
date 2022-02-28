import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import math

class ant_colony:
    #inital
    def __init__(self,cor_point,alpha=1,beta=2,e=0.15,num_ant = 10, iteration = 20):
        self.cor_point = cor_point
        self.alpha = alpha
        self.beta = beta
        self.e = e
        self.num_ant = num_ant
        self.iteration = iteration

        
        self.distance = self.distance_generation(cor_point)     #generate distance matrix here
        prob_city = 1/self.distance                             #transfer distance to probility
        prob_city[prob_city==math.inf] =0                       # x/0 = inf, so zeroing out an infinite value
        self.prob_city = np.array(prob_city)
    

    def run(self):
        route = np.ones((self.num_ant,self.distance.shape[0]+1))        #create a numpy array for route. N cities need N+1 route point(we need go back to the start city).
        pheromne = 0.1*np.ones((self.num_ant,self.distance.shape[0]))   #crate a numpy array for pheromne
        cost_list = []
        for iteration in range(self.iteration):
            route[:,0] = 1                                              #set the first city as our start city

            for i in range(self.num_ant):
                
                temp_prob = self.prob_city.copy()                       
                temp_prob[:,0] = 0                                      #since we start from first city, we set the probility of first city to 0
                for j in range(self.distance.shape[0]-1):

                    current = int(route[i,j]-1)
                    
                    r_para = np.power(pheromne[current,:],self.beta)    #from formula
                    s_para = np.power(temp_prob[current,:],self.alpha)  #from formula

                    combine = np.multiply(r_para,s_para)                #from formula

                    prob = combine/np.sum(combine)                      #from formula

                    next_city = self.roulette(prob)+1                   #roulette wheel to generate next city
                    route[i,j+1] = next_city
                    
                    temp_prob[:,(next_city-1)] = 0                      #After traveling the city, setting probility to 0, we will never go to this city again

            distance_cost = self.distance_calc(route,self.distance)     

            min_distance_index = np.argmin(distance_cost)               #The index of min distance route
            min_distance_cost = distance_cost[min_distance_index]

            best_route = route[min_distance_index]

            pheromne = self.pheromne_calc(pheromne,distance_cost,route) #update pherome
            cost_list.append(np.sum(distance_cost))


        self.best_route = best_route                                    #final result here
        self.min_distance = min_distance_cost
        self.cost_list = cost_list

    #show best result, not necessary, just ignore
    def result(self):
        print(self.best_route)
        f, img = plt.subplots(1,2)
        for i in range(len(self.best_route)-1):
            x1 = int(self.best_route[i])-1
            x2 = int(self.best_route[i+1])-1
            
            img[0].plot([self.cor_point[x1,0],self.cor_point[x2,0]],[self.cor_point[x1,1],self.cor_point[x2,1]])

        img[0].set_title('Best route')
        img[0].set(xlabel='x', ylabel='y')

        img[1].plot(self.cost_list)
        img[1].set_title('Total cost')
        img[1].set(xlabel='Iteration', ylabel='Distance')

        plt.show()

        

    #Generate distance matrix from cordinate points
    def distance_generation(self,cor_point):
        return spatial.distance.cdist(cor_point,cor_point,metric='euclidean')

    #roulette wheel
    def roulette(self,prob):
        cum_prob = np.cumsum(prob)
        cum_prob = cum_prob/cum_prob[-1]
        
        sample = np.random.uniform()
        for i in range(cum_prob.shape[0]):
            if cum_prob[i] > sample:
                break
            else:
                continue
        return i

    #Calculate total distance of route
    def distance_calc(self,route,distance_matrix):
        distance_count = []
        for i in range(route.shape[0]):
            count = 0
            for j in range(route.shape[1]-1):
                count += distance_matrix[int(route[i,j])-1,int(route[i,j+1])-1]
            distance_count.append(count)
        return np.array(distance_count)
    
    #Update pheromne after each iteration    p = (1-e)*p + dt
    def pheromne_calc(self,pheromne,distance_cost,rute):
        pheromne = (1-self.e)*pheromne
        for i in range(rute.shape[0]):
            dt = 1/distance_cost[i]
            for j in range(rute.shape[1]-1):
                pheromne[int(rute[i,j]-1),int(rute[i,j+1]-1)] += dt
        return pheromne


cor_point = np.random.rand(20,2)            #Generate 20 cities position


a = ant_colony(cor_point=cor_point,alpha=20,beta=15,e=0.15,num_ant=100)     #Set parameters
a.run()                                     
a.result()                                  #Show result