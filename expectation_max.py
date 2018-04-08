
import random
import numpy as np
from gaussian import Gaussian
import matplotlib.pyplot as plt
import matplotlib

class Expectation_maximization():

    def __init__(self, data_set, k_values , max_iterations = None):
        self.pi_values = self.generate_pis(k_values)
        self.execute_maximization(k_values, data_set, max_iterations)

    def execute_maximization(self, k_values, data_set, max_iterations):
        gaussians = [ Gaussian(self.generateMu(), self.generateSigma(), pi) for pi in self.pi_values]
        
        previous_gaussians = ''
        current_gaussians = ''
        
        for i in range(len(gaussians)):
            current_gaussians += gaussians[i].to_string()

        self.plot_gaussians(gaussians)

        current_iteration = 0
        print('Max: ' ,max_iterations)
        while((current_gaussians != previous_gaussians) and (current_iteration <= max_iterations)):
            #print(current_iteration <= max_iterations)
            posterior = np.zeros((len(data_set),k_values))
            e_ijs = np.zeros((len(data_set),k_values))
            # Paso E :
            for j in range(len(data_set)):
                R = 0.0

                for i in range(len(gaussians)):
                    posterior[j, i] = gaussians[i].pi * gaussians[i].probabilidad_no_normalizada(data_set[j])
                    R += posterior[j,i]

                for i in range(len(gaussians)):
                    posterior[j,i] = posterior[j,i]/R
            #print(posterior)

            # Paso M :
            for i in range(len(gaussians)):

                addition = 0
                for j in range(len(data_set)):
                    addition += posterior[j,i]

                new_pi = addition/len(data_set)

                addition_mu = 0
                for j in range(len(data_set)):
                    addition_mu += posterior[j,i] * data_set[j]

                new_mu = addition_mu / addition

                addition_sigma = 0
                for j in range(len(data_set)):
                    addition_sigma += posterior[j,i] * gaussians[i].deviation(data_set[j])

                new_sigma = addition_sigma / addition
                gaussians[i].set_pi( new_pi )
                gaussians[i].set_mu( new_mu )
                gaussians[i].set_sigma( new_sigma )


            current_iteration += 1
        self.plot_gaussians(gaussians)


    def generateMu(self):
        return np.array([[random.randint(-500, 500)], [random.randint(-500, 500)]])

    def generateSigma(self):
        sigma = np.array([[1,0],[0,1]]) * random.random()
        return np.array(sigma)

    def generate_pis(self, k_values):
        piCuts = sorted([random.random() for i in range(k_values - 1)])
        pis = [piCuts[0]]
        for i in range(1, len(piCuts)):
          pis.append(piCuts[i] - piCuts[i-1])
        pis.append(1-piCuts[-1])
        return pis

    def plot_gaussians(self, gaussians):
        figures = []
        for gaussian in gaussians:
            figures.append(np.random.multivariate_normal([gaussian.mu[0,0], gaussian.mu[1,0]], gaussian.sigma, 10000))
        for figure in figures:
            plt.scatter(figure[:,0],figure[:,1],c='r',s=20,edgecolors='none')
        plt.show()