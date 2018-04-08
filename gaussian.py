import numpy
import math
class Gaussian():

    def __init__(self, mu, sigma, pi):
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def set_mu(self, mu):
        self.mu = mu

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_pi(self, pi):
        self.pi = pi

    def to_string(self):
        return 'Mu:\n {},\nSigma:\n {},\nPi:\n {} \n'.format(self.mu, self.sigma, self.pi)

    def deviation(self, punto):
        # new_mu = (self.mu[0], self.mu[1])
        # punto = (punto[0], punto[1])
        dev = numpy.subtract(punto, self.mu)
        #print('Deviation: ', dev)
        result = (numpy.transpose(dev) * dev)
        #print('Result: ', result)
        return result

    def probabilidad_no_normalizada(self, punto):
        N = 2
        # new_mu = (self.mu[0], self.mu[1])
        # punto = (punto[0], punto[1])
        det_sigma = abs(numpy.linalg.det(self.sigma))
        dev = numpy.subtract(punto, self.mu)
        exp = (
                (-0.5)
                * numpy.matmul(
                    numpy.transpose(
                        dev
                    ),
                    numpy.matmul(
                        numpy.linalg.inv(self.sigma),
                        dev
                    )
                )
        )
        result = math.pow((2 * math.pi), -(N/2))
        result = result * det_sigma ** (-1/2)
        result = result * math.pow(math.e, exp)
        if result == 0:
            return 0.1
        else:
            return result
