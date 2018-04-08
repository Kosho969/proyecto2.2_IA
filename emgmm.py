import math
import numpy


'''
    f(*x*) = (2*pi)^(-N/2) * |sigma|^(-1/2) e^(-0.5(x - mu)^T sigma^1 (x - mu))
'''
def probabilidad_no_normalizada(sigma, mu, punto):
    N = 2

    det_sigma = abs(numpy.linalg.det(sigma))
    subs = numpy.subtract(punto, mu)
    first = numpy.matmul(numpy.transpose(subs), sigma)


    exp = (
            (-0.5)
            * numpy.matmul(first, subs)
    )

    result = math.pow((2 * math.pi), -(N/2))
    result = result * det_sigma ** (-1/2)
    result = result * math.pow(math.e, exp)

    return result

sigma = numpy.array([[8., 3.], [4., 2.]])
mu = (3, 4)
x = (3, 4)
print(sigma)

# print(sigma)
# print(mu)
# print(x)
print(probabilidad_no_normalizada(sigma, mu, x))
