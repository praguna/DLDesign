import numpy as np
class LossFunction(object):
    #This class contains methods of loss functions and their derivatives
    @staticmethod
    def compute_loss(T,P,type:str):
        if type == "rms":
            return LossFunction.rms(T,P)
        elif type == "cross_entropy":
            return LossFunction.cross_entropy(T,P)

    @staticmethod
    def compute_derivative(T,P,type:str):
        if type == "rms":
            return LossFunction.rms_prime(T,P)
        elif type == "cross_entropy":
            return LossFunction.cross_entropy_prime(T,P)  

    @staticmethod
    def mse(T,P):
        return np.mean(np.sum(np.square(T-P)))

    @staticmethod
    def mse_prime(T,P):
        N = P.shape[1]
        return 2*(T-P)/N    

    @staticmethod
    def rms(T,P):
        return np.sqrt(np.square(np.sum(P-T)).mean())

    @staticmethod
    def rms_prime(T,P):
        N = P.shape[1]
        return P/np.sqrt(N)

    @staticmethod
    def cross_entropy(T,P,esp=1e-10):
        #not dividing by n for testing sake
        #N = 1
        #for i in P.shape : N*=i
        return -np.sum(T*np.log10(P+esp) + (1 - T)* np.log10(1 - P + esp))

    @staticmethod
    def cross_entropy_prime(T,P,esp=1e-10):
        return (1 - T) * 1/(1 - P+esp) - T * 1/(P+esp)