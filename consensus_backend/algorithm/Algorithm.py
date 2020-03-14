from abc import ABCMeta, abstractmethod

class Algorithm(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass
