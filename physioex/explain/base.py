from abc import ABC, abstractmethod

class PhysioExplainer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def explain(self):
        pass