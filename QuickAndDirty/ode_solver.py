from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch


class AbstractOdeSolver(ABC):
 
    @abstractmethod
    def step(self, func, t, y, dt):
        pass
    
    def integrate(self, func, ts, y0):
        dt = ts[1] -ts[0]
        ys = y0.clone().detach()
        current_y = y0
        for current_t in ts[1:]:
            y = self.step(func, current_t, current_y, dt)
            current_y = y
            ys = torch.cat((ys, current_y))
        return ys
    

class Euler(AbstractOdeSolver):
    
    def __init__(self):
        super().__init__()
        
    def step(self, func, t, y, dt):
        return y + dt * func(t, y)
    
class RK2(AbstractOdeSolver):
    def __init__(self):
        super().__init__()
        
    def step(self, func, t, y, dt):
        k1 = func(t, y)
        k2 = func(
            t + dt,
            y + dt * k1
        )
        return y + dt / 2 * (k1 + k2)
    
class Ralston(AbstractOdeSolver):
    def __init__(self):
        super().__init__()
        
    def step(self, func, t, y, dt):
        k1 = func(t, y)
        k2 = func(
            t + 2/3*dt,
            y + 2/3*dt * k1
        )
        return y + dt * (1/4*k1 + 3/4*k2)

class RK4(AbstractOdeSolver):
    def __init__(self):
        super().__init__()
        
    def step(self, func, t, y, dt):
        k1 = func(t, y)
        k2 = func(
            t + 1/2*dt,
            y + 1/2*dt * k1
        )
        k3 = func(t + 1/2*dt, y + 1/2*dt * k2)
        k4 = func(t+dt, y + dt * k3)
        return y + 1/6 * dt * (k1 + 2 *k2 + 2*k3 + k4)
    
