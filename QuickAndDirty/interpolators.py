import torch
import numpy as np

class TorchLinearInterpolator():

    def __init__(self,vals,time=None):

        self.vals = vals
        if time is None: self.time = torch.arange(vals.shape[0])
        else: 
            self.time = time

        self.update_coeffs()

    def update_coeffs(self):

        if self.vals.shape[1]==1:
            self.coeffs = None
        else:

            self.coeffs = self.vals[:,1:]-self.vals[:,:-1]
            self.coeffs = self.coeffs / (self.time[1:] - self.time[:-1]).view(1,-1,1)
    
    def __call__(self,t):
        if self.coeffs is None:
            return self.vals[:,0]

        if t in self.time:
            t_i = torch.where(self.time==t)[0][0]
            return self.vals[:,t_i]

        ret = torch.where(t>=self.time)[0]
        if len(ret)==0:
            if abs(t-self.time[0])>1e-6: 
                print('No Extrapolation',t,self.time[0],self.time[-1])
                return None
            else : ret = [0]
        else:
            if t>self.time[-1] :
                print('No Extrapolation',t,self.time[0],self.time[-1])
                return None

        t_i = ret[-1]

        dt = t-self.time[t_i]

        ret = self.vals[:,t_i] + dt * self.coeffs[:,t_i]
        return ret

    def add_point(self,t,val):

        if t in self.time:
            t_i = torch.where(self.time==t)[0][0]
            self.vals[:,t_i] = val
            
        else:
            ret = torch.where(t>self.time)[0]

            t_to_add = torch.tensor(t).to(self.time.device,self.time.dtype).reshape(1)
            v_to_add = val.reshape(val.shape[0],1,*val.shape[1:])
            if len(ret) == 0:
                self.vals = torch.cat([v_to_add,self.vals],dim=1)
                self.time = torch.cat([t_to_add,self.time])
            elif t > self.time[-1]:
                self.vals = torch.cat([self.vals,v_to_add],dim=1)
                self.time = torch.cat([self.time,t_to_add])
            else:
                self.vals = torch.cat([self.vals[:,:t_i+1],v_to_add,
                                        self.vals[:,t_i:]],dim=1)
                self.time = torch.cat([self.time[:t_i+1],
                                        t_to_add,self.time[t_i:]])
        
        self.update_coeffs()



