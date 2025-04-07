# FAQ

## How would I define a DDE with several delays ?

You just have to specify a delay tensor size that corresponds to the number of delays you desire.

```python
solver = ....
delays = torch.tensor([1.0, 2.0])
history_function = lambda t : ...
ts = ...

def simple_dde(t, y, func_args, *, history):
    # this correspond to y'(t) = -y(t-1) - y(t-2)
    return - history[0] - history[1]

ys = torchdde.integrate(f, solver, ts[0], ts[-1], ts, history_function, func_args=None, dt0=ts[1]-ts[0], delays=delays)
```

## How about if I want a neural network to have also several delays ?

Well if it's the same forward pass in the [Neural DDE](./neural-dde.md), then nothing needs to be changed ! The term `torch.cat([z, *history], dim=-1)` unpacks all the delayed terms.

## What if my trajectories have different evaluation points ? 

`torchdde` was developed with in mind that the evaluation points `ts` are the same for all trajectories.  
Nonetheless, if this isn't the case, that's not an issue. One could think of doing the following : 

```python


# creating different evaluation times ts 
different_ts_list = [torch.linspace(0, 10, torch.randint(1, 10, (1,)).item()) for _ in range(4)]
all_values = torch.cat(different_ts_list)
# taking all unique values from `different_ts_list`
ts, _ = torch.sort(all_values.unique())

# integrate your ODE/DDE with ts 
# after successful integration, we will have to take the values of ys at the times ts_list
ys = torch.rand((4, ts.shape[0], 2))

ys_list_pred = []
for ys_, ts_ in zip(ys, different_ts_list): 
    # creating a mask to get the values of ys at the times ts_list
    mask = torch.where(torch.isin(ts, ts_))
    assert torch.all(different_ts_list[0] == ts[mask]), "Error"
    ys_subset = ys_[mask]
    ys_list_pred.append(ys_subset)
```
