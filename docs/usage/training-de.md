# Training

Two following ways are possible to train Neural DDE / Neural ODE :

- optimize-then-discretize (with the adjoint method)
- discretize-then-optimize (regular backpropagation)

Please see the doctorial thesis [On Neural Differential Equations](https://arxiv.org/pdf/2202.02435.pdf) for more information on both procedures.

To choose from either two possibilities is easy, you just need to set `discretize_then_optimize` to `True` or `False`.

!!! warning

    You are unable to learn the DDE's delays if using the `discretize_then_optimize=True`.


If you are training an ODE then `delays=None` and `y0` is a `Tensor`.  
If you are training an DDE then `delays` is a `Tensor` and `y0` is a `Callable`. 

A simple training loop would look like: 

```python
for step, data in enumerate(train_loader):
    optimizer.zero_grad()
    data = data.to(device)
    ys_pred = integrate(
        model,
        solver=...,
        t0=ts[0],
        t1=ts[-1],
        ts=ts,
        y0=...,
        args=None,
        dt0=ts[1] - ts[0],
        delays=...,
    )
    loss = loss_fn(ys_pred, data)
    loss.backward()
    optimizer.step()

```