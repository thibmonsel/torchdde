import time

import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
from torchdde import ConstantStepSizeController
from torchdde.solver import Dopri5


torch.set_default_dtype(torch.float64)


def vf(t, y, args):
    return -y + t**2


def vf2(t, y):
    return -y + t**2


# with torch.no_grad():
#     ts = torch.linspace(0, 2.5, 5000)
#     y0 = torch.rand((3,1))
# ys = integrate(vf, Dopri5(), ts, y0, None, stepsize_controller=AdaptiveStepSizeController(10e-6, 10-6))
# ys2 = integrate(vf, RK4(), ts, y0, None)
# ys3 = integrate(vf, RK2(), ts, y0, None)


t0, tf = 0.0, 5.0
y0 = torch.rand((3, 1))
solver = Dopri5()
# stepsize_controller = AdaptiveStepSizeController(10e-3, 10e-6)
stepsize_controller = ConstantStepSizeController()
tnext, controller_state = stepsize_controller.init(
    vf, t0, tf, y0, torch.tensor(0.01), None, solver.order()
)
ys0 = torch.unsqueeze(y0.clone(), dim=1)
l_ts = [t0]
tprev = torch.tensor(t0)
t = time.time()
i = 0
while tprev < tf:
    i += 1
    y_candidate, y_error, dense_info, _ = solver.step(
        vf, tprev, ys0[:, -1], controller_state, None, has_aux=False
    )
    keep_step, tprev, tnext, controller_state = stepsize_controller.adapt_step_size(
        vf,
        tprev,
        tnext,
        ys0[:, -1],
        y_candidate,
        None,
        y_error,
        solver.order(),
        controller_state,
    )
    if keep_step:
        l_ts.append(torch.clip(tprev, max=torch.tensor(tf)))
        ys0 = torch.cat((ys0, torch.unsqueeze(y_candidate, dim=1)), dim=1)
    if i > 1500:
        break

print("time took", time.time() - t)
t_eval = torch.tensor(l_ts).flatten()
print(t_eval)
ys4 = solve_ivp(
    vf2, [t0, tf], y0[0], method="RK23", t_eval=t_eval.numpy(), rtol=10e-3, atol=10e-6
).y
print(ys4.dtype)
# plt.plot(ts, ys[0], label="dopri5")
plt.plot(t_eval, ys0[0, :, 0].numpy(), label="dopri5_manuel")
plt.plot(t_eval, ys4[0], "-.", label="scipy")
plt.legend()
plt.show()
