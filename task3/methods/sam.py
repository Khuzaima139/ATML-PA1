import torch


class SAM:
    def __init__(self, params, opt, rho):
        self.params = [p for p in params if p.requires_grad]
        self.opt = opt
        self.rho = rho

    def step(self, closure):
        self.opt.zero_grad()
        loss, logs = closure()
        loss.backward()
        grads = [p.grad for p in self.params if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)

        saved = []
        with torch.no_grad():
            for p in self.params:
                saved.append(p.detach().clone())
                if p.grad is not None:
                    p.add_(self.rho * p.grad / (norm + 1e-12))

        self.opt.zero_grad()
        loss_adv, _ = closure()
        loss_adv.backward()

        with torch.no_grad():
            for p, s in zip(self.params, saved):
                p.copy_(s)
        self.opt.step()
        return loss, {**logs, "cls_perturbed": loss_adv.item()}