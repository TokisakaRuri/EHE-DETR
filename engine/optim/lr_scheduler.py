import math
from functools import partial
import matplotlib.pyplot as plt
from ..extre_module.utils import plt_settings, TryExcept

def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) / (total_iter - flat_iter - no_aug_iter)))
        return min_lr + (init_lr - min_lr) * cosine_decay

class FlatCosineLRScheduler:

    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs, warmup_iter, flat_epochs, no_aug_epochs, scheduler_type='cosine', lr_scyedule_save_path=None):
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        total_iter = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)
        if flat_iter > total_iter:
            flat_iter = total_iter - warmup_iter
        print(self.base_lrs, self.min_lrs, total_iter, warmup_iter, flat_iter, no_aug_iter)
        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)
        for (i, _) in enumerate(optimizer.param_groups):
            plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, self.base_lrs[i], self.min_lrs[i], lr_scyedule_save_path / f'lr_schedule_{i}.png')

    def step(self, current_iter, optimizer):
        for (i, group) in enumerate(optimizer.param_groups):
            group['lr'] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
        return optimizer

@TryExcept('WARNING ⚠️ plot_lr_schedule failed.')
@plt_settings()
def plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr, save_path):
    is_four_stage = True
    iters = list(range(total_iter))
    if flat_iter == total_iter - warmup_iter:
        is_four_stage = False
    lrs = [flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, i, init_lr, min_lr) for i in iters]
    plt.figure(figsize=(8, 5))
    plt.plot(iters, lrs, label='Learning Rate')
    plt.axvline(x=warmup_iter, color='r', linestyle='--', label='Warmup End')
    if is_four_stage:
        plt.axvline(x=flat_iter, color='g', linestyle='--', label='Flat End')
        plt.axvline(x=total_iter - no_aug_iter, color='b', linestyle='--', label='No Aug Start')
    else:
        plt.axvline(x=flat_iter + warmup_iter, color='g', linestyle='--', label='Flat End')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Flat Cosine Learning Rate Schedule')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close('all')
if __name__ == '__main__':
    total_iter = 10000
    warmup_iter = 1000
    flat_iter = 3000
    no_aug_iter = 500
    init_lr = 0.01
    min_lr = 0.0001
    plot_lr_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, init_lr, min_lr)
