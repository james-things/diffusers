# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for diffusion models."""

import math
from enum import Enum
from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .utils import logging


logger = logging.get_logger(__name__)


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    PIECEWISE_CONSTANT = "piecewise_constant"
    ONE_CYCLE = "one_cycle"


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_piecewise_constant_schedule(optimizer: Optimizer, step_rules: str, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        step_rules (`string`):
            The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
            if multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
            steps and multiple 0.005 for the other steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    rules_dict = {}
    rule_list = step_rules.split(",")
    for rule_str in rule_list[:-1]:
        value_str, steps_str = rule_str.split(":")
        steps = int(steps_str)
        value = float(value_str)
        rules_dict[steps] = value
    last_lr_multiple = float(rule_list[-1])

    def create_rules_function(rules_dict, last_lr_multiple):
        def rule_func(steps: int) -> float:
            sorted_steps = sorted(rules_dict.keys())
            for i, sorted_step in enumerate(sorted_steps):
                if steps < sorted_step:
                    return rules_dict[sorted_steps[i]]
            return last_lr_multiple

        return rule_func

    rules_func = create_rules_function(rules_dict, last_lr_multiple)

    return LambdaLR(optimizer, rules_func, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_one_cycle_schedule(
    optimizer: Optimizer, max_lr: float, num_training_steps: int, pct_start: float = 0.3, anneal_strategy: str = 'cos',
    cycle_momentum: bool = True, base_momentum: float = 0.85, max_momentum: float = 0.95, div_factor: float = 25.0,
    final_div_factor: float = 10000.0, three_phase: bool = False, last_epoch: int = -1, verbose: bool = False
):
    """
    Creates a OneCycleLR schedule with learning rate that varies according to the OneCycleLR 
    policy. This policy was initially described in the paper `Super-Convergence: Very Fast 
    Training of Neural Networks Using Large Learning Rates` (https://arxiv.org/abs/1708.07120).

    The learning rate starts from a small initial value, gradually increases to the max learning rate 
    across the first portion (default: 30%) of the training steps, and then gradually decreases to the 
    base learning rate across the remaining steps.

    There are many optional arguments which have been incorporated to allow for flexible use. 
    Please see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html for 
    more information about this scheduler.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        max_lr (float or list):
            The peak/maximum learning rate boundaries in the cycle for each parameter group.
        num_training_steps (int): 
            The total number of training steps for the scheduler.
        pct_start (float, optional):
            The percentage of the cycle (in number of steps) spent increasing the learning rate. Default: 0.3
        anneal_strategy (str, optional):
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for linear annealing. Default: 'cos'
        cycle_momentum (bool, optional):
            If True, momentum is cycled inversely to learning rate between 'base_momentum' and 'max_momentum'. Default: True
        base_momentum (float or list, optional):
            Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate; 
            at the peak of a cycle, momentum is 'base_momentum' and learning rate is 'max_lr'. Default: 0.85
        max_momentum (float or list, optional):
            Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is 'max_momentum' and learning rate is 'base_lr'. Default: 0.95
        div_factor (float, optional):
            Determines the initial learning rate via initial_lr = max_lr / div_factor. Default: 25.0
        final_div_factor (float, optional):
            Determines the minimum learning rate via min_lr = initial_lr / final_div_factor. Default: 10000.0
        three_phase (bool, optional):
            If True, use a third phase of the schedule to annihilate the learning rate according to 'final_div_factor' instead of modifying the second phase.
            The first two phases will be symmetrical about the step indicated by 'pct_start'. Default: False
        last_epoch (int, optional): 
            The index of the last epoch when resuming training. Default is -1, which means start from scratch.
        verbose (bool, optional):
            Specify whether the adjustment of the learning rate should be logged with each step.
        
    Return:
        `torch.optim.lr_scheduler.OneCycleLR` with the appropriate schedule.
    """
    return OneCycleLR(
        optimizer, max_lr, total_steps=num_training_steps, last_epoch=last_epoch, pct_start=pct_start, 
        anneal_strategy=anneal_strategy, cycle_momentum=cycle_momentum, base_momentum=base_momentum, 
        max_momentum=max_momentum, div_factor=div_factor, final_div_factor=final_div_factor, three_phase=False, 
        verbose=verbose
    )


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.PIECEWISE_CONSTANT: get_piecewise_constant_schedule,
    SchedulerType.ONE_CYCLE: get_one_cycle_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
    pct_start: Optional[float] = 0.3, 
    anneal_strategy: Optional[str] = 'cos',
    cycle_momentum: Optional[bool] = True, 
    base_momentum: Optional[float] = 0.85, 
    max_momentum: Optional[float] = 0.95, 
    div_factor: Optional[float] = 25.0,
    final_div_factor: Optional[float] = 10000.0, 
    three_phase: Optional[bool] = False, 
    verbose: Optional[bool] = False
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        step_rules (`str`, *optional*):
            A string representing the step rules to use. This is only used by the `PIECEWISE_CONSTANT` scheduler.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        pct_start (`float`, *optional*, defaults to 0.3): 
            Percentage of total steps to take to reach max lr in `ONE_CYCLE` scheduler.
        anneal_strategy (`str`, *optional*, defaults to "cos"): 
            Annealing strategy used in `ONE_CYCLE` scheduler.
        cycle_momentum (`bool`, *optional*, defaults to True): 
            Cycle momentum value used in `ONE_CYCLE` scheduler.
        base_momentum (`float`, *optional*, defaults to 0.85): 
            Base momentum value used in ONE_CYCLE scheduler.
        max_momentum (`float`, *optional*, defaults to 0.95): 
            Maximum momentum value used in ONE_CYCLE scheduler.
        div_factor (`float`, *optional*, defaults to 25.0): 
            Determines initial LR used by ONE_CYCLE scheduler.
        final_div_factor (`float`, *optional*, defaults to 10000.0): 
            Determines final LR in ONE_CYCLE scheduler.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, last_epoch=last_epoch)

    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, step_rules=step_rules, last_epoch=last_epoch)

    # OneCycle requires `num_training_steps` and not `num_warmup_steps`.
    if name == SchedulerType.ONE_CYCLE:
        if num_warmup_steps is not None and not num_warmup_steps == 0:
            print(f'{name} is not compatible with num_warmup_steps, which will be ignored.')
        
        # Retrieve learning rate from optimizer
        max_lr = optimizer.param_groups[0]['lr']
        print("Initializing OneCycleLR with the following parameters:\n" + 
            f"max_lr: {max_lr}\n" + 
            f"num_training_steps: {num_training_steps}")

        return schedule_func(
            optimizer, 
            max_lr=max_lr, 
            num_training_steps=num_training_steps,
            pct_start=pct_start, 
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum, 
            base_momentum=base_momentum, 
            max_momentum=max_momentum, 
            div_factor=div_factor,
            final_div_factor=final_div_factor, 
            three_phase=three_phase, 
            verbose=verbose,
        )

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
        )


    return schedule_func(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, last_epoch=last_epoch
    )
