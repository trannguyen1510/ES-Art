#!/usr/bin/env python3

from datetime import datetime
import numpy as np
import os
from PIL import Image
from utils import (isnotebook, arr2img, save_as_gif, save_as_frames)


def _tell_fn_pgpe(solver, solutions, fitnesses):
    solver.tell(fitnesses)  # PGPE maximizes.


def get_tell_fn(flavor='pgpe'):
    return {'pgpe': _tell_fn_pgpe}[flavor]


def _best_params_fn_pgpe(solver):
    return solver.center


def get_best_params_fn(flavor='pgpe'):
    return {'pgpe': _best_params_fn_pgpe}[flavor]


class Hook(object):
    def __init__(self):
        pass

    def __call__(self, i, solver, fitness_fn, best_params_fn):
        raise NotImplementedError

    def close(self):
        pass


class PrintStepHook(Hook):
    def __init__(self):
        super().__init__()

    def __call__(self, i, solver, fitness_fn, fitnesses_fn, best_params_fn):
        print(i, end=' ... ')

    def load(self):
        print('*Loading checkpoint')


class PrintCostHook(Hook):
    def __init__(self, fitnesses_fn_is_wrapper=True):
        super().__init__()
        self.fitnesses_fn_is_wrapper = fitnesses_fn_is_wrapper

    def __call__(self, i, solver, fitness_fn, fitnesses_fn, best_params_fn):
        best_params = best_params_fn(solver)
        if self.fitnesses_fn_is_wrapper:
            cost = fitnesses_fn(fitness_fn, [best_params])
        else:
            cost = fitnesses_fn([best_params])
        print()
        print(f'[{datetime.now()}]   Iteration: {i}   cost: {cost}')

    def load(self):
        pass


class SaveCostHook(Hook):
    def __init__(self, save_fp, fitnesses_fn_is_wrapper=True):
        super().__init__()
        self.save_fp = save_fp
        self.fitnesses_fn_is_wrapper = fitnesses_fn_is_wrapper
        self.record = []  # list of (i, cost)

    def __call__(self, i, solver, fitness_fn, fitnesses_fn, best_params_fn):
        best_params = best_params_fn(solver)
        if self.fitnesses_fn_is_wrapper:
            cost = fitnesses_fn(fitness_fn, [best_params])
        else:
            cost = fitnesses_fn([best_params])
        self.record.append(f'[{datetime.now()}]   Iteration: {i}   cost: {cost}')
        with open(self.save_fp, 'w') as fout:
            list(map(lambda r: print(r, file=fout), self.record))

    def load(self):
        with open(self.save_fp, 'r') as fin:
            self.record.extend(fin.read().split('\n'))


class StoreImageHook(Hook):
    def __init__(self, render_fn, save_fp, fps=12, save_interval=1):
        super().__init__()
        self.render_fn = render_fn
        self.save_fp = save_fp
        self.fps = fps
        self.save_interval = save_interval

        self.imgs = []

    def __call__(self, i, solver, fitness_fn, fitnesses_fn, best_params_fn):
        best_params = best_params_fn(solver)
        img = arr2img(self.render_fn(best_params))
        self.imgs.append(img)
        if i % self.save_interval == 0:
            self.save()

    def load(self):
        img_dir = f'{self.save_fp}.frames'
        for img_p in os.listdir(img_dir):
            img = Image.open(os.path.join(img_dir, img_p))
            self.imgs.append(img)

    def close(self):
        self.save()

    def save(self):
        save_as_gif(f'{self.save_fp}.gif', self.imgs, fps=self.fps)
        save_as_frames(f'{self.save_fp}.frames', self.imgs, overwrite=False)


class StoreParamHook(Hook):
    def __init__(self, save_fp):
        super().__init__()
        self.save_fp = save_fp

    def __call__(self, i, solver, fitness_fn, fitnesses_fn, best_params_fn):
        self.best_params = best_params_fn(solver)
        self.save()

    def load(self):
        self.best_params = np.load(self.save_fp)

    def close(self):
        self.save()

    def save(self):
        np.save(self.save_fp, self.best_params)


class ShowImageHook(Hook):
    def __init__(self, render_fn):
        super().__init__()
        self.render_fn = render_fn

    def __call__(self, i, solver, fitness_fn, fitnesses_fn, best_params_fn):
        if isnotebook():
            best_params = best_params_fn(solver)
            img = arr2img(self.render_fn(best_params))
            # pylint:disable=undefined-variable
            display(img)  # type: ignore

    def load(self):
        pass
