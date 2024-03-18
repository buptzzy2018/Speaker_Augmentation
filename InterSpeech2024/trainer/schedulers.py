# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math


class MarginScheduler:

    def __init__(self,
                 loss,
                 epoch_iter,
                 increase_start_epoch,
                 fix_start_epoch,
                 initial_margin,
                 final_margin,
                 update_margin,
                 increase_type='exp'):
        '''
        The margin is fixed as initial_margin before increase_start_epoch,
        between increase_start_epoch and fix_start_epoch, the margin is
        exponentially increasing from initial_margin to final_margin
        after fix_start_epoch, the margin is fixed as final_margin.
        '''
        self.loss = loss
        self.increase_start_iter = increase_start_epoch * epoch_iter
        self.fix_start_iter = fix_start_epoch * epoch_iter
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type

        self.fix_already = False
        self.current_iter = 0
        self.update_margin = update_margin 
        self.increase_iter = self.fix_start_iter - self.increase_start_iter

        self.init_margin()

    def init_margin(self):
        self.loss.update(margin=self.initial_margin)

    def get_increase_margin(self):
        initial_val = 1.0
        final_val = 1e-3

        current_iter = self.current_iter - self.increase_start_iter

        if self.increase_type == 'exp':  # exponentially increase the margin
            ratio = 1.0 - math.exp((current_iter / self.increase_iter) *
                math.log(final_val / (initial_val + 1e-6))) * initial_val
        else:  # linearly increase the margin
            ratio = 1.0 * current_iter / self.increase_iter
        return self.initial_margin + (self.final_margin - self.initial_margin) * ratio

    def step(self, current_iter=None):
        if not self.update_margin or self.fix_already:
            return

        if current_iter is not None:
            self.current_iter = current_iter

        if self.current_iter >= self.fix_start_iter:
            self.fix_already = True
            self.loss.update(margin=self.final_margin)
        elif self.current_iter >= self.increase_start_iter:
            self.loss.update(margin=self.get_increase_margin())

        self.current_iter += 1

    def get_margin(self):
        try:
            margin = self.loss.m
        except Exception:
            margin = 0.0

        return margin
