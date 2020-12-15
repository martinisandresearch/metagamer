#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import collections
import random

from typing import Sequence, Dict, Tuple, Optional


class QTicTacTable:
    ALPHA = 0.1

    def __init__(self):
        """
        this will contain the state
        qdict[state][action] = qvalue
        where state is a string representing the board and action is an integer
        """
        # choice to prune or build
        # we choose prune here since it's less likely to go wrong
        self.qdict: Dict[str, Dict[int, float]] = collections.defaultdict(
            lambda: {i: 0 for i in range(9)}
        )

    def __getitem__(self, item):
        state, action = item
        return self.qdict[state][action]

    def __setitem__(self, key: Tuple[str, int], value: float):
        """
        Only allowed to set a state, value pair.
        Update is as per alpha value
        """
        state, action = key
        self.qdict[state][action] = (
            self.qdict[state][action] * (1 - self.ALPHA) + value * self.ALPHA
        )

    def prune_qdict(self, state: str, valid_actions: Sequence[int]):
        """Optional function that allows us to reduce space"""
        if set(valid_actions) - self.qdict[state].keys():
            raise ValueError("We have found valid actions not in our qtable for state")
        elif self.qdict[state].keys() - set(valid_actions):
            # we can prune down since we don't need the other moves
            self.qdict[state] = {k: self.qdict[state][k] for k in valid_actions}

    def get_max(self, state: str, valid_actions: Optional[Sequence[int]] = None):
        """get maximum q score for a state over actions"""
        if valid_actions:
            self.prune_qdict(state, valid_actions)
        return max(self.qdict[state].values())

    def get_arg_max(self, state, valid_actions: Optional[Sequence[int]] = None):
        """
        which action gives max q score. If identical max q scrores exist,
           this'll choose somehing at random
        """
        mx = self.get_max(state, valid_actions)
        # get all moves that return optimal outcome
        all_arg_maxes = [k for k, v in self.qdict[state].items() if v == mx]
        return random.choice(all_arg_maxes)
