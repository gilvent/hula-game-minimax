import math
import copy
from collections import deque
import threading
import pygame
import time
import random

class GameAI:
    def __init__(self, hexagon_board, color):
        self.color = color
        self.hexagon_board = hexagon_board

        self.turn_done_event = threading.Event()
        self.TURN_TIME_LIMIT = 28

    def __timer_thread(self, start_tick):
        """
        Countdown timer that runs in a separate thread.
        It prints the remaining time (in seconds) in the terminal once per second.
        The loop stops when either time runs out or turn_done_event is set.
        """
        while not self.turn_done_event.is_set():
            elapsed_ms = pygame.time.get_ticks() - start_tick
            remaining = self.TURN_TIME_LIMIT - (elapsed_ms / 1000)
            if remaining <= 0:
                # print("[MCTS] Time limit reached")
                self.turn_done_event.set()
                # Optionally, you could also set a global flag here to trigger an auto move in the main loop.
                break

            time.sleep(1)

    def __start_turn_timer(self):
        """
        Starts the countdown timer in a separate thread.
        Returns the start tick of the current turn.
        """
        start_tick = pygame.time.get_ticks()
        self.turn_done_event.clear()  # Clear the event at the start of the turn
        threading.Thread(
            target=self.__timer_thread, args=(start_tick,), daemon=True
        ).start()
        return start_tick

    def __decide_move_node(self):
        return


    def run_minimax(self):
        self.__start_turn_timer()

        # Build the game tree
        while not self.turn_done_event.is_set():
            break

        # Decide move
        pos_x, pos_y, color = self.__decide_move_node()

        return (pos_x, pos_y, color)


def make_move(
    hexagon_board,
    color,
):
    ai = GameAI(hexagon_board=hexagon_board, color=color)
    return ai.run_minimax()
