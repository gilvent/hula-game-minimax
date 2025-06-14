import math
import copy
from collections import deque
import threading
import pygame
import time
import random


# Utilities
def neighbor_hexes(hx, hy):
    DIR = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    neighbors = []

    for dx, dy in DIR:
        nx, ny = hx + dx, hy + dy
        neighbor = (nx, ny)

        neighbors.append(neighbor)

    return neighbors


class EndGameStates:
    SHORTER_LOOP = "SHORTER_LOOP"
    FIRST_FORMING_LOOP = "FIRST_FORMING_LOOP"
    EQUAL_LOOP_LESS_NEUTRAL = "EQUAL_LOOP_LESS_NEUTRAL"
    EQUAL_LOOP_EQUAL_NEUTRAL = "EQUAL_LOOP_EQUAL_NEUTRAL"
    LARGER_CONNECTED_GROUP = "LARGER_CONNECTED_GROUP"
    EQUAL_CONNECTED_GROUP = "EQUAL_CONNECTED_GROUP"


class HeuristicEvaluation:
    def __init__(self, hexagon_board, player_color):
        self.hexagon_board = hexagon_board
        self.player_color = player_color
        self.game_turn = 0

        for (hx, hy), hex_info in hexagon_board.items():
            if hex_info.get("selected", True) and (hx, hy) != (0, 0):
                self.game_turn += 1

    def __search_loop_bases(self, player_color):
        # Loop base consists of center and opponent's color connected to center.
        opponent_color = "white" if player_color == "black" else "black"
        CENTER = (0, 0)

        loop_bases = set()
        visited = set()
        reach_end = False

        # Check if loop bases connected to the end
        def is_connected_end_dfs(pos):
            hx, hy = pos

            loop_bases.add(pos)
            visited.add(pos)

            for neighbor in neighbor_hexes(hx=hx, hy=hy):
                if neighbor not in self.hexagon_board:
                    return True

                if neighbor in visited:
                    continue

                if self.hexagon_board[(neighbor)]["color"] != opponent_color:
                    continue

                if is_connected_end_dfs(neighbor):
                    return True

            return False

        reach_end = is_connected_end_dfs(CENTER)

        return loop_bases, reach_end

    def __possible_loop(self, player_color):
        loop_bases, reach_end = self.__search_loop_bases(player_color=player_color)

        if reach_end:
            return False, [], 0

        # Find possible loop around the base coordinates
        shortest_loop_path = set()
        n_moves_to_form_loop = 0
        visited = set()
        queue = deque()

        for base in loop_bases:
            queue.append(base)
            visited.add(base)

        def bfs():
            nonlocal n_moves_to_form_loop
            while queue:
                cur_x, cur_y = queue.popleft()

                for neighbor in neighbor_hexes(hx=cur_x, hy=cur_y):
                    if neighbor in visited:
                        continue

                    if neighbor not in self.hexagon_board:
                        return False, shortest_loop_path

                    color = self.hexagon_board[neighbor]["color"]
                    if color == player_color or color == "gray" or color == None:
                        shortest_loop_path.add(neighbor)
                        continue

                    visited.add(neighbor)
                    queue.append(neighbor)

            return True, shortest_loop_path

        has_loop, shortest_loop_path = bfs()

        for path in shortest_loop_path:
            if self.hexagon_board[(path)]["color"] == None:
                n_moves_to_form_loop += 1

        return has_loop, shortest_loop_path, n_moves_to_form_loop

    def __path_to_corners(self):
        CORNERS = [(5, -5), (-5, 5), (0, 5), (0, -5), (5, 0), (-5, 0)]
        opponent_color = "white" if self.player_color == "black" else "black"

        paths_to_corners = {}

        def bfs(start, goal):
            queue = deque([(start, [start])])
            visited = set([start])

            while queue:
                (hx, hy), path = queue.popleft()

                if (hx, hy) == goal:
                    return path

                for neighbor in neighbor_hexes(hx=hx, hy=hy):
                    if neighbor not in self.hexagon_board:
                        continue

                    if (
                        self.hexagon_board[neighbor]["selected"]
                        and self.hexagon_board[neighbor]["color"] == opponent_color
                    ):
                        continue

                    if neighbor in visited:
                        continue

                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

            return None

        for corner in CORNERS:
            if (
                self.hexagon_board[corner]["selected"]
                and self.hexagon_board[corner]["color"] == opponent_color
            ):
                continue

            path = bfs((0, 0), corner)
            paths_to_corners[corner] = path

        return paths_to_corners

    def evaluate_board(self):
        # Score by controlling center
        score = 0
        CENTER_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

        for hx, hy in CENTER_MOVES:
            if not self.hexagon_board[(hx, hy)]["selected"]:
                score += 0.25
                continue
            if self.hexagon_board[(hx, hy)]["color"] == self.player_color:
                score += 0.5
                continue

            score -= 0.5

        # Score by loop possibility
        has_loop, shortest_loop_path, n_moves_to_loop = self.__possible_loop(
            player_color=self.player_color
        )

        if has_loop:
            if n_moves_to_loop < 3:
                return math.inf
            elif n_moves_to_loop <= 5:
                score += 1
            elif n_moves_to_loop <= 10:
                score = 0.5
            else:
                score = 0.25

        # Score by controlling corners
        if self.game_turn < 10:
            path_to_corner_score = 2.5
        elif self.game_turn < 70:
            path_to_corner_score = 5

        for corner, paths in self.__path_to_corners().items():
            # The less required path, the better
            if paths != None:
                path_to_corner_score -= len(paths) * 0.1

        score += path_to_corner_score

        return score


class Node:
    def __init__(
        self,
        color_to_move=None,
        game_ai_color=None,
        hexagon_board=None,
        last_move=None,
        depth=0,
    ):
        self.t_value = None  # Terminal value for leaf nodes
        self.children = []
        self.best_value = None
        self.alpha = None
        self.beta = None
        self.color_to_move = color_to_move
        self.game_ai_color = game_ai_color
        self.hexagon_board = hexagon_board
        self.last_move = last_move  # Operator to get to this state
        self.depth = depth

    def add_child(self, child):
        self.children.append(child)

    def is_terminal(self):
        return self.__decide_winner()

    def __log_end_game_state(self, message):
        print(f"[Minimax] End game at depth {self.depth}:", message)
        return

    def __decide_winner(self):
        black_form_a_loop, black_best_loop, black_num_neutral_stones, black_loop_len = (
            self.__form_loop("black")
        )
        white_form_a_loop, white_best_loop, white_num_neutral_stones, white_loop_len = (
            self.__form_loop("white")
        )

        if black_form_a_loop == True and white_form_a_loop == True:
            self.__log_end_game_state(
                f"Black player - Neutral stones selected: {black_num_neutral_stones}, Loop length: {black_loop_len}"
            )
            self.__log_end_game_state(
                f"White player - Neutral stones selected: {white_num_neutral_stones}, Loop length: {white_loop_len}"
            )
            if black_loop_len < white_loop_len:
                self.__log_end_game_state("black win with shorter loop")
                self.t_value = self.__evaluate_terminal_node(
                    end_game_state=EndGameStates.SHORTER_LOOP,
                    is_win=self.game_ai_color == "black",
                )
                self.game_over = True
                return True
            elif black_loop_len > white_loop_len:
                self.__log_end_game_state("white win with shorter loop")
                self.t_value = self.__evaluate_terminal_node(
                    end_game_state=EndGameStates.SHORTER_LOOP,
                    is_win=self.game_ai_color == "white",
                )
                self.game_over = True
                return True
            else:
                if black_num_neutral_stones < white_num_neutral_stones:
                    self.__log_end_game_state(
                        "equal loop, black win with less neutral stones"
                    )
                    self.t_value = self.__evaluate_terminal_node(
                        end_game_state=EndGameStates.EQUAL_LOOP_LESS_NEUTRAL,
                        is_win=self.game_ai_color == "black",
                    )
                    self.game_over = True
                    return True
                elif black_num_neutral_stones > white_num_neutral_stones:
                    self.__log_end_game_state(
                        "equal loop, white win with less neutral stones"
                    )
                    self.t_value = self.__evaluate_terminal_node(
                        end_game_state=EndGameStates.EQUAL_LOOP_LESS_NEUTRAL,
                        is_win=self.game_ai_color == "white",
                    )
                    self.game_over = True
                    return True
                else:
                    self.__log_end_game_state(
                        "equal loop, equal neutral stones, white win"
                    )
                    self.t_value = self.__evaluate_terminal_node(
                        end_game_state=EndGameStates.EQUAL_LOOP_EQUAL_NEUTRAL,
                        is_win=self.game_ai_color == "white",
                    )
                    self.game_over = True
                    return True

        elif black_form_a_loop == True:
            self.__log_end_game_state("black win")
            self.__log_end_game_state(
                f"Black player - Neutral stones selected: {black_num_neutral_stones}, Loop length: {black_loop_len}"
            )
            self.t_value = self.__evaluate_terminal_node(
                end_game_state=EndGameStates.FIRST_FORMING_LOOP,
                is_win=self.game_ai_color == "black",
                reward_params={"move_counter": self.depth},
            )

            self.game_over = True
            return True

        elif white_form_a_loop == True:
            self.__log_end_game_state("white win!!")
            self.__log_end_game_state(
                f"White player - Neutral stones selected: {white_num_neutral_stones}, Loop length: {white_loop_len}"
            )
            self.t_value = self.__evaluate_terminal_node(
                end_game_state=EndGameStates.FIRST_FORMING_LOOP,
                is_win=self.game_ai_color == "white",
                reward_params={"move_counter": self.depth},
            )
            self.game_over = True
            return True

        elif (
            black_form_a_loop == False
            and white_form_a_loop == False
            and self.__check_all_hexes_selected() == True
        ):
            self.__bigger_group()
        else:
            return False

    def __check_all_hexes_selected(self):
        """Checks if all hexes on the board have been selected."""
        return all(hex_info["selected"] for hex_info in self.hexagon_board.values())

    def __find_shortest_loop(self, walls):

        def shortest_cycle(start, walls):

            queue = deque()
            queue.append((start, [start]))

            while queue:
                current, path = queue.popleft()
                for neighbor in neighbor_hexes(hx=current[0], hy=current[1]):

                    if neighbor == start and len(path) >= 6:

                        return True, path

                    if neighbor in walls and neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))

            return False, []

        shortest_loop = None
        shortest_length = float("inf")
        shortest_gray_count = None

        for wall in walls:

            loop, path = shortest_cycle(wall, walls)

            if loop:

                loop_length = len(path)
                gray_count = sum(
                    1 for point in path if self.hexagon_board[point]["color"] == "gray"
                )

                if loop_length < shortest_length:
                    shortest_length = loop_length
                    shortest_loop = path
                    shortest_gray_count = gray_count

                if loop_length == shortest_length:

                    if gray_count < shortest_gray_count:
                        shortest_loop = path
                        shortest_gray_count = gray_count

        if shortest_loop:
            return shortest_length, shortest_loop, shortest_gray_count
        else:
            return None, None, None

    def __form_loop(self, player_color):
        CENTER = (0, 0)

        visited = set()
        queue = deque()
        queue.append(CENTER)
        visited.add(CENTER)
        walls = set()

        def bfs():

            while queue:
                cur_x, cur_y = queue.popleft()

                for neighbor in neighbor_hexes(hx=cur_x, hy=cur_y):

                    if neighbor in visited:
                        continue

                    if neighbor not in self.hexagon_board:
                        return False, walls

                    color = self.hexagon_board[neighbor]["color"]
                    if color == player_color or color == "gray":
                        walls.add(neighbor)
                        continue

                    visited.add(neighbor)
                    queue.append(neighbor)

            return True, walls

        result, walls = bfs()

        if result == True:
            shortest_length, shortest_loop, shortest_gray_count = (
                self.__find_shortest_loop(walls)
            )
            return True, shortest_loop, shortest_gray_count, shortest_length
        else:
            return False, [], 0, 0

    def __bigger_group(self):
        """Neither player forms a loop, the player with the largest connected group of stones wins."""

        black_max_group = 0
        white_max_group = 0
        black_best_group = set()
        white_best_group = set()
        hexagon_board_copy = copy.deepcopy(self.hexagon_board)

        def dfs(x, y, player_color):
            """DFS to explore connected group and return its size."""
            stack = [(x, y)]
            local_visited = set()
            count = 0

            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in local_visited:
                    continue

                local_visited.add((cx, cy))
                info = hexagon_board_copy[(cx, cy)]
                if hexagon_board_copy[(cx, cy)]["color"] == "gray":
                    count += 2
                else:
                    count += 1

                # Only clear black/white stones, not gray
                if info["color"] in ["black", "white"]:
                    hexagon_board_copy[(cx, cy)]["color"] = None

                # 6 directions on hex grid
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in hexagon_board_copy or (nx, ny) in local_visited:
                        continue
                    neighbor_color = hexagon_board_copy[(nx, ny)]["color"]
                    if neighbor_color == player_color or neighbor_color == "gray":
                        stack.append((nx, ny))

            return count, local_visited

        for (x, y), info in list(hexagon_board_copy.items()):
            color = info["color"]
            if color in ["black", "white"]:
                group_size, group_path = dfs(x, y, color)
                if color == "black" and group_size > black_max_group:
                    black_max_group = group_size
                    black_best_group = group_path
                elif color == "white" and group_size > white_max_group:
                    white_max_group = group_size
                    white_best_group = group_path

        # print(f"Black max group size: {black_max_group}")
        # print(f"White max group size: {white_max_group}")
        if black_max_group > white_max_group:
            # print("Black wins by larger group!")
            self.t_value = self.__evaluate_terminal_node(
                end_game_state=EndGameStates.LARGER_CONNECTED_GROUP,
                is_win=self.game_ai_color == "black",
            )
            self.game_over = True
        elif white_max_group > black_max_group:
            # print("White wins by larger group!")
            self.t_value = self.__evaluate_terminal_node(
                end_game_state=EndGameStates.LARGER_CONNECTED_GROUP,
                is_win=self.game_ai_color == "white",
            )
            self.game_over = True
        else:
            # print("White wins!")
            self.t_value = self.__evaluate_terminal_node(
                end_game_state=EndGameStates.EQUAL_CONNECTED_GROUP,
                is_win=self.game_ai_color == "white",
            )
            self.game_over = True

    def __first_forming_loop_reward(self, reward_params):
        # Moves needed for the player to form the loop
        n_move_to_loop = reward_params["move_counter"] / 2

        if n_move_to_loop < 3:
            return math.inf, -math.inf
        elif n_move_to_loop < 5:
            return 7.5, -7.5
        elif n_move_to_loop < 10:
            return 5, -5
        else:
            return 3, -3

    def __evaluate_terminal_node(self, end_game_state, is_win, reward_params=None):
        """
        Set t-value as a reward when winning and punishment for losing
        """

        default_rewards = dict(
            [
                (EndGameStates.SHORTER_LOOP, (5, -5)),
                (EndGameStates.EQUAL_LOOP_LESS_NEUTRAL, (2.5, -2.5)),
                (EndGameStates.EQUAL_LOOP_EQUAL_NEUTRAL, (2.5, -2.5)),
                (EndGameStates.LARGER_CONNECTED_GROUP, (10, -10)),
                (EndGameStates.EQUAL_CONNECTED_GROUP, (8, -8)),
            ]
        )

        custom_rewards_fn = dict(
            [
                (EndGameStates.FIRST_FORMING_LOOP, self.__first_forming_loop_reward),
            ]
        )

        win_reward, lose_punishment = (
            custom_rewards_fn[end_game_state](reward_params=reward_params)
            if end_game_state in custom_rewards_fn
            else default_rewards[end_game_state]
        )
        return win_reward if is_win else lose_punishment

    def possible_moves(self):
        # Expand possible moves
        valid_move_coords = {}
        for mx, my in self.__possible_move_coordinates():
            valid_move_coords[(mx, my)] = True

        possible_moves = []
        for hx, hy in valid_move_coords.keys():
            for color in [self.color_to_move, "gray"]:
                possible_moves.append((hx, hy, color))

        return possible_moves

    def __possible_move_coordinates(self):
        CENTER = (0, 0)

        move_options = []
        visited = set()
        visited.add(CENTER)

        expandable_coordinates = []

        for (hx, hy), hex_info in self.hexagon_board.items():
            if hex_info.get("selected", True):
                expandable_coordinates.append((hx, hy))

        # Use neighbors of expandable coordinates as possible next move
        for hx, hy in expandable_coordinates:
            for neighbor in neighbor_hexes(hx=hx, hy=hy):

                if neighbor in visited:
                    continue

                if neighbor not in self.hexagon_board:
                    continue

                if self.hexagon_board[neighbor]["selected"]:
                    continue

                visited.add(neighbor)
                move_options.append(neighbor)

        return move_options

    def make_move(self, action):
        (hx, hy, color) = action
        self.hexagon_board[(hx, hy)]["selected"] = True
        self.hexagon_board[(hx, hy)]["color"] = color

        next_color_to_move = "white" if self.color_to_move == "black" else "black"

        child_node = Node(
            last_move=action,
            hexagon_board=self.hexagon_board,
            game_ai_color=self.game_ai_color,
            color_to_move=next_color_to_move,
            depth=self.depth + 1,
        )
        # self.add_child(child_node)

        return child_node

    def revert_last_move(self):
        hx, hy, color = self.last_move
        self.hexagon_board[(hx, hy)]["selected"] = False
        self.hexagon_board[(hx, hy)]["color"] = None


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

    def run_minimax(self, max_depth):
        self.__start_turn_timer()
        best_move = -1
        best_value = -math.inf

        print(f"[Minimax] Starting iterative deepening minimax (max_depth={max_depth})...")

        root_node = Node(
            hexagon_board=self.hexagon_board,
            color_to_move=self.color,
            game_ai_color=self.color,
            depth=0,
        )

        for md in range(1, max_depth + 1):
            move, value, time_cutoff = self.__root_minimax(root_node=root_node, max_depth=md)

            print(f"[Minimax] Max Depth {md}: Move={move}, Value={value}")
            
            if time_cutoff:
                break
            
            best_move = move
            best_value = value

            if best_value == math.inf:
                return best_move

        print(f"[Minimax] Selected Move:{best_move}, Value:{best_value}")


        return best_move

    def __root_minimax(self, root_node, max_depth):
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        best_move = None

        for move in root_node.possible_moves():
            hx, hy, color = move
            child_node = root_node.make_move((hx, hy, color))
            value, time_cutoff = self.minimax(
                node=child_node,
                max_depth=max_depth,
                is_maximizing=False,
                alpha=alpha,
                beta=beta,
            )

            # Revert last move because we use
            # a single instance of hexagon board for all child nodes
            child_node.revert_last_move()

            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)

            if time_cutoff:
                break

        # Get best move
        pos_x, pos_y, color = best_move
        root_node.best_value = best_value
        return (pos_x, pos_y, color), best_value, time_cutoff

    def minimax(
        self,
        node: Node,
        max_depth: int,
        is_maximizing: bool,
        alpha: int = -math.inf,
        beta: int = math.inf,
    ) -> int:
        # Use heuristic value when reaching depth limit and time runs out
        time_cutoff = self.turn_done_event.is_set()

        if node.depth == max_depth or time_cutoff:
            heuristic = HeuristicEvaluation(
                hexagon_board=self.hexagon_board, player_color=node.color_to_move
            )
            score = heuristic.evaluate_board()
            val = score if is_maximizing else -score
            return val, time_cutoff

        if node.is_terminal():
            return node.t_value, time_cutoff

        if is_maximizing:
            max_eval = -math.inf

            for hx, hy, color in node.possible_moves():
                child_node = node.make_move((hx, hy, color))
                eval_score, time_cutoff = self.minimax(
                    node=child_node,
                    max_depth=max_depth,
                    is_maximizing=False,
                    alpha=alpha,
                    beta=beta,
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                # Revert last move because we use
                # a single instance of hexagon board for all child nodes
                child_node.revert_last_move()

                # Alpha-beta pruning
                if beta <= alpha:
                    # print("[Maximizer] Pruned")
                    break

            node.best_value = max_eval
            return max_eval, time_cutoff

        else:
            min_eval = math.inf

            for hx, hy, color in node.possible_moves():

                child_node = node.make_move((hx, hy, color))

                eval_score, time_cutoff = self.minimax(
                    node=child_node,
                    max_depth=max_depth,
                    is_maximizing=True,
                    alpha=alpha,
                    beta=beta,
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                # Revert last move
                child_node.revert_last_move()

                # Alpha-beta pruning
                if beta <= alpha:
                    # print(f"[Minimizer] Prune:", action)
                    break

            node.best_value = min_eval
            return min_eval, time_cutoff


def make_move(
    hexagon_board,
    color,
):
    ai = GameAI(hexagon_board=hexagon_board, color=color)
    return ai.run_minimax(max_depth=30)
