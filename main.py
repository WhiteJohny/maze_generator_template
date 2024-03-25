from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from random import randint as rnd


@dataclass
class MazeCell:
    x: int
    y: int
    component: int
    is_open: bool = field(default=False)
    walls: list = field(default_factory=list)


class MazeGenerator:
    def __init__(self) -> None:
        self.__cells = []
        self.__roots = []

    @staticmethod
    def __get_random_neighbor(x: int, y: int, n: int) -> (int, int):
        cell_neighbors = []

        if x != 0:
            cell_neighbors.append((x - 1, y))
        if y != 0:
            cell_neighbors.append((x, y - 1))
        if x != n - 1:
            cell_neighbors.append((x + 1, y))
        if y != n - 1:
            cell_neighbors.append((x, y + 1))

        random_neighbour = cell_neighbors[rnd(0, len(cell_neighbors) - 1)]

        return random_neighbour

    def __change_roots(self, new_root: int, old_root: int) -> None:
        for i in range(len(self.__roots)):
            if self.__roots[i] == old_root:
                self.__roots[i] = new_root

    def __change_cells_component(self, new_root: int, old_root: int) -> None:
        for i in range(len(self.__cells)):
            for j in range(len(self.__cells[i])):
                if self.__cells[i][j].component == old_root:
                    self.__cells[i][j].component = new_root

    def __find(self, x, y) -> int:
        return self.__cells[x][y].component

    def __union(self, x1, y1, x2, y2) -> None:
        count_first_component = np.count_nonzero(np.array(self.__roots == self.__cells[x1][y1].component))
        count_second_component = np.count_nonzero(np.array(self.__roots == self.__cells[x2][y2].component))

        if count_first_component > count_second_component:
            new_root = self.__cells[x1][y1].component
            old_root = self.__cells[x2][y2].component
        else:
            new_root = self.__cells[x2][y2].component
            old_root = self.__cells[x1][y1].component

        self.__change_cells_component(new_root, old_root)
        self.__change_roots(new_root, old_root)

    def __start_end_points(self, n: int):
        start = rnd(0, n - 1)
        end = rnd(0, n - 1)

        self.__cells[start][0].is_open = True
        self.__cells[start][0].walls[3] = False
        self.__cells[end][n - 1].is_open = True
        self.__cells[end][n - 1].walls[1] = False

        return self.__cells

    def generate_maze(self, n: int) -> list[list[MazeCell]]:
        self.__cells = [[MazeCell(i, j, i * n + j, False, [True, True, True, True]) for j in range(n)] for i in range(n)]
        self.__roots = np.array([i for i in range(n ** 2)])

        while len(np.unique(self.__roots)) != 1:
            x = rnd(0, n - 1)
            y = rnd(0, n - 1)

            random_neighbor = self.__get_random_neighbor(x, y, n)

            if self.__find(*random_neighbor) != self.__find(x, y):
                if x < random_neighbor[0]:
                    self.__cells[x][y].walls[2] = False
                    self.__cells[random_neighbor[0]][random_neighbor[1]].walls[0] = False
                elif y < random_neighbor[1]:
                    self.__cells[x][y].walls[1] = False
                    self.__cells[random_neighbor[0]][random_neighbor[1]].walls[3] = False
                elif x > random_neighbor[0]:
                    self.__cells[x][y].walls[0] = False
                    self.__cells[random_neighbor[0]][random_neighbor[1]].walls[2] = False
                elif y > random_neighbor[1]:
                    self.__cells[x][y].walls[3] = False
                    self.__cells[random_neighbor[0]][random_neighbor[1]].walls[1] = False

                self.__union(x, y, *random_neighbor)

        return self.__start_end_points(n)


def draw_maze(maze):
    y = 0

    for cells in maze:
        x = 0

        for cell in cells:
            if cell.walls[0]:
                plt.plot([x, x + LINE_WIDTH], [y, y], 'k-', lw=2)
            if cell.walls[1]:
                plt.plot([x + LINE_WIDTH, x + LINE_WIDTH], [y, y - LINE_WIDTH], 'k-', lw=2)
            if cell.walls[2]:
                plt.plot([x, x + LINE_WIDTH], [y - LINE_WIDTH, y - LINE_WIDTH], 'k-', lw=2)
            if cell.walls[3]:
                plt.plot([x, x], [y, y - LINE_WIDTH], 'k-', lw=2)

            x += LINE_WIDTH

        y -= LINE_WIDTH


def main() -> None:
    maze = MazeGenerator().generate_maze(N)

    fig = plt.figure(figsize=(10, 10))

    draw_maze(maze)

    plt.show()


if __name__ == "__main__":
    N = 30
    LINE_WIDTH = 50

    main()
