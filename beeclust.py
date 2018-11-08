import numpy as np
import random
from queue import *
import numbers

class BeeClust:
    def __init__(self, map, p_changedir=0.2, p_wall=0.8, p_meet=0.8, k_temp=0.9,
                 k_stay=50, T_ideal=35, T_heater=40, T_cooler=5, T_env=22, min_wait=2):
        self.p_changedir = p_changedir
        self.p_wall = p_wall
        self.p_meet = p_meet
        self.k_stay = k_stay
        self.min_wait = min_wait
        self.T_ideal = T_ideal
        self.T_heater = T_heater
        self.T_cooler = T_cooler
        self.T_env = T_env
        self.k_temp = k_temp
        try:
            self.map = np.array(map)
        except ValueError as err:
            raise TypeError('map')
        try:
            self.heatmap = np.array(map, dtype=np.float64)
        except ValueError as err:
            raise TypeError('heatmap')

        if not len(np.array(map).shape) == 2:
            raise ValueError('shape')

        self.test_init()
        self.recalculate_heat()

    def test_init(self):
        if not isinstance(self.p_changedir, numbers.Number):
            raise TypeError('p_changedir')
        if not isinstance(self.p_wall, numbers.Number):
            raise TypeError('p_wall')
        if not isinstance(self.p_meet, numbers.Number):
            raise TypeError('p_meet')
        if not isinstance(self.k_stay, numbers.Number):
            raise TypeError('k_stay')
        if not isinstance(self.min_wait, numbers.Number):
            raise TypeError('min_wait')
        if not isinstance(self.T_ideal, numbers.Number):
            raise TypeError('T_ideal')
        if not isinstance(self.T_heater, numbers.Number):
            raise TypeError('T_heater')
        if not isinstance(self.T_cooler, numbers.Number):
            raise TypeError('T_cooler')
        if not isinstance(self.T_env, numbers.Number):
            raise TypeError('T_env')
        if not isinstance(self.k_temp, numbers.Number):
            raise TypeError('k_temp')

        if self.k_temp < 0 or self.min_wait < 0 or self.k_stay < 0 or self.p_meet < 0 or self.p_wall < 0 or self.p_changedir<0:
            raise ValueError('negative')

        if self.p_changedir>1 or self.p_wall>1 or self.p_meet>1:
            raise ValueError('probability, var>1')

        if not (self.T_cooler<=self.T_env <= self.T_heater):
            raise ValueError('T_env or  T_cooler or T_heater error')
    @property
    def swarms(self):
        bees = self.bees
        swarms = []

        for bee in bees:
            found = False
            for swarm in swarms:
                if bee in swarm:
                    found = True
                    break

            if not found:
                new_swarm = []
                self.find(bee, new_swarm)
                swarms.append(new_swarm)
        return swarms

    def find(self, old_bee, swarm):
        swarm.append(old_bee)

        for bee in self.bees:
            if bee[0] == old_bee[0] and abs(old_bee[1] - bee[1]) == 1 and not bee in swarm:
                self.find(bee, swarm)

            elif bee[1] == old_bee[1] and abs(old_bee[0] - bee[0]) == 1 and not bee in swarm:
                self.find(bee, swarm)

    @property
    def score(self):
        self.recalculate_heat()
        number_of_bees = 0
        temperature_sum = 0.0
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] in [1, 2, 3, 4] or self.map[y, x] < 0:
                    number_of_bees += 1
                    temperature_sum += self.heatmap[y, x]

        if number_of_bees == 0:
            return 0
        return temperature_sum / number_of_bees

    @property
    def bees(self):
        bees = []
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] in [1, 2, 3, 4] or self.map[y, x] < 0:
                    bees.append((y, x))
        return bees

    def tick(self):
        moved = 0
        old_map = self.map.copy()
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if old_map[y, x] < 0 or old_map[y, x] in [1, 2, 3, 4]:
                    if self.stay_or_move(x, y):
                        moved += 1
        return moved

    def forget(self):
        for y in range(self.heatmap.shape[0]):
            for x in range(self.heatmap.shape[1]):
                if not self.map[y, x] in [5, 6, 7, 0]:
                    self.map[y, x] = -1

    def recalculate_heat(self):
        for y in range(self.heatmap.shape[0]):
            for x in range(self.heatmap.shape[1]):
                if self.map[y, x] == 5:
                    self.heatmap[y, x] = 0
                elif self.map[y, x] == 6:
                    self.heatmap[y, x] = self.T_heater
                elif self.map[y, x] == 7:
                    self.heatmap[y, x] = self.T_cooler
                else:
                    self.heatmap[y, x] = float(self.get_heat(x, y))

    def check_if_goal_exist(self, goal):
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] == goal:
                    return True

        return False

    def BFS(self, x, y, goal):

        if not self.check_if_goal_exist(goal):
            return -1

        queue = Queue()
        queue.put((x, y))
        seen = {(x, y)}
        pairs = []
        while not queue.empty():
            curr_x, curr_y = queue.get()
            nei = self.get_neighbours(curr_x, curr_y, goal)
            if type(nei) is bool:
                if curr_x == x and curr_y == y:
                    return 1
                return self.get_path_len(pairs, (x, y), (curr_x, curr_y)) + 1
            for i in nei:
                if not i in seen:
                    queue.put(i)
                    seen.add(i)
                    pairs.append([(curr_x, curr_y), i])
        return -1

    def get_path_len(self, path, start, end):
        len = 0
        to = end
        while True:
            for i in path:
                if i[1] == to:
                    len += 1
                    if i[0] == start:
                        return len
                    to = i[0]

    def get_neighbours(self, curr_x, curr_y, goal):
        out = []

        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y, x] in [5, 6, 7] and not self.map[y, x] == goal:
                    continue
                if abs(x - curr_x) <= 1 and abs(y - curr_y) <= 1:
                    if self.map[y, x] == goal:
                        return True
                    if x == curr_x and y == curr_y:
                        continue
                    out.append((x, y))

        return out

    def get_heat(self, x, y):
        dist_heater = self.BFS(x, y, 6)
        dist_cooler = self.BFS(x, y, 7)
        heating = 0
        cooling = 0
        if dist_heater == -1:
            heating = 0
        else:
            heating = float((1 / float(dist_heater)) * (float(self.T_heater) - float(self.T_env)))

        if dist_cooler == -1:
            cooling = 0
        else:
            cooling = float((1 / float(dist_cooler)) * (float(self.T_env) - float(self.T_cooler)))

        return float(float(self.T_env) + float(self.k_temp) * (float(max(heating, 0)) - float(max(cooling, 0))))

    def stay_or_move(self, x, y):
        curr_dir = self.map[y, x]
        if curr_dir < -1:
            self.map[y, x] = curr_dir + 1
            return False
        else:
            self.map[y, x] = self.get_direction(curr_dir)
            if curr_dir == -1:
                return False

            try:
                if self.move(x, y):
                    return True
                else:
                    return False
            except IndexError:
                self.wall_collision(x, y)
                return False

    def stop_timer(self, x, y):
        T_local = self.heatmap[y, x]
        time_to_wait = int(self.k_stay / (1 + abs(self.T_ideal - T_local)))

        return -max(time_to_wait, self.min_wait)

    def wall_collision(self, x, y):
        rand = random.randint(0, 100)

        # stop
        if self.p_wall * 100 > rand:
            pass
            self.map[y, x] = self.stop_timer(x, y)
        # turn
        else:
            self.map[y, x] = self.turn_bee(x, y)

    def turn_bee(self, x, y):
        curr_dir = self.map[y, x]
        if curr_dir == 1:
            return 3
        elif curr_dir == 2:
            return 4
        elif curr_dir == 3:
            return 1
        elif curr_dir == 4:
            return 2

    def get_direction(self, curr_dir):
        dirs = [1, 2, 3, 4]
        rand = random.randint(0, 100)

        if (not curr_dir == -1) and (rand >= (self.p_changedir * 100)):
            return curr_dir

        if not curr_dir == -1:
            dirs.remove(curr_dir)

        new_dir_index = random.randint(0, len(dirs) - 1)
        return dirs[new_dir_index]

    def bee_collision(self, x, y):
        rand = random.randint(0, 100)
        if rand < self.p_meet * 100:
            self.map[y, x] = self.stop_timer(x, y)

    def solve_collision(self, x, y, curr_dir):
        if curr_dir == 1:
            if self.map[y - 1, x] in [5, 6, 7]:
                self.wall_collision(x, y)
            else:
                self.bee_collision(x, y)
        elif curr_dir == 2:
            if self.map[y, x + 1] in [5, 6, 7]:
                self.wall_collision(x, y)
            else:

                self.bee_collision(x, y)
        elif curr_dir == 3:
            if self.map[y + 1, x] in [5, 6, 7]:
                self.wall_collision(x, y)
            else:
                self.bee_collision(x, y)
        elif curr_dir == 4:
            if self.map[y, x - 1] in [5, 6, 7]:
                self.wall_collision(x, y)
            else:
                self.bee_collision(x, y)
        else:
            exit(1)

    def move(self, x, y):
        curr_dir = self.map[y, x]

        if curr_dir == 1:
            if y == 0:
                self.wall_collision(x, y)
                return False
            elif not self.map[y - 1, x] == 0:
                self.solve_collision(x, y, curr_dir)
                return False
            else:
                self.map[y - 1, x] = curr_dir
                self.map[y, x] = 0

        elif curr_dir == 2:
            if x + 1 == self.map.shape[1]:
                self.wall_collision(x, y)
                return False
            elif not self.map[y, x + 1] == 0:
                self.solve_collision(x, y, curr_dir)
                return False
            else:
                self.map[y, x + 1] = curr_dir
                self.map[y, x] = 0

        elif curr_dir == 3:
            if y + 1 == self.map.shape[0]:
                self.wall_collision(x, y)
                return False

            elif not self.map[y + 1, x] == 0:
                self.solve_collision(x, y, curr_dir)
                return False
            else:
                self.map[y + 1, x] = curr_dir
                self.map[y, x] = 0

        elif curr_dir == 4:
            if x == 0:
                self.wall_collision(x, y)
                return False
            elif not self.map[y, x - 1] == 0:
                self.solve_collision(x, y, curr_dir)
                return False
            else:
                self.map[y, x - 1] = curr_dir
                self.map[y, x] = 0

        else:
            print('WRONG DIR - ERROR - ' + str(curr_dir))
            exit(1)
        return True