import random
import numpy as np
import math
import matplotlib.pyplot as plt
import dm_objects


TIME_STEP = 0.01
TSPF = 100
SAMPLE_NUM = 10


def generate_random_w_mean(mean, size=1, maxi=1, mini=0):
    bs_array = np.random.uniform(size=size)
    t_array = np.zeros_like(bs_array)
    t_array[bs_array >= mean] = np.random.uniform(low=mini, high=mean, size=np.size(bs_array[bs_array >= mean]))
    t_array[bs_array < mean] = np.random.uniform(low=mean, high=maxi, size=np.size(bs_array[bs_array < mean]))
    return t_array


class epuck:
    def __init__(self):
        # constants
        self.vl = 0.16
        self.va = 0.75
        self.r = 0.035
        #self.comm_dist = 0.5  # 0.5
        # movement
        self.dir = random.random() * 2 * math.pi
        self.dir_v = np.array([math.sin(self.dir), math.cos(self.dir)])
        self.turn_dir = int(random.random() * 2) * 2 - 1
        self.walk_state = int(random.random() * 2)
        self.walk_timer = 0


class arena:
    def __init__(self, fill_ratio, dm_object, N=20, dim=np.array([2, 2]), pattern='random', dist_decay_coeff=1, axis=None):
        # initialise arena
        self.robot_spec = epuck()
        self.length = int(dim[0]/0.1)
        self.width = int(dim[1]/0.1)
        self.fill_ratio = fill_ratio/np.sum(fill_ratio)
        self.num_color = np.size(fill_ratio)
        self.tile_array = self.generate_pattern(self.length, self.width, pattern=pattern, dist_decay_coeff=dist_decay_coeff)
        # initialise agents
        self.coo_array = np.array([]).reshape([0, 2])
        self.N = N
        self.n = float(N)
        self.dim = dim
        for i in range(N):
            coo = np.array([random.random(), random.random()] * self.dim)
            #self.robots.append(epuck())
            while self.collision_detect(self.coo_array, coo):
                coo = np.array([random.random(), random.random()] * self.dim)
                #print('new position', i, coo)
            self.coo_array = np.vstack((self.coo_array, coo))
        self.dm_object = dm_object
        self.dm_object.tile_array = self.tile_array
        self.dir_array = np.random.rand(N) * 2 * math.pi
        self.dir_v_array = np.vstack((np.sin(self.dir_array), np.cos(self.dir_array))).T
        self.turn_dir_array = np.floor(np.random.rand(N) * 2) * 2 - 1  # randomly -1 or +1
        self.walk_state_array = np.floor(np.random.rand(N) * 2)  # randomly 0 or 1
        self.walk_timer_array = np.zeros(N)

    def check_fill_ratio_deviation(self, tiles_mat):
        # assume tiles are normalized
        current_fill_ratio = np.mean(tiles_mat, axis=(1, 2))
        diff_fill_ratio = self.fill_ratio - current_fill_ratio
        return diff_fill_ratio

    def generate_pattern(self, length, width, pattern='random', dist_decay_coeff=1):
        print('fill ratio ', self.fill_ratio)
        # generate arena
        if pattern == 'random':
            tiles = np.zeros((self.num_color, length, width))
            # proportion_left_mat = np.ones((length, width))
            for color_id in range(self.num_color):
                prop_slice = generate_random_w_mean(mean=self.fill_ratio[color_id], size=length * width).reshape((length, width))
                # c = np.mean(prop_slice) / self.fill_ratio_real[color_id]
                # prop_slice = prop_slice / c
                # proportion_left_mat = proportion_left_mat - prop_slice
                tiles[color_id, :, :] = prop_slice
            # normalize
            tiles = tiles / np.sum(tiles, axis=0)
            print('result proportion ', np.mean(tiles, axis=(1, 2)))
            # print(tiles)
            print('sum of all c, mean ', np.mean(np.sum(tiles, axis=0)), ' std ', np.std(np.sum(tiles, axis=0)))
            return tiles
        else:
            # Block pattern
            tiles = np.zeros((self.num_color, length, width))
            tiles[np.argmin(self.fill_ratio), :, :] = 1
            tiles = tiles / np.sum(tiles, axis=0)
            diff_fill_ratio = self.check_fill_ratio_deviation(tiles)
            while np.max(np.abs(diff_fill_ratio)) > 0.01:
                color_ind = np.argmax(diff_fill_ratio)
                diff_ = np.max(diff_fill_ratio)
                center_pos = [int(np.random.uniform()*length), int(np.random.uniform()*width)]
                length_coo_mat = np.tile(np.arange(length), (width, 1)).T
                width_coo_mat = np.tile(np.arange(width), (length, 1))
                length_dist_mat = center_pos[0] - length_coo_mat
                width_dist_mat = center_pos[1] - width_coo_mat
                total_dist_mat = np.sqrt(length_dist_mat**2 + width_dist_mat**2)
                weighting_mat = diff_ * 10 * np.exp(-total_dist_mat * dist_decay_coeff)
                tiles[color_ind, :, :] += weighting_mat
                # re-normalize and prepare for next loop
                tiles = tiles / np.sum(tiles, axis=0)
                diff_fill_ratio = self.check_fill_ratio_deviation(tiles)
            print('result proportion ', np.mean(tiles, axis=(1, 2)))
            # print(tiles)
            print('sum of all c, mean ', np.mean(np.sum(tiles, axis=0)), ' std ', np.std(np.sum(tiles, axis=0)))
            return tiles

    def check_neighbouring_tiles(self, tiles, coo, value):
        appended_tile_array = np.ones((self.length + 2, self.width + 2)) * (1 - value)
        appended_tile_array[1:self.length + 1, 1:self.width + 1] = tiles
        vicinity_block = np.array([appended_tile_array[coo[0]+1, coo[1]+1], appended_tile_array[coo[0], coo[1]+1],
                                   appended_tile_array[coo[0]+1, coo[1]], appended_tile_array[coo[0]+1, coo[1]+2],
                                   appended_tile_array[coo[0]+2, coo[1]+1]])
        if np.any(vicinity_block == value):
            return True
        else:
            return False

    def oob(self, coo):
        # out of bound
        if self.robot_spec.r < coo[0] < self.dim[0] - self.robot_spec.r \
                and self.robot_spec.r < coo[1] < self.dim[1] - self.robot_spec.r:
            return False
        else:
            return True

    def collision_detect(self, coo_array, new_coo):
        # check if new_coo clip with any old coo, or oob
        if self.oob(new_coo):
            return True
        elif coo_array.shape[0] == 0:
            return False
        else:
            dist_array = np.sqrt(np.sum((coo_array - new_coo) ** 2, axis=1))
            if np.min(dist_array) < 2 * self.robot_spec.r:
                #print(dist_array)
                #print('collision ')
                return True
            else:
                return False

    def collision_detect_2(self):  # output array indicating collision
        new_coo_array = self.coo_array + self.dir_v_array * self.robot_spec.vl * TIME_STEP * 10
        new_coo_mat = np.tile(new_coo_array, (int(self.n), 1, 1)).transpose((1, 0, 2))
        coo_mat = np.tile(self.coo_array, (int(self.n), 1, 1))
        dist_mat = np.sqrt(np.sum((new_coo_mat - coo_mat) ** 2, axis=2))
        dist_mat += np.identity(dist_mat.shape[0]) * 100
        collision_mat = np.zeros_like(dist_mat)
        collision_mat[dist_mat < 2 * self.robot_spec.r] = 1
        collision_array = np.sum(collision_mat, axis=1)
        collision_array[collision_array > 1] = 1
        oob_arr = self.oob_array(new_coo_array)
        collision_array[oob_arr] = 1
        return collision_array

    def oob_array(self, coo_array):
        horizontal = np.logical_or(coo_array[:, 0] < self.robot_spec.r,
                                   coo_array[:, 0] > self.dim[0] - self.robot_spec.r)
        vertical = np.logical_or(coo_array[:, 1] < self.robot_spec.r,
                                 coo_array[:, 1] > self.dim[1] - self.robot_spec.r)
        return np.logical_or(horizontal, vertical)

    def random_walk_mat(self):
        self.walk_timer_array -= 1
        # for state=0
        collision_array = self.collision_detect_2()
        move_array = collision_array + self.walk_state_array
        self.coo_array[move_array == 0, :] += self.dir_v_array[move_array == 0, :] * self.robot_spec.vl * TIME_STEP
        time_out_array_0 = np.logical_and(self.walk_timer_array < 0, self.walk_state_array == 0)
        switch_array_0 = np.logical_or(time_out_array_0, np.logical_and(collision_array == 1,
                                                                        self.walk_state_array == 0))
        switch_array_0 = np.logical_or(switch_array_0, np.logical_and(self.walk_timer_array < 0, collision_array == 1))
        self.walk_state_array[switch_array_0] = 1
        self.walk_timer_array[switch_array_0] = np.random.rand(
            self.walk_state_array[switch_array_0].size) * 4.5 / TIME_STEP
        self.turn_dir_array[switch_array_0] = np.floor(
            np.random.rand(self.walk_state_array[switch_array_0].size) * 2) * 2 - 1
        # for state=1
        self.dir_array[self.walk_state_array == 1] += \
            self.turn_dir_array[self.walk_state_array == 1] * self.robot_spec.va * TIME_STEP
        self.dir_v_array = np.vstack((np.sin(self.dir_array), np.cos(self.dir_array))).T
        switch_array_1 = np.logical_and(self.walk_state_array == 1, self.walk_timer_array < 0)
        self.walk_state_array[switch_array_1] = 0
        self.walk_timer_array[switch_array_1] = np.random.rand(
            self.walk_state_array[switch_array_1].size) * 4.5 / TIME_STEP

    def plot_arena(self, t_step, axis):
        if t_step % TSPF == 0:
            axis[0, 0].cla()
            axis[0, 1].cla()
            axis[1, 0].cla()
            axis[1, 1].cla()

            axis[0, 0].set_title('time '+str(t_step/100))
            # Draw arena
            color_sequence = "bgrcmykw"
            for color_id in range(self.num_color):
                for i in range(self.width):
                    for j in range(self.length):
                        axis[0, 0].fill_between([i * 0.1, (i + 1) * 0.1], [j * 0.1, j * 0.1], [(j + 1) * 0.1, (j + 1) * 0.1],
                                                facecolor=color_sequence[color_id], alpha=self.tile_array[color_id, i, j])
            # Draw agents
            for i in range(np.shape(self.coo_array)[0]):
                circle = plt.Circle((self.coo_array[i, 0], self.coo_array[i, 1]), self.robot_spec.r, color='g', fill=False)
                axis[0, 0].add_artist(circle)
                axis[0, 0].plot(np.array([self.coo_array[i, 0], self.coo_array[i, 0]+self.dir_v_array[i, 0]*0.05]), np.array([self.coo_array[i, 1], self.coo_array[i, 1]+self.dir_v_array[i, 1]*0.05]),'k')
                if self.dm_object.neighbour_coo_array[i, 0] >= 0:
                    axis[0, 0].plot(np.array([self.coo_array[i, 0], self.dm_object.neighbour_coo_array[i, 0]]), np.array([self.coo_array[i, 1], self.dm_object.neighbour_coo_array[i, 1]]), 'c')
            axis[0, 0].plot(self.coo_array[:, 0], self.coo_array[:, 1], 'ro', markersize=3)
            axis[0, 0].set(xlim=(0, self.dim[0]), ylim=(0, self.dim[1]))
            axis[0, 0].set_aspect('equal', adjustable='box')

            # strategy specific monitoring, comment out if error
            if self.dm_object.dm_type == 'lcp':
                for i in range(self.N):
                    for color_id in range(self.num_color):
                        axis[0, 1].scatter(np.arange(self.num_color), self.fill_ratio, 'g+')
                        axis[0, 1].scatter(color_id, self.dm_object.freq_array_ob[i, color_id], c='r', alpha=0.5)
                        axis[0, 1].scatter(color_id, self.dm_object.freq_array_dm[i, color_id], c='b', alpha=0.5)
                axis[0, 1].set(xlim=(-1, self.num_color+1))
            elif self.dm_object.dm_type == 'fusion':
                axis[0, 1].plot(5 * self.N * np.sum((self.dm_object.hypotheses_mat-np.tile(self.fill_ratio, (self.dm_object.num_options, 1)).T)**2, axis=0), np.arange(self.dm_object.num_options))
                axis[0, 1].scatter(np.arange(self.N), self.dm_object.decision_array)
                axis[0, 1].set(ylim=(0, self.dm_object.num_options))
                axis[0, 1].set(xlim=(-1, self.N))
                for i in range(1):
                    axis[1, 0].plot(self.dm_object.quality_mat_self[i, :])
                    axis[1, 1].plot(self.dm_object.quality_mat_neigh[i, :])
            elif self.dm_object.dm_type == 'rv':
                axis[0, 1].plot(5 * self.N * np.sum((self.dm_object.hypotheses_mat-np.tile(self.fill_ratio, (self.dm_object.num_options, 1)).T)**2, axis=0), np.arange(self.dm_object.num_options))
                axis[0, 1].scatter(np.arange(self.N), self.dm_object.decision_array)
                axis[0, 1].set(ylim=(0, self.dm_object.num_options))
                axis[0, 1].set(xlim=(-1, self.N))
                for i in range(self.N):
                    axis[1, 0].plot(5 * self.N * np.sum((self.dm_object.hypotheses_mat - np.tile(self.fill_ratio, (self.dm_object.num_options, 1)).T) ** 2, axis=0), np.arange(self.dm_object.num_options))
                    axis[1, 0].scatter(np.arange(self.dm_object.num_options), self.dm_object.ballot_array[i, :], alpha=0.5)
                    axis[1, 0].set(ylim=(0, self.dm_object.num_options))
                    axis[1, 0].set(xlim=(-1, self.dm_object.num_options))
            plt.draw()
            plt.pause(0.001)
        else:
            pass



