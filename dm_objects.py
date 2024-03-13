import numpy as np


def check_dominant_decision(dm_object, threshold):
    if dm_object.dm_type == 'lcp':
        return -1
    else:
        decisions_tally = np.bincount(dm_object.decision_array)
        dominant_decision = np.argmax(decisions_tally)
        consensus_number = np.max(decisions_tally)
        if consensus_number > dm_object.n * threshold:
            return dominant_decision
        else:
            return -1


def normalise(q_array):
    return q_array/np.sum(q_array)


def normalise_log(q_array):  # normalise for log values, bring mean to 0
    return q_array - np.mean(q_array)


def make_observation_from_env(old_belief, robot_pos, hypothesis_mat, belief_storage='cumulative', tile_dim=0.1, tile_array=np.array([-1])):
    """
    update single belief using observation
    :param old_belief: belief size or num_color
    :param robot_pos: [x, y]
    :param tile_dim:
    :param tile_array: shape num_color * xx * yy
    :return: new belief in the same shape as old belief
    """
    # compute robot tile pos
    tile_coo = (robot_pos / tile_dim).astype(int)
    color_prob = tile_array[:, tile_coo[0], tile_coo[1]]
    ra = np.arange(np.size(color_prob))
    color_detected = np.random.choice(ra, p=color_prob)
    color_array = np.zeros_like(color_prob)
    color_array[color_detected] = 1
    if belief_storage == 'cumulative':
        # continuous cdm, track number of colors detected
        return old_belief + color_array
    else:
        # discrete cdm, update belief mat Bayesian-ly
        return normalise(old_belief * color_array.dot(hypothesis_mat))


def make_hypothesis_mat(num_color=3, discretization_step=0.2):
    hypotheses_1d = np.arange(0, 1.000001, discretization_step)
    discretization_per_dim = np.size(hypotheses_1d)
    hypotheses_md = hypotheses_1d
    for color_ind in range(num_color-2):
        hypotheses_md = np.tile(hypotheses_md, discretization_per_dim)
        hypotheses_additional_line = np.tile(hypotheses_1d, (discretization_per_dim**(color_ind+1), 1)).T.reshape(-1)
        hypotheses_md = np.vstack((hypotheses_md, hypotheses_additional_line))
    hypotheses_mat = hypotheses_md
    if hypotheses_mat.ndim > 1:
        hypotheses_sum = np.sum(hypotheses_mat, axis=0)
        hypotheses_mat = hypotheses_mat[:, hypotheses_sum <= 1.000001]
        hypotheses_mat = np.vstack((hypotheses_mat, 1-np.sum(hypotheses_mat, axis=0)))
    else:
        hypotheses_mat = np.vstack((hypotheses_mat, 1-hypotheses_mat))
    return hypotheses_mat


class DM_object:  # discrete multi-option dm base class
    def __init__(self, num_colors, discretization_step, resample_prob=-1, comm_prob=-1, comm_dist=0.5, N=20):
        self.tile_array = np.array([])
        self.num_colors = num_colors
        self.hypotheses_mat = make_hypothesis_mat(num_colors, discretization_step)
        self.num_options = np.shape(self.hypotheses_mat)[1]
        print('Num colors ', self.num_colors)
        print('Num options ', self.num_options)
        self.decision_array = np.random.choice(range(self.num_options), N)
        self.vote_mat = np.zeros((N, self.num_options))
        #self.hypotheses = hypotheses
        self.n = N
        self.sending_message_flag_array = np.zeros(N)
        self.comm_prob = comm_prob
        self.comm_dist = comm_dist
        self.resample_prob = resample_prob
        #self.observation_class = observation_from_env()
        # self sensing
        #self.observation_timer_array = OBS_TIME_INTERVAL / TIME_STEP * np.ones(N)
        self.quality_array = np.zeros(N)
        self.quality_mat_self = np.ones((N, self.num_options))
        # dissemination
        self.neighbour_mat = np.zeros((N, N))
        self.neighbour_mat_record = np.zeros((N, N))
        self.neighbour_decision_mat = -np.ones((N, N))  # n1 robot's record of n2's decision
        self.neighbour_vote_mat = np.zeros((N, N, self.num_options))
        self.neighbour_quality_mat = -100 * np.ones((N, N))  # n1 robot's record of n2's quality, used only for DC
        self.message_count = 0

    def exploration(self, walk_state_array, coo_array):
        tile_ind_array = tuple(np.trunc(coo_array / 0.1).astype(int))
        print(self.tile_array)
        print(tile_ind_array)
        colour_array = self.tile_array[tile_ind_array]
        colour_mat = np.vstack((colour_array, 1 - colour_array)).T
        colour_mat[walk_state_array == 1, :] = 1
        mult_belief_mat = colour_mat.dot(self.hypotheses_mat.T)
        self.quality_mat_self *= mult_belief_mat
        self.quality_mat_self /= np.sum(self.quality_mat_self, axis=1)

    def compute_error_scatter(self, correct_fill_ratio):
        # mean absolute error
        predicted_fill_ratio = self.hypotheses_mat.T[tuple(self.decision_array), :]
        error_array = np.sum(np.abs(predicted_fill_ratio - correct_fill_ratio), axis=1)
        mean_error = np.mean(error_array)
        # mean distance to centroid as variation
        centroid = np.mean(predicted_fill_ratio, axis=0)
        mean_scatter = np.mean(np.sum(np.abs(centroid - predicted_fill_ratio), axis=1))
        return mean_error, mean_scatter


class DM_object_BF(DM_object):
    def __init__(self, num_colors=3, discretization_step=0.2, decay_coeff_1=0.7, decay_coeff_2=0.4, comm_prob=1, comm_dist=0.5, N=20):
        super().__init__(num_colors, discretization_step, 0, comm_prob, comm_dist, N)
        self.dm_type = 'fusion'
        self.decay_coeff_1 = decay_coeff_1
        self.decay_coeff_2 = decay_coeff_2
        self.quality_mat_neigh = np.ones((N, self.num_options))
        self.neighbour_coo_array = -np.ones((N, 2))

    def apply_decay(self, d_array, decay_coeff):
        return d_array ** decay_coeff

    def make_decision(self, walk_state_array, coo_array):
        sending_message_prob_array = np.random.random(self.n)
        for i in range(self.n):
            self.neighbour_coo_array[i, :] = -1
            # exploration
            if walk_state_array[i] == 0:
                self.quality_mat_self[i, :] = make_observation_from_env(self.quality_mat_self[i, :],
                                                                        coo_array[i, :], self.hypotheses_mat,
                                                                        belief_storage='bayesian',
                                                                        tile_array=self.tile_array)
                self.quality_mat_self[self.quality_mat_self < 0.001 / self.num_options] = 0.001 / self.num_options
            self.quality_array[i] = self.quality_mat_self[i, self.decision_array[i]]
            # dissemination
            dist_array = np.sqrt(np.sum((coo_array - coo_array[i, :]) ** 2, axis=1))
            dist_array[i] = 100
            potential_neighbour_list = np.array(range(self.n))[
                np.logical_and(dist_array <= self.comm_dist, sending_message_prob_array <= self.comm_prob)]
            if potential_neighbour_list.size > 0:
                new_neighbour_tag = np.random.choice(potential_neighbour_list)
                self.neighbour_coo_array[i, :] = coo_array[new_neighbour_tag, :]
                self.message_count += 1
                self.quality_mat_neigh[i, :] = normalise_log(#0.001/self.num_options +  # current result 0.01
                    self.quality_mat_neigh[i, :] * self.decay_coeff_1 +
                    np.log(self.quality_mat_self[new_neighbour_tag, :]) +
                    self.quality_mat_neigh[new_neighbour_tag, :] * self.decay_coeff_2)
                #if np.any(np.isnan(self.quality_mat_neigh[i, :])):
                #    self.quality_mat_neigh[i, :] = 1
        s = self.quality_mat_self.sum(axis=1)
        d = np.argmax(np.log(self.quality_mat_self) + self.quality_mat_neigh, axis=1)
        self.decision_array[s != self.num_options] = d[s != self.num_options]


class DM_object_RV(DM_object):
    def __init__(self, num_colors=3, discretization_step=0.2, evidence_rate=0.5, ballot_size=5, comm_prob=1,
                 comm_dist=0.5, N=20):
        super().__init__(num_colors, discretization_step, 0, comm_prob, comm_dist, N)
        self.dm_type = 'rv'
        self.ballot_array = -np.ones((N, self.num_options))
        self.neighbour_coo_array = -np.ones((N, 2))
        self.evidence_rate = evidence_rate
        self.ballot_size = int(ballot_size)

    def update_ranking(self, ranking_a, ranking_b):
        # combine two rankings, empty entries are represented by -1 and ranked last
        score_array_a = -np.ones(self.num_options)
        score_array_b = -np.ones(self.num_options)
        for i in range(self.num_options):
            if ranking_a[i] >= 0:
                score_array_a[int(ranking_a[i])] = i
            if ranking_b[i] >= 0:
                score_array_b[int(ranking_b[i])] = i
        score_array_a[score_array_a < 0] = np.max(score_array_a) + 1
        score_array_b[score_array_b < 0] = np.max(score_array_b) + 1
        sum_score = score_array_a + score_array_b
        sum_score_ranked = np.unique(np.sort(sum_score))
        ra = np.arange(self.num_options)
        op_ranking = np.array([])
        for score in sum_score_ranked:
            ind_added = ra[sum_score == score]
            if np.size(ind_added) > 1:
                ind_added = np.random.permutation(ind_added)
            op_ranking = np.hstack((op_ranking, ind_added))
        return op_ranking

    def make_ranking(self, i):
        prob_array = self.quality_mat_self[i, :]
        prob_array_sorted = np.unique(np.sort(-prob_array))
        ra = np.arange(self.num_options)
        op_ranking = np.array([])
        for neg_prob in prob_array_sorted:
            ind_added = ra[prob_array == -neg_prob]
            if np.size(ind_added) > 1:
                ind_added = np.random.permutation(ind_added)
            op_ranking = np.hstack((op_ranking, ind_added))
        return op_ranking

    def make_decision(self, walk_state_array, coo_array):
        sending_message_prob_array = np.random.random(self.n)
        for i in range(self.n):
            self.neighbour_coo_array[i, :] = -1
            # exploration
            if walk_state_array[i] == 0:
                self.quality_mat_self[i, :] = make_observation_from_env(self.quality_mat_self[i, :],
                                                                        coo_array[i, :], self.hypotheses_mat,
                                                                        belief_storage='bayesian',
                                                                        tile_array=self.tile_array)
                rnd = np.random.random()
                if rnd < self.evidence_rate:
                    # update ranking with observation
                    ranking_from_ob = self.make_ranking(i)
                    self.ballot_array[i, :] = self.update_ranking(self.ballot_array[i, :], ranking_from_ob)
                    self.decision_array[i] = self.ballot_array[i, 0]
            # dissemination
            dist_array = np.sqrt(np.sum((coo_array - coo_array[i, :]) ** 2, axis=1))
            dist_array[i] = 100
            potential_neighbour_list = np.array(range(self.n))[
                np.logical_and(np.logical_and(dist_array <= self.comm_dist,
                                              sending_message_prob_array <= self.comm_prob),
                               np.min(self.ballot_array, axis=1) >= 0)]
            if potential_neighbour_list.size > 0:
                new_neighbour_tag = np.random.choice(potential_neighbour_list)
                self.neighbour_coo_array[i, :] = coo_array[new_neighbour_tag, :]
                self.message_count += 1
                incoming_ballot = self.ballot_array[new_neighbour_tag, :]
                if np.size(incoming_ballot) > self.ballot_size:
                    incoming_ballot[self.ballot_size:] = -1
                    self.ballot_array[i, :] = self.update_ranking(self.ballot_array[i, :], incoming_ballot)
                    self.decision_array[i] = self.ballot_array[i, 0]


class DM_object_LCP:
    """
    Linear Consensus Protocol
    Unmodulated message frequency, same format as quality
    """
    def __init__(self, dm_cycle=10, comm_prob=1, num_color=3, dm_delay=10, comm_dist=0.5, N=20):
        self.dm_type = 'lcp'
        self.dm_cycle = dm_cycle
        self.observation_counter_mat = np.zeros((N, num_color))
        self.num_color = num_color
        self.comm_prob = comm_prob
        self.comm_dist = comm_dist
        #self.observation_class = observation_from_env()
        self.tile_array = np.array([])
        self.n = N
        self.freq_array_ob = np.zeros((N, num_color))
        self.freq_array_dm = -np.ones((N, num_color))
        self.dm_cache_mat = -np.ones((int(N), int(dm_cycle), int(num_color)))
        self.message_count = 0
        self.dm_delay = dm_delay
        self.neighbour_coo_array = -np.ones((N, 2))

    def make_decision(self, walk_state_array, coo_array):
        sending_message_prob_array = np.random.random(self.n)
        for i in range(self.n):
            self.neighbour_coo_array[i, :] = -1
            # exploration
            if walk_state_array[i] == 0:
                self.observation_counter_mat[i, :] = make_observation_from_env(self.observation_counter_mat[i, :],
                                                                               coo_array[i, :], None,
                                                                               tile_array=self.tile_array)
                self.freq_array_ob[i, :] = normalise(self.observation_counter_mat[i, :])
                if np.min(self.freq_array_dm[i, :]) < 0 and np.sum(self.observation_counter_mat[i, :]) > self.dm_delay:
                    self.freq_array_dm[i, :] = self.freq_array_ob[i, :]
            # dissemination
            dist_array = np.sqrt(np.sum((coo_array - coo_array[i, :]) ** 2, axis=1))
            dist_array[i] = 100
            potential_neighbour_list = np.array(range(self.n))[
                np.logical_and(np.logical_and(dist_array <= self.comm_dist,
                                              sending_message_prob_array <= self.comm_prob),
                               np.min(self.freq_array_dm, axis=1) >= 0)]
            if potential_neighbour_list.size > 0:
                new_neighbour_tag = np.random.choice(potential_neighbour_list)
                self.neighbour_coo_array[i, :] = coo_array[new_neighbour_tag, :]
                dm_cache_slice = self.dm_cache_mat[i, :, 0]
                self.dm_cache_mat[i, np.size(dm_cache_slice[dm_cache_slice >= 0]), :] = self.freq_array_dm[new_neighbour_tag, :]
                self.message_count += 1
                if np.size(dm_cache_slice[dm_cache_slice >= 0]) >= self.dm_cycle-1:
                    # update estimate with mean value
                    opinion_mat = np.vstack((self.freq_array_ob[i, :],
                                             self.freq_array_dm[i, :],
                                             self.dm_cache_mat[i, :, :]))
                    opinion_mat = opinion_mat / np.tile(np.sum(opinion_mat, axis=1), (self.num_color, 1)).T
                    self.freq_array_dm[i, :] = np.mean(opinion_mat, axis=0)/np.sum(np.mean(opinion_mat, axis=0))
                    self.dm_cache_mat[i, :, :] = -1

    def compute_error_scatter(self, correct_fill_ratio):
        if np.min(self.freq_array_dm) < 0:
            return -1, -1
        else:
            # mean absolute error
            predicted_fill_ratio = np.copy(self.freq_array_dm)
            error_array = np.sum(np.abs(predicted_fill_ratio - correct_fill_ratio), axis=1)
            total_error = np.mean(error_array)
            # mean distance to centroid as variation
            centroid = np.mean(predicted_fill_ratio, axis=0)
            mean_scatter = np.mean(np.sqrt(np.sum((centroid - predicted_fill_ratio)**2, axis=1)))
            return total_error, mean_scatter



