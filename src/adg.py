import bitarray
from collections import defaultdict
import numpy as np
import time
import torch as th
from src.utils import is_sent_finish, num_same_from_beg, int2bits, bits2int, DRBG, bin_sort, kl2, entropy

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
	res = 0
	for i, bit in enumerate(bits):
		res += bit*(2**i)
	return res


def int2bits(inp, num_bits):
	if num_bits == 0:
		return []
	strlist = ('{0:0%db}'%num_bits).format(inp)
	return [int(strval) for strval in reversed(strlist)]


def near(alist, anum):
	up = len(alist) - 1
	if up == 0:
		return 0
	bottom = 0
	while up - bottom > 1:
		index = int((up + bottom)/2)
		if alist[index] < anum:
			up = index
		elif alist[index] > anum:
			bottom = index
		else:
			return index
	if up - bottom == 1:
		if alist[bottom] - anum < anum - up:
			index = bottom
		else:
			index = up
	return index

class ADGEncoder:

    def __init__(self, medium, **kwargs):
        self.medium = medium
        self.context = kwargs.get("context", None)
        self.finish_sent = kwargs.get("finish_sent")
        self.precision = kwargs.get("precision")
        self.is_sort = kwargs.get("is_sort")
        self.clean_up_output = kwargs.get("clean_up_output", False)
        self.input_key = kwargs.get("input_key")
        self.sample_seed_prefix = kwargs.get("sample_seed_prefix")
        self.input_nonce = kwargs.get("input_nonce")
        self.seed = kwargs.get("seed", None)
        self.g = th.Generator(device="cpu")
        if self.seed is None:
            self.g.seed()
        else:
            self.g.manual_seed(self.seed)
        pass

    def encode(self, private_message_bit: bitarray.bitarray, context: str = None, verbose = False):
        """
        :param msg: np array of 0s and 1s constituting a message bitstring
        :return:
        """
        message = private_message_bit
        enc = self.medium.enc
        device = self.medium.device  # may want to restrict to CPU!
        topk = self.medium.probs_top_k
        precision = self.precision
        is_sort = self.is_sort
        finish_sent = self.finish_sent

        if verbose:
            print("Starting reset...")

        max_val = 2 ** precision
        threshold = 2 ** (-precision)
        cur_interval = [0, max_val]  # bottom inclusive, top exclusive

        prev = context
        enc_context = th.LongTensor(self.medium.encode_context(context))
        output = enc_context
        past = None

        total_num = 0
        total_num_for_stats = 0
        total_log_probs = 0
        total_kl = 0  # in bits
        total_entropy_ptau = 0
        total_num_sents = 0
        mask_generator = DRBG(self.input_key, self.sample_seed_prefix + self.input_nonce)

        stats = {"message_len_bits": len(message),
                 "loop_error": 0.0}
        stats_traj = defaultdict(list)

        # bit_index = int(th.randint(0, high=1000, size=(1,)))  # ADG STEGA
        bit_index = 0  # we don't need bit_index here as our message is already random...
        with th.no_grad():
            i = 0
            j = 0
            sent_finish = False
            stega_bit = ''
            while i < len(message) or (finish_sent and not sent_finish):

                if j == 0:
                    probs, info = self.medium.reset(context=context)
                else:
                    t_medium_1 = time.time()
                    probs, info = self.medium.step(prev)
                    delta_t_medium = time.time() - t_medium_1
                    stats_traj["enc_t_medium_per_step"].append(delta_t_medium)

                j += 1
                probs = th.from_numpy(probs.astype(np.float64))  # remember we are getting numpy arrays out of this!
                # probs[1] = 0  # set unk to zero # CHECK: ADG
                probs, indices = probs.sort(descending=True) # ADG
                # start recursion
                bit_tmp = 0

                t_iter_1 = time.time()

                if "kl(sampled|true)" in info:
                    stats_traj["kl(sampled|true)"].append(info["kl(sampled|true)"])
                    stats_traj["kl(uniform|true)"].append(info["kl(uniform|true)"])
                    stats_traj["kl(sampled|uniform)"].append(info["kl(sampled|uniform)"])
                    stats_traj["kl(uniform|sampled)"].append(info["kl(uniform|sampled)"])
                    stats_traj["chisquare_p(sampled|true)"].append(info["chisquare_p(sampled|true)"])
                    stats_traj["chisquare_p(uniform|true)"].append(info["chisquare_p(uniform|true)"])

                # log medium qualities
                stats_traj["medium_entropy_raw"].append(info["medium_entropy_raw"])
                stats_traj["medium_entropy"].append(info["medium_entropy"])
                stats_traj["medium_entropy_over_raw"].append(info["medium_entropy"] /
                                                             info["medium_entropy_raw"])
                stats_traj["medium_logit_dim"].append(probs.shape[0])

                probs_temp = probs
                log_probs_temp = th.from_numpy(info["log_probs"].astype(np.float64))
                log_probs = th.from_numpy(info["log_probs_T=1"].astype(np.float64))
                entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)
                total_entropy_ptau += entropy_in_this_distribution

                probs_temp_int = probs_temp
                indices_orig = th.from_numpy(info["indices"])

                # conditions for having reached the end of the message
                if i >= len(message):
                    selection = 0
                    sent_finish = is_sent_finish(indices[selection].item(), enc)
                    print("Is Finished is true for i={}!".format(i))
                else:
                    total_num_for_stats += 1
                    prob = probs.cpu().clone()  #ADG fix
                    q = probs.cpu().clone().zero_()
                    # while prob[0] <= 0.5: # BUG IN STEGA
                    if prob[0].item() > 0.5:
                        indices = indices_orig
                    while prob[0].item() <= 0.5: 
                        """ 
                        ADG code taken from https://github.com/Mhzzzzz/ADG-steganography
                        with minimal modificatons
                        """
                        # embedding bit
                        bit = 1
                        while (1 / 2 ** (bit + 1)) > prob[0]:
                            bit += 1
                        mean = 1 / 2 ** bit
                        # dp
                        prob = prob.tolist()
                        indices = indices_orig.tolist()
                        result = []
                        for _ in range(2 ** bit):
                            result.append([[], []])
                        for i_ in range(2 ** bit - 1):
                            result[i_][0].append(prob[0])
                            result[i_][1].append(indices[0])
                            del (prob[0])
                            del (indices[0])
                            while sum(result[i_][0]) < mean:
                                delta = mean - sum(result[i_][0])
                                index = near(prob, delta)
                                if prob[index] - delta < delta:
                                    result[i_][0].append(prob[index])
                                    result[i_][1].append(indices[index])
                                    del (prob[index])
                                    del (indices[index])
                                else:
                                    break
                            mean = sum(prob) / (2 ** bit - i_ - 1)
                        result[2 ** bit - 1][0].extend(prob)
                        result[2 ** bit - 1][1].extend(indices)
                        # read secret message
                        bit_embed = [int(_) for _ in message[bit_index + bit_tmp:bit_index + bit_tmp + bit]]
                        int_embed = bits2int(bit_embed)
                        # updating
                        prob = th.FloatTensor(result[int_embed][0]).to(device)
                        indices = th.LongTensor(result[int_embed][1]).to(device)
                        prob = prob / prob.sum()
                        prob, _ = prob.sort(descending=True)
                        indices = indices[_]
                        bit_tmp += bit
                        i += bit  # we have added an extra few bits
                        # Now reconstruct q
                        for _, g in enumerate(result):
                            ps, idxs = g
                            g_sum = sum(ps)
                            for id_ in idxs:
                                pidx = indices_orig.cpu().tolist().index(id_)
                                q[pidx] = probs[pidx] / (g_sum*(2**bit))
                        kl = kl2(q/q.sum(), probs[:len(q)])
                        if kl < 0.0:
                            h = 1
                            pass
                        total_kl += kl
                        pass
                    if j > len(private_message_bit) * 100:
                        stats["loop_error"] = 1.0
                        break

                # ADG UPDATE
                selection = int(th.multinomial(prob, 1))
                prev = indices[selection].view(1)
                output = th.cat((output, prev.to(output[0].device)))
                total_log_probs += log_probs[selection].item()

                total_num += 1

                # For text->bits->text
                partial = enc.decode(output[len(context):].tolist())
                if '<eos>' in partial:
                    break

                if j > 0:
                    delta_t_step_no_medium = time.time() - t_iter_1
                    stats_traj["enc_t_step_no_medium"].append(delta_t_step_no_medium)

            avg_NLL = -total_log_probs / total_num_for_stats
            avg_KL = total_kl / total_num_for_stats
            avg_Hq = total_entropy_ptau / total_num_for_stats
            words_per_bit = total_num_for_stats / i

            stats["avg_NLL"] = avg_NLL
            stats["avg_KL"] = avg_KL
            stats["avg_Hq"] = avg_Hq
            stats["words_per_bit"] = words_per_bit

            for k, v in stats_traj.items():
                if k in ["kl(sampled|true)", "kl(uniform|true)", "kl(sampled|uniform)", "kl(uniform|sampled)"]:
                    continue
                stats[k + "/mean"] = np.array(v).mean()
                stats[k + "/std"] = np.array(v).std()
                stats[k + "/80"] = np.sort(np.array(v))[int(len(v) * 0.8)]
                stats[k + "/20"] = np.sort(np.array(v))[int(len(v) * 0.2)]
                stats[k + "/95"] = np.sort(np.array(v))[int(len(v) * 0.95)]
                stats[k + "/5"] = np.sort(np.array(v))[int(len(v) * 0.05)]

            if "kl(sampled|true)" in stats_traj:
                i = 0
                while i < len(stats_traj["kl(sampled|true)"]):
                    stats["kl(sampled|true)_it{}".format(i)] = stats_traj["kl(sampled|true)"][i]
                    stats["kl(uniform|true)_it{}".format(i)] = stats_traj["kl(uniform|true)"][i]
                    stats["kl(uniform|sampled)_it{}".format(i)] = stats_traj["kl(uniform|sampled)"][i]
                    stats["kl(sampled|uniform)_it{}".format(i)] = stats_traj["kl(sampled|uniform)"][i]
                    stats["chisquare_p(sampled|true)_it{}".format(i)] = stats_traj["chisquare_p(sampled|true)"][i]
                    stats["chisquare_p(uniform|true)_it{}".format(i)] = stats_traj["chisquare_p(uniform|true)"][i]
                    stats["(kl(sampled|true)-kl(uniform|true))_it{}".format(i)] = stats_traj["kl(sampled|true)"][i] -\
                        stats_traj["kl(uniform|true)"][i]
                    i += 100

            stats["n_steps"] = j
            stats["bits_per_step"] = len(private_message_bit) / float(j)
            stats["steps_per_bit"] = j / float(len(private_message_bit))
            stats["eff"] = len(private_message_bit) / sum(stats_traj["medium_entropy"])
            stats["eff_raw"] = len(private_message_bit) / sum(stats_traj["medium_entropy_raw"])
            stats["eff_output"] = len(private_message_bit) / len(output[len(enc_context):])

            return self.medium.enc.decode(output[len(enc_context):].tolist()), \
                   output[len(enc_context):].tolist(), \
                   stats
