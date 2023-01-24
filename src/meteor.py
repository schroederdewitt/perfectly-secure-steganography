import bitarray
from collections import defaultdict
import numpy as np
import time
import torch as th
from utils import is_sent_finish, num_same_from_beg, int2bits, bits2int, DRBG, bin_sort, kl2, entropy2


class METEOREncoder:

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
        with th.no_grad():
            i = 0
            j = 0
            sent_finish = False
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
                probs_temp_int = probs_temp
                indices = th.from_numpy(info["indices"])

                # conditions for having reached the end of the message
                if i >= len(message):
                    selection = 0
                    sent_finish = is_sent_finish(indices[selection].item(), enc)
                    print("Is Finished is true for i={}!".format(i))
                else:
                    # Cutoff low probabilities that would be rounded to 0
                    cur_int_range = cur_interval[1] - cur_interval[0]
                    cur_threshold = 1 / cur_int_range

                    # Rescale to correct range
                    probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                    #entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)
                    entropy_in_this_distribution = entropy2(probs_temp)

                    # Round probabilities to integers given precision
                    probs_temp_int = probs_temp_int.round().long()

                    if is_sort:
                        probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                                           entropy_in_this_distribution, device="cpu")
                    cum_probs = probs_temp_int.cumsum(0)

                    # Remove any elements from the bottom if rounding caused the total prob to be too large
                    overfill_index = (cum_probs > cur_int_range).nonzero()
                    if len(overfill_index) > 0:
                        cum_probs = cum_probs[:overfill_index[0]]

                    # Add any mass to the top if removing/rounding causes the total prob to be too small
                    cum_probs += cur_int_range - cum_probs[-1]  # add

                    # Get out resulting probabilities
                    probs_final = cum_probs.clone()
                    probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                    # Convert to position in range
                    cum_probs += cur_interval[0]

                    # Apply the mask to the message
                    message_bits = message[i:i + precision]
                    if i + precision > len(message):
                        message_bits = message_bits + [0] * (i + precision - len(message))

                    mask_bits = mask_generator.generate_bits(precision)
                    for b in range(0, len(message_bits)):
                         message_bits[b] = message_bits[b] ^ mask_bits[b]

                    # Get selected index based on binary fraction from message bits
                    message_idx = bits2int(reversed(message_bits))
                    selection = (cum_probs > message_idx).nonzero()[0].item()

                    # Calculate new range as ints
                    new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                    new_int_top = cum_probs[selection]

                    # Convert range to bits
                    new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                    new_int_top_bits_inc = list(
                        reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                    # Consume most significant bits which are now fixed and update interval
                    num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                    i += num_bits_encoded
                    # print("Num bits encoded: {}".format(num_bits_encoded))

                    # Gather statistics
                    total_log_probs += log_probs[selection].item()

                    q = probs_final.double() / probs_final.sum()
                    logq = q.log()
                    # total_kl += kl(q, logq, log_probs[:len(q)]) # wrong direction according to Cachin!
                    total_kl += kl2(q, probs[:len(q)])
                    total_entropy_ptau += entropy_in_this_distribution
                    total_num_for_stats += 1

                    if j > len(private_message_bit) * 100:
                        stats["loop_error"] = 1.0
                        break

                # Update history with new token
                prev = indices[selection].view(1)
                output = th.cat((output, prev))
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

class METEORDecoder:

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

    def decode(self, public_message_token, text=None, context: str = None,
               verbose: bool=False, private_message_bitlen=None, **kwargs):

        inp = public_message_token
        enc = self.medium.enc
        device = self.medium.device  # may want to restrict to CPU!
        topk = self.medium.probs_top_k
        precision = self.precision
        is_sort = self.is_sort
        finish_sent = self.finish_sent
        max_val = 2 ** precision
        threshold = 2 ** (-precision)
        cur_interval = [0, max_val]  # bottom inclusive, top exclusive

        mask_generator = DRBG(self.input_key, self.sample_seed_prefix + self.input_nonce)


        stats_traj = defaultdict(list)
        stats = {"public_message_len": len(public_message_token)}
        t_iter_1 = None

        enc_context = self.medium.encode_context(context)
        enc_context = th.tensor(enc_context[-1022:], device=device, dtype=th.long)
        prev = enc_context
        message = []
        with th.no_grad():
            i = 0
            j = 0
            while i < len(inp):

                if t_iter_1 is not None:
                    delta_t_step_no_medium = time.time() - t_iter_1
                    stats_traj["dec_t_step_no_medium"].append(delta_t_step_no_medium)
                t_medium_1 = time.time()

                if j == 0:
                    probs, info = self.medium.reset(context=context)
                else:
                    probs, info = self.medium.step(prev)
                j += 1
                probs = th.from_numpy(probs.astype(np.float64))

                delta_t_medium = time.time() - t_medium_1
                stats_traj["dec_t_medium_per_step"].append(delta_t_medium)
                t_iter_1 = time.time()

                probs_temp = probs
                log_probs_temp = th.from_numpy(info["log_probs"].astype(np.float64))
                indices = th.from_numpy(info["indices"])
                probs_temp_int = probs_temp

                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                k = len(probs)

                # probs_temp_int = probs_temp[:k]  # Cutoff all but top k

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
                # entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)
                entropy_in_this_distribution = entropy2(probs_temp)

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                if is_sort:
                    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                                       entropy_in_this_distribution, device="cpu")
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]
                    k = overfill_index[0].item()

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Covnert to position in range
                cum_probs += cur_interval[0]

                rank = (indices == inp[i]).nonzero().item()

                # Handle most errors that could happen because of BPE with heuristic
                is_bpe_error = 0
                if rank >= k:
                    true_token_text = enc.decoder[inp[i]]
                    for rank_idx in range(k):
                        prop_token_text = enc.decoder[indices[rank_idx].item()]
                        # common case that is not caught
                        if inp[i] == 128 and indices[rank_idx] == 198:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            break

                        # Is there a more likely prefix token that could be the actual token generated?
                        if len(prop_token_text) <= len(true_token_text) and \
                                prop_token_text == true_token_text[:len(prop_token_text)]:
                            rank = rank_idx
                            suffix = true_token_text[len(prop_token_text):]
                            suffix_tokens = enc.encode(suffix)  # a list
                            inp[i] = indices[rank_idx].item()
                            inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break

                        # Is there a more likely longer token that could be the actual token generated?
                        elif len(prop_token_text) > len(true_token_text) and \
                                true_token_text == prop_token_text[:len(true_token_text)]:
                            whole_text = true_token_text
                            num_extra = 1
                            while len(whole_text) < len(prop_token_text):
                                whole_text += enc.decoder[inp[i + num_extra]]
                                num_extra += 1
                            if prop_token_text == whole_text[:len(prop_token_text)]:
                                rank = rank_idx
                                inp[i] = indices[rank_idx].item()
                                for j in range(1, num_extra):
                                    del inp[i + j]

                                if len(whole_text) > len(prop_token_text):
                                    suffix = whole_text[len(prop_token_text):]
                                    suffix_tokens = enc.encode(suffix)  # a list
                                    inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                                break

                    else:
                        print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                        rank = 0
                        is_bpe_error = 1

                stats_traj["BPE_ERROR"].append(is_bpe_error)

                selection = rank

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Emit most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                if i == len(inp) - 1:
                    new_bits = new_int_bottom_bits_inc
                else:
                    new_bits = new_int_top_bits_inc[:num_bits_encoded]

                # Get the mask and apply it to the recovered bits
                mask_bits = mask_generator.generate_bits(precision)
                for b in range(0, len(new_bits)):
                    new_bits[b] = new_bits[b] ^ mask_bits[b]
                message += new_bits

                # Update history with new token
                prev = th.tensor([inp[i]], device=device, dtype=th.long)

                i += 1

        for k, v in stats_traj.items():
            stats[k + "/mean"] = np.array(v).mean()
            stats[k + "/std"] = np.array(v).std()

        output = bitarray.bitarray(message)
        if private_message_bitlen is not None:
            output = output[:private_message_bitlen]
        return output, stats
