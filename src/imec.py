import bitarray
from copy import deepcopy
from collections import defaultdict
import numpy as np
import time
import torch as th

from mec import minimum_entropy_coupling
from medium import ActionLabelException
from utils import DRBG, kl2, entropy2


class IMECEncoder:

    def __init__(self, medium, block_size=2 ** 8, **kwargs):
        self.medium = medium
        self.context = kwargs.get("context", None)
        self.block_size = block_size
        self.send_block_size_header = kwargs.get("send_block_size_header", None)  # not supported yet
        self.send_n_chunks_header = kwargs.get("send_n_chunks_header", True)  # first
        self.pad_last_belief_chunk = kwargs.get("pad_last_belief_chunk", True)
        # ensure all belief chunks are of same size
        self.mec_mode = kwargs.get("mec_mode", "dense")
        self.mec_atol = kwargs.get("mec_atol", 1E-7)
        self.mec_warning_atol = kwargs.get("mec_warning_atol", 1E-5)
        self.belief_entropy_threshold = kwargs.get("belief_entropy_threshold", 10E-10)
        self.clean_up_output = kwargs.get("clean_up_output", False)
        self.seed = kwargs.get("seed", None)
        self.g = np.random.default_rng(self.seed)
        self.use_lowmem_variant = kwargs.get("use_lowmem_variant", 0)
        pass

    def encode(self, private_message_bit: bitarray.bitarray, context: str = None, verbose=False):
        """
        :param msg: np array of 0s and 1s constituting a message bitstring
        :return:
        """
        t_iter_1 = time.time()
        if verbose:
            print("Starting reset...")
        probs, info = self.medium.reset(context=context)
        if verbose:
            print("Reset finished!")

        # calculate chunks
        block_sizes = [self.block_size for i in range(int(len(private_message_bit) // self.block_size))]
        if len(private_message_bit) % self.block_size:
            block_sizes += [int(len(private_message_bit) % self.block_size)]

        # break bitstring down into decimal numbers
        idx = 0
        msg_chunks = []
        for cs in block_sizes:
            msg_chunk = np.array(private_message_bit[idx:idx + cs].tolist()).dot(
                1 << np.arange(cs, dtype='int64')[::-1])
            msg_chunks.append(msg_chunk)
            idx += cs

        # initialise beliefs
        if self.pad_last_belief_chunk:
            beliefs = [np.zeros(2 ** self.block_size, dtype=np.longdouble) + 1.0 / (2 ** self.block_size) for k, _ in
                       enumerate(block_sizes)]
        else:
            beliefs = [np.zeros(2 ** cs, dtype=np.longdouble) + 1.0 / (cs ** 2) for k, cs in enumerate(block_sizes)]

        stats_traj = defaultdict(list)
        stats = {"block_sizes": block_sizes,
                 "bitlen_msg_enc": len(private_message_bit),
                 "loop_error": 0.0}

        n_steps = 0
        while True:
            n_steps += 1
            # choose next chunk to be encoded
            belief_entropies = np.array([entropy2(b) for b in beliefs])

            # Check if we are done from an encoding threshold perspective
            if max(belief_entropies) <= self.belief_entropy_threshold:
                if self.clean_up_output:
                    if info.get("end_of_line", False):
                        # We are done encoding, but need to finish off current line.
                        break
                else:
                    break

            next_chunk_id = np.argmax(belief_entropies)

            next_chunk_content = msg_chunks[next_chunk_id]
            t1 = time.time()

            # calculate minimum entropy coupling
            t00 = time.time()
            mec_dict = minimum_entropy_coupling(
                beliefs[next_chunk_id],
                probs,
                select_row=next_chunk_content,
                select_col="all",
                mode=self.mec_mode,
                algo_atol=self.mec_atol,
                warning_atol=self.mec_warning_atol,
            )
            t01 = time.time() - t00
            stats_traj["mec_time_pure[s]"].append(t01)
            dt = time.time() - t1
            stats_traj["mec_step_time[s]"].append(dt)
            if "M_entropy" in mec_dict:
                stats_traj["avg_HM[bit]"].append(mec_dict["M_entropy"])
                stats_traj["avg_Hq[bit]"].append(mec_dict["q_entropy"])
                kl_q = mec_dict["kl_q"] if \
                    np.isfinite(mec_dict["kl_q"]) else \
                    kl2(np.array(mec_dict["q_est"], dtype=np.longdouble)/np.array(mec_dict["q_est"],
                                                                                  dtype=np.longdouble).sum(),
                                                                                  probs/probs.sum())
                kl_q = np.around(kl_q, decimals=18)
                if kl_q < 0.0:
                    z = 5
                stats_traj["avg_KL"].append(kl_q)
                stats_traj["avg_KL_p"].append(mec_dict["kl_p"])
                stats_traj["additive_gap"].append(mec_dict["additive_gap"])
                stats_traj["mec_time[s]"].append(mec_dict["mec_time[s]"])

            M_row_next_chunk = mec_dict["M_selected_row"]
            M_row_next_chunk = M_row_next_chunk / M_row_next_chunk.sum()

            # select next action
            next_action = self.g.choice(M_row_next_chunk.shape[0], p=M_row_next_chunk.astype(np.float64))

            # update beliefs
            belief_update = mec_dict["M_colfirst"][next_action]

            beliefs[next_chunk_id] = belief_update / belief_update.sum()
            belief_entropy_delta = belief_entropies[next_chunk_id] - entropy2(beliefs[next_chunk_id])

            delta_t_step_no_medium = time.time() - t_iter_1
            stats_traj["enc_t_step_no_medium"].append(delta_t_step_no_medium)

            t_medium_1 = time.time()
            probs, info = self.medium.step(self.medium.action_labels[next_action])
            delta_t_medium = time.time() - t_medium_1
            stats_traj["enc_t_medium_per_step"].append(delta_t_medium)

            t_iter_1 = time.time()
            if "kl(sampled|true)" in info:
                stats_traj["kl(sampled|true)"].append(info["kl(sampled|true)"])
                stats_traj["kl(uniform|true)"].append(info["kl(uniform|true)"])
                stats_traj["kl(sampled|uniform)"].append(info["kl(sampled|uniform)"])
                stats_traj["kl(uniform|sampled)"].append(info["kl(uniform|sampled)"])
                stats_traj["chisquare_p(sampled|true)"].append(info["chisquare_p(sampled|true)"])
                stats_traj["chisquare_p(uniform|true)"].append(info["chisquare_p(uniform|true)"])

            stats_traj["medium_entropy_raw"].append(info["medium_entropy_raw"])
            stats_traj["medium_entropy"].append(info["medium_entropy"])
            stats_traj["medium_entropy_over_raw"].append(info["medium_entropy"] /
                                                         info["medium_entropy_raw"])

            stats_traj["active_belief_entropy_delta"].append(belief_entropy_delta)
            stats_traj["active_belief_entropy(delta_over_raw)"].append(belief_entropy_delta /
                                                                       info["medium_entropy"])
            stats_traj["active_belief_entropy(delta_over_raw)"].append(belief_entropy_delta /
                                                                       info["medium_entropy_raw"])
            stats_traj["medium_logit_dim"].append(probs.shape[0])

        for k, v in stats_traj.items():
            if k in ["kl(sampled|true)", "kl(uniform|true)", "kl(sampled|uniform)", "kl(uniform|sampled)",
                         "chisquare_p(sampled|true)", "chisquare_p(sampled|true)"]:
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
                stats["abs(kl(sampled|true),kl(uniform|true))_it{}".format(i)] = abs(stats_traj["kl(sampled|true)"][i] -
                                                                                     stats_traj["kl(uniform|true)"][i])
                stats["(kl(sampled|true)-kl(uniform|true))_it{}".format(i)] = stats_traj["kl(sampled|true)"][i] -\
                                                                              stats_traj["kl(uniform|true)"][i]
                i += 100

        # clean up output ensures that we backtrack the output to the last EOL
        output = self.medium.get_output(clean_up=self.clean_up_output)

        stats["n_steps"] = n_steps
        stats["bits_per_step"] = len(private_message_bit) / float(n_steps)
        stats["steps_per_bit"] = n_steps / float(len(private_message_bit))
        stats["eff"] = len(private_message_bit) / sum(stats_traj["medium_entropy"])
        stats["eff_output"] = len(private_message_bit) / len(output)
        stats["eff_raw"] = len(private_message_bit) / sum(stats_traj["medium_entropy_raw"])

        output_str = self.medium.humanify(output)
        return output_str, output, stats


class IMECDecoder:

    def __init__(self, medium, block_size=None, n_chunks=None, last_block_size=None, use_header=False, **kwargs):
        self.use_header = use_header
        self.medium = medium
        self.context = kwargs.get("context", None)
        self.block_size = block_size
        self.send_block_size_header = kwargs.get("send_block_size_header", None)  # not supported yet
        self.send_n_chunks_header = kwargs.get("send_n_chunks_header", True)  # first
        self.pad_last_belief_chunk = kwargs.get("pad_last_belief_chunk", True)
        self.last_block_size = last_block_size
        self.n_chunks = n_chunks
        if not self.pad_last_belief_chunk:
            assert (last_block_size is not None) and not use_header, "need to set last_block_size and cannot " \
                                                                     "use header if pad_last_belief_chunk is being used!"
        # ensure all belief chunks are of same size
        self.mec_mode = kwargs.get("mec_mode", "dense")
        self.mec_atol = kwargs.get("mec_atol", 1E-7)
        self.mec_warning_atol = kwargs.get("mec_warning_atol", 1E-5)
        self.belief_entropy_threshold = kwargs.get("belief_entropy_threshold", 10E-10)
        self.clean_up_output = kwargs.get("clean_up_output", False)
        self.header_bit_sizes = kwargs.get("header_bit_size", (4, 8))  # number of bits for (block_size, n_chunks)
        self.header_block_size = kwargs.get("header_block_size", 4)
        self.header_belief_entropy_threshold = kwargs.get("header_belief_entropy_threshold",
                                                          10E-10)  # number of bits for (block_size, n_chunks)
        if not use_header:
            assert block_size is not None, "If header is not used, need to set chunk size!"
            assert n_chunks is not None, "If header is not used, need to set n_chunks!"
        pass

    def decode(self, public_message_token, context: str = None, verbose: bool = False, text=None, **kwargs):
        probs, info = self.medium.reset(context=context)

        msgt_header_offset = 0
        block_sizes = None

        block_sizes = [self.block_size] * self.n_chunks
        if not self.pad_last_belief_chunk:
            block_sizes += [self.last_block_size]

        # initialise beliefs
        if self.pad_last_belief_chunk:
            beliefs = [np.zeros(2 ** self.block_size, dtype=np.longdouble) + 1.0 / (2 ** self.block_size) for k, _ in
                       enumerate(block_sizes)]
        else:
            beliefs = [np.zeros(2 ** cs, dtype=np.longdouble) + 1.0 / (2 ** cs) for k, cs in enumerate(block_sizes)]

        stats_traj = defaultdict(list)
        stats = {"public_message_len": len(public_message_token)}

        t_iter_1 = None
        for msg_token in public_message_token:
            if verbose:
                print("DEC PROBS:", probs[:5])

            # choose next chunk to be encoded
            belief_entropies = np.array([entropy2(b) for b in beliefs])
            next_chunk_id = np.argmax(belief_entropies)

            # select next action
            try:
                next_action = self.medium.action_labels.cpu().tolist().index(msg_token)
            except:
                a = 5
                raise ActionLabelException(action_labels=self.medium.action_labels.cpu().tolist(),
                                           msg_token=msg_token,
                                           msg_tokens=public_message_token,
                                           message="Not in list!")

            mec_dict = minimum_entropy_coupling(
                beliefs[next_chunk_id],
                probs,
                select_row=None,
                select_col=next_action,
                method="kocaoglu",
                mode=self.mec_mode,
                algo_atol=self.mec_atol,
                warning_atol=self.mec_warning_atol)
            vec2 = mec_dict["M_selected_col"]

            if t_iter_1 is not None:
                delta_t_step_no_medium = time.time() - t_iter_1
                stats_traj["dec_t_step_no_medium"].append(delta_t_step_no_medium)
            t_medium_1 = time.time()

            probs, info = self.medium.step(self.medium.action_labels[next_action])

            delta_t_medium = time.time() - t_medium_1
            stats_traj["dec_t_medium_per_step"].append(delta_t_medium)
            t_iter_1 = time.time()
            beliefs[next_chunk_id] = vec2 / vec2.sum()

        for k, v in stats_traj.items():
            stats[k + "/mean"] = np.array(v).mean()
            stats[k + "/std"] = np.array(v).std()
            stats[k + "/80"] = np.sort(np.array(v))[int(len(v) * 0.8)]
            stats[k + "/20"] = np.sort(np.array(v))[int(len(v) * 0.2)]
            stats[k + "/95"] = np.sort(np.array(v))[int(len(v) * 0.95)]
            stats[k + "/5"] = np.sort(np.array(v))[int(len(v) * 0.05)]

        output = [format(np.argmax(b), '0{}b'.format(cs)) for b, cs in zip(beliefs, block_sizes)]
        output = bitarray.bitarray("".join(output))
        return output, stats


def apply_random_mask(message_bits, input_key, sample_seed_prefix, input_nonce):
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)
    mask_bits = mask_generator.generate_bits(len(message_bits))
    masked_message_bits = deepcopy(message_bits)
    for b in range(0, len(message_bits)):
        masked_message_bits[b] = message_bits[b] ^ mask_bits[b]
    return masked_message_bits


def remove_random_mask(message_bits, input_key, sample_seed_prefix, input_nonce):
    return apply_random_mask(message_bits, input_key, sample_seed_prefix, input_nonce)


if __name__ == "__main__":
    block_size = 4  # in bits
    medium = METEORMedium(
        seed=12342,
        temp=0.95,
        probs_top_k=50  # severe entropy loss!
    )

    plaintext_message = "bds04I7"
    use_arithmetic_coding = False

    # plaintext_message = "sample text"
    # use_arithmetic_coding = True

    bit_msg, cinfo = str2bit(plaintext_message, use_arithmetic_coding=False, medium=medium)

    mask_cfg = {"input_key": b'0x01' * 64,
                "sample_seed_prefix": b'sample',
                "input_nonce": b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'}
    msg_bits, rinfo = apply_random_mask(bit_msg, **mask_cfg)

    import math

    encoder = IMECEncoder(block_size=block_size, medium=medium,
                          clean_up_output=False)  # medium needs to support the logit() function

    chosen_context = "Despite a long history of research and wide-spread applications to censorship " \
                     "resistant systems, practical steganographic systems capable of embedding messages " \
                     "into realistic communication distributions, like text, do not exist."
    chosen_context += "\n\n"  # to add a little spacing
    encoded_message, enc_stats = encoder.encode(msg_bits=msg_bits, context=chosen_context, verbose=True)

    print("ENCODED MESSAGE:")
    print(encoded_message)

    decoder = IMECDecoder(block_size=block_size, n_chunks=int(math.ceil(len(bit_msg) / block_size)), medium=medium,
                          clean_up_output=False)  # medium needs to support the logit() function

    decoded_message, dec_stats = decoder.decode(encoded_msg=encoded_message, context=chosen_context)

    bit_msg, rinfo = remove_random_mask(decoded_message, **rinfo)
    decoded_message = bit2str(bit_msg, **cinfo)

    print("PLAINTEXT MESSAGE:")
    print(plaintext_message)

    print("DECODED MESSAGE:")
    print(decoded_message)

    pass
