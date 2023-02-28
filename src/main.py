import argparse
import bitarray
from collections import defaultdict, namedtuple
from copy import deepcopy
import dill
import _pickle as dill
import math
import numpy as np
import os
import random
import sys
import time
from torch.multiprocessing import set_start_method
import traceback
import uuid
import yaml
import wandb
import zmq

from wikitext import Wikitext103Generator
from medium import METEORMedium, CIFARMedium, RandomMedium, VariRandomMedium, WaveRNNMedium
from imec import apply_random_mask, remove_random_mask, IMECEncoder, IMECDecoder
from meteor import METEOREncoder, METEORDecoder
from arithmetic import ArithmeticEncoder, ArithmeticDecoder
from utils import bcolors, str2bit, bit2str, get_model


def parse_args():
    parser = argparse.ArgumentParser(description="IMEC Experiment flags")
    parser.add_argument("--cfg-file", help="path to train experiment config yaml file",
                        type=str, default=None)
    parser.add_argument("--block-size", help="size of each chunk (in bits)", type=int, default=4)
    parser.add_argument("--context-len-bytes", help="length of message",
                        type=int, default=1024)
    parser.add_argument("--device", help="device", type=str, default="cpu")
    parser.add_argument("--dbg-mode", help="Debug mode",
                        type=int, default=0)
    parser.add_argument("--dbg-print-context", help="Debug mode - print context",
                        type=int, default=0)
    parser.add_argument("--dbg-save-audio", help="Debug mode - save audio",
                        type=int, default=0)
    parser.add_argument("--group-name", help="experiment group name", type=str)
    parser.add_argument("--mec-mode", help="sparse, or dense",
                        type=str, default="dense")
    parser.add_argument("--meteor-finish-sent", help="sparse, or dense",
                        type=str, default=False)
    parser.add_argument("--meteor-precision", help="default: 32 (should be 0 for imec)",
                        type=int, default=0)
    parser.add_argument("--meteor-is-sort", help="",
                        type=int, default=0)
    parser.add_argument("--medium", help="medium (meteor/random)",
                        type=str, default="meteor")
    parser.add_argument("--medium-entropy-loss-threshold", help="entropy loss threshold",
                        type=float, default=0.95)
    parser.add_argument("--medium-temp", help="medium sample temperature",
                        type=float, default=0.95)
    parser.add_argument("--medium-top-k", help="medium sample topk",
                        type=int, default=50)
    parser.add_argument("--imec-belief-entropy-threshold", help="", type=float, default=10E-10)
    parser.add_argument("--message-len-bytes", help="length of message",
                        type=int, default=10)
    parser.add_argument("--message-mode", help="Either text, or randombits",
                        type=str, default="randombits")
    parser.add_argument("--method", help="imec, or meteor", type=str, default="imec")
    parser.add_argument("--model-device", help="model device", type=str, default="cpu")
    parser.add_argument("--model-name", help="model name", type=str, default="gpt2")
    parser.add_argument("--name", help="experiment name",
                        type=str)
    parser.add_argument("--num-procs", help="Number of processes used",
                        type=int, default=1)
    parser.add_argument("--n-model-procs", help="Number of model processes used",
                        type=int, default=1)
    parser.add_argument("--seed", help="general random seed",
                        type=int, default=-1)
    parser.add_argument("--stop-after-n-trajectories", help="",
                        type=int, default=0)  # if 0 then run for infinite amount of samples
    parser.add_argument("--use-arithmetic-coding", help="If 1 use arithmetic coding",
                        type=int, default=0)
    parser.add_argument("--use-chunk-header", help="If 1 add a chunk header to the messages",
                        type=int, default=0)
    parser.add_argument("--verbose", help="Whether process is verbose or not",
                        type=int, default=1)
    parser.add_argument("--wandb-project", help="project name",
                        type=str)
    parser.add_argument("--wandb-entity", help="entity name",
                        type=str)

    args = parser.parse_args()

    cfg = {}
    if args.cfg_file is not None:
        with open(args.cfg_file, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def update_dct(d, u, overwrite=False):
        import collections.abc
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dct(d.get(k, {}), v, overwrite=overwrite)
            else:
                if k not in d:
                    d[k] = v
                else:
                    d[k] = v if overwrite else d[k]
        return d

    # Update cfg with any default config that isn't present in the config file
    cfg_default = {}
    if cfg_default:
        cfg = update_dct(cfg, cfg_default, overwrite=False)

    # Overwrite cfg with any argparse items that aren't present
    final_cfg = update_dct(cfg, {k: v for k, v in args.__dict__.items() if v is not None}, overwrite=True)

    return final_cfg


def run(cfg, model, enc, context_generator, _id):
    print("Run process started.".format(_id))
    # wandb.init(**cfg_dct["wandb_dct"])
    success_rates = []

    # Set up a process-wise random generator
    proc_seed = (cfg.seed * 99999) % 77777
    proc_rng = np.random.default_rng(proc_seed)

    stats_traj = defaultdict(list)
    stats = {}

    for i in range(cfg.stop_after_n_trajectories):

        print("Starting trajectory {} in process #{}".format(i, _id))

        try:
            if cfg.medium == "meteor":
                medium = METEORMedium(entropy_loss_threshold=cfg.medium_entropy_loss_threshold,
                                      precision=cfg.meteor_precision,
                                      seed=cfg.seed,
                                      temp=cfg.medium_temp,
                                      probs_top_k=cfg.medium_top_k,
                                      model=model,
                                      enc=enc,
                                      device=cfg.model_device,
                                      mec_mode=cfg.mec_mode,
                                      model_name=cfg.model_name
                                      )
            elif cfg.medium == "random":
                medium = RandomMedium(entropy_loss_threshold=cfg.medium_entropy_loss_threshold,
                                      precision=cfg.meteor_precision,
                                      seed=cfg.seed,
                                      temp=cfg.medium_temp,
                                      probs_top_k=cfg.medium_top_k,
                                      model=model,
                                      enc=enc,
                                      device=cfg.model_device,
                                      mec_mode=cfg.mec_mode
                                      )
            if cfg.medium == "cifar":
                medium = CIFARMedium( entropy_loss_threshold=cfg.medium_entropy_loss_threshold,
                                      precision=cfg.meteor_precision,
                                      seed=cfg.seed,
                                      temp=cfg.medium_temp,
                                      probs_top_k=cfg.medium_top_k,
                                      model=model,
                                      enc=enc,
                                      device=cfg.model_device,
                                      mec_mode=cfg.mec_mode,
                                      model_name=cfg.model_name,
                                      collect_medium_dists=cfg.collect_medium_dists
                                      )
            elif cfg.medium == "vari_random":
                medium = VariRandomMedium(entropy_loss_threshold=cfg.medium_entropy_loss_threshold,
                                          precision=cfg.meteor_precision,
                                          seed=cfg.seed,
                                          temp=cfg.medium_temp,
                                          probs_top_k=cfg.medium_top_k,
                                          model=model,
                                          enc=enc,
                                          device=cfg.model_device,
                                          mec_mode=cfg.mec_mode
                                          )
            elif cfg.medium == "wavernn":
                medium = WaveRNNMedium(entropy_loss_threshold=cfg.medium_entropy_loss_threshold,
                                       precision=cfg.meteor_precision,
                                       seed=cfg.seed,
                                       temp=cfg.medium_temp,
                                       probs_top_k=cfg.medium_top_k,
                                       model=model,
                                       enc=enc,
                                       device=cfg.model_device,
                                       mec_mode=cfg.mec_mode
                                       )
            else:
                raise Exception("Unknown medium: {}".format(cfg.medium))

            # Retrieve context and generate message
            context = None
            private_message_str = None
            private_message_bit = None
            if not cfg.dbg_mode:
                context = context_generator.sample()
                if context is None:
                    break
                else:
                    context = context[:cfg.context_len_bytes]

                if cfg.dbg_print_context:
                    print("Context:")
                    print(context)

                if cfg.message_mode == "randombits":
                    private_message_bit = bitarray.bitarray(
                        proc_rng.integers(2, size=(cfg.message_len_bytes * 8,)).tolist())
                elif cfg.message_mode == "text":
                    private_message_str = context_generator.get()[:cfg.message_len_bytes]
                    use_arithmetic_coding = cfg.use_arithmetic_coding
                    private_message_bit = str2bit(private_message_str,
                                                  use_arithmetic_coding=use_arithmetic_coding,
                                                  medium=medium)
                else:
                    raise Exception("UNKNOWN message mode: {}".format(cfg.message_mode))
            else:
                print("### ATTENTION: DEBUG MODE ACTIVATED! ###")
                context = 'The La Galissonnière class ironclads were a group of wooden @-@ hulled , armored corvettes built ' \
                          'for the French Navy during the 1870s , meant as a heavier armed and faster ' \
                          'version of the Alma @-@ class ironclad . While all three ships were begun before the Franco @-@ ' \
                          'Prussian War of 1870 – 71 , the construction of the last two ships was delayed for years . ' \
                          'The navy took advantage of the extended construction time of the latter ships to upgrade their ' \
                          'armament . La Galissonnière bombarded Sfax in 1881 as part of the French occupation of Tunisia . ' \
                          'She and her half @-@ sister Triomphante participated in a number of battles during the Sino @-@ ' \
                          'French War of 1884 – 85 . Their sister Victorieuse had a much quieter career . All three ships ' \
                          'were decommissioned in the 1890s . \n \n = = Design and description = = \n \n The La Galissonnière ' \
                          '@-@ class ironclads were designed as faster , more heavily armed versions of the Alma @-@ class ' \
                          'ironclads by Henri Dupuy de Lôme . They used the same central battery layout as their prede'

                private_message_str = 'The taxono'
                use_arithmetic_coding = True
                private_message_bit = str2bit(private_message_str,
                                              use_arithmetic_coding=use_arithmetic_coding,
                                              medium=medium)

            # Set up random text masking params
            mask_cfg = {"input_key": b'\x03' * 64,
                        "sample_seed_prefix": b'sample' if cfg.dbg_mode else proc_rng.integers(2, size=(6,)).tobytes(),
                        "input_nonce": b'\x01' * 64}

            # Now branch into different methods
            encoder = None
            decoder = None
            if cfg.method == "imec":
                encoder_input = apply_random_mask(private_message_bit, **mask_cfg)
                block_size = cfg.block_size  # in bits
                encoder = IMECEncoder(block_size=block_size, medium=medium,
                                      clean_up_output=False,
                                      seed=proc_seed,
                                      mec_mode=cfg.mec_mode,
                                      belief_entropy_threshold=cfg.imec_belief_entropy_threshold)
                                      # medium needs to support the logit() function

                decoder = IMECDecoder(block_size=block_size,
                                      n_chunks=int(math.ceil(len(private_message_bit) / block_size)),
                                      medium=medium,
                                      clean_up_output=False,
                                      belief_entropy_threshold=cfg.imec_belief_entropy_threshold,
                                      mec_mode=cfg.mec_mode)

            elif cfg.method == "meteor":
                encoder_input = private_message_bit
                encoder = METEOREncoder(medium=medium,
                                        seed=cfg.seed,
                                        cleanup_output=False,
                                        finish_sent=cfg.meteor_finish_sent,
                                        precision=cfg.meteor_precision,
                                        is_sort=cfg.meteor_is_sort,
                                        **mask_cfg
                                        )  # medium needs to support the logit() function

                decoder = METEORDecoder(medium=medium,
                                        seed=proc_seed,
                                        cleanup_output=False,
                                        finish_sent=cfg.meteor_finish_sent,
                                        precision=cfg.meteor_precision,
                                        is_sort=cfg.meteor_is_sort,
                                        **mask_cfg
                                        )  # medium needs to support the logit() function

            elif cfg.method == "arithmetic":
                encoder_input = private_message_bit
                encoder = ArithmeticEncoder(medium=medium,
                                        seed=cfg.seed,
                                        cleanup_output=False,
                                        finish_sent=cfg.meteor_finish_sent,
                                        precision=cfg.meteor_precision,
                                        is_sort=cfg.meteor_is_sort,
                                        **mask_cfg
                                        )  # medium needs to support the logit() function

                decoder = ArithmeticDecoder(medium=medium,
                                        seed=proc_seed,
                                        cleanup_output=False,
                                        finish_sent=cfg.meteor_finish_sent,
                                        precision=cfg.meteor_precision,
                                        is_sort=cfg.meteor_is_sort,
                                        **mask_cfg
                                        )  # medium needs to support the logit() function

            else:
                raise Exception("UNKNOWN method: {}".format(cfg.method))

            t0 = time.time()
            t1 = time.time()
            # Proceed with communication
            public_message_str, public_message_token, enc_stats = encoder.encode(
                private_message_bit=encoder_input,
                context=context,
                verbose=cfg.verbose)
            end = time.time() - t1
            enc_stats["enc_t_wall"] = end
            enc_stats["enc_t_wall_per_step"] = enc_stats["enc_t_wall"] / enc_stats["n_steps"]
            if enc_stats["loop_error"] == 1.0:
                wandb.log({"loop_error": 1.0}, step=i)
                raise Exception("Encoding likely encountered a loop! Excepting this one.")

            wandb.log(enc_stats, step=i)

            # add to aggregate statistics
            for k, v in enc_stats.items():
                stats_traj[k].append(v)

            if cfg.verbose:
                print("ENCODED MESSAGE (proc #{}):".format(_id))
                print(public_message_str)

            wandb.log({"iter": i}, step=i)

            ################# CHANNEL DIVIDE ###############################################################

            if cfg.medium == "wavernn" or cfg.medium == "cifar":
                print("WaveRNN currently does not support decoding! s")
                # update aggregate statistics
                for k, v in stats_traj.items():
                    stats["agg_" + k + "/mean"] = np.array(v).mean()
                    stats["agg_" + k + "/std"] = np.array(v).std()
                    stats["agg_" + k + "/80"] = np.sort(np.array(v))[int(len(v) * 0.8)]
                    stats["agg_" + k + "/20"] = np.sort(np.array(v))[int(len(v) * 0.2)]
                    stats["agg_" + k + "/95"] = np.sort(np.array(v))[int(len(v) * 0.95)]
                    stats["agg_" + k + "/05"] = np.sort(np.array(v))[int(len(v) * 0.05)]
                wandb.log(stats, step=i)
                continue

            print("START DECODING...")
            t1 = time.time()
            # Note: We aren't interested in BEP reversal, so just propagate public_message_token
            decoded_message_bit, dec_stats = decoder.decode(public_message_str=public_message_str,
                                                            public_message_token=public_message_token,
                                                            private_message_bitlen=
                                                            len(private_message_bit),  # otherwise need header!
                                                            context=context,
                                                            dehumanify=False)
            end = time.time() - t1
            dec_stats["dec_t_wall"] = end
            wandb.log(dec_stats, step=i)

            # add to aggregate statistics
            for k, v in dec_stats.items():
                stats_traj[k].append(v)

            if cfg.method == "imec":
                decoded_message_bit_masked = deepcopy(decoded_message_bit)  # For DBG
                decoded_message_bit = remove_random_mask(decoded_message_bit, **mask_cfg)

            # Now reconstruct message
            reconstructed_message_bit = None
            reconstructed_message_str = None
            if cfg.message_mode == "randombits":
                reconstructed_message_bit = decoded_message_bit[:len(private_message_bit)]
                if cfg.verbose:
                    print("private message:        `{}` \n reconstructed message: `{}`".format(private_message_bit,
                                                                                               reconstructed_message_bit))
                if private_message_bit != reconstructed_message_bit:
                    diff_bits = 0
                    for pm, rm in zip(private_message_bit, reconstructed_message_bit):
                        if pm != rm:
                            diff_bits += 1
                    success_rate = 1.0 - float(diff_bits) / float(len(private_message_bit))
                    if cfg.verbose:
                        print("{}Decoding ERROR (proc #{}):{} rate: {}".format(bcolors.WARNING,
                                                                               _id, bcolors.ENDC, success_rate))
                    # wandb.log_artifact(e_str)
                    wandb.log({"success": success_rate}, step=i)
                    success_rates.append(success_rate)
                    wandb.log({"agg_success/mean": np.array(success_rates).mean()}, step=i)
                    wandb.log({"agg_success/std": np.array(success_rates).std()}, step=i)
                    wandb.log({"agg_success/80": np.sort(np.array(success_rates))[int(len(success_rates)*0.8)]}, step=i)
                    wandb.log({"agg_success/20": np.sort(np.array(success_rates))[int(len(success_rates)*0.2)]}, step=i)
                    wandb.log({"agg_success/95": np.sort(np.array(success_rates))[int(len(success_rates)*0.95)]}, step=i)
                    wandb.log({"agg_success/05": np.sort(np.array(success_rates))[int(len(success_rates)*0.05)]}, step=i)
                else:
                    print("{}Decoding SUCCESS (proc #{}):{}".format(bcolors.OKGREEN, _id, bcolors.ENDC))
                    wandb.log({"success": 1.0}, step=i)
                    success_rates.append(1.0)
                    wandb.log({"agg_success/mean": np.array(success_rates).mean()}, step=i)
                    wandb.log({"agg_success/std": np.array(success_rates).std()}, step=i)
                    wandb.log({"agg_success/80": np.sort(np.array(success_rates))[int(len(success_rates)*0.8)]}, step=i)
                    wandb.log({"agg_success/20": np.sort(np.array(success_rates))[int(len(success_rates)*0.2)]}, step=i)
                    wandb.log({"agg_success/95": np.sort(np.array(success_rates))[int(len(success_rates)*0.95)]}, step=i)
                    wandb.log({"agg_success/05": np.sort(np.array(success_rates))[int(len(success_rates)*0.05)]}, step=i)


            else:
                reconstructed_message_bit = decoded_message_bit
                reconstructed_message_str = bit2str(decoded_message_bit,
                                                    use_arithmetic_coding=use_arithmetic_coding,
                                                    medium=medium)[:-5]  # remove "<eos>"
                if cfg.verbose:
                    print("BITS:")
                    print("private message:       `{}` \n reconstructed message: `{}`".format(private_message_bit,
                                                                                              reconstructed_message_bit))
                    print("STRING:")
                    print("private message:       `{}` \n reconstructed message: `{}`".format(private_message_str,
                                                                                              reconstructed_message_str))

                if reconstructed_message_str != private_message_str:
                    if cfg.verbose:
                        print("{}Decoding ERROR{}".format(bcolors.WARNING, bcolors.ENDC))
                    wandb.log({"success": 0.0}, step=i)
                    success_rates.append(0.0)
                    wandb.log({"agg_success/mean": np.array(success_rates).mean()}, step=i)
                    wandb.log({"agg_success/std": np.array(success_rates).std()}, step=i)
                    wandb.log({"agg_success/80": np.sort(np.array(success_rates))[int(len(success_rates)*0.8)]}, step=i)
                    wandb.log({"agg_success/20": np.sort(np.array(success_rates))[int(len(success_rates)*0.2)]}, step=i)
                    wandb.log({"agg_success/95": np.sort(np.array(success_rates))[int(len(success_rates)*0.95)]}, step=i)
                    wandb.log({"agg_success/05": np.sort(np.array(success_rates))[int(len(success_rates)*0.05)]}, step=i)
                    # wandb.log_artifact(e_str)
                else:
                    print("{}Decoding SUCCESS{}".format(bcolors.OKGREEN, bcolors.ENDC))
                    wandb.log({"success": 1.0}, step=i)
                    success_rates.append(1.0)
                    wandb.log({"agg_success/mean": np.array(success_rates).mean()}, step=i)
                    wandb.log({"agg_success/std": np.array(success_rates).std()}, step=i)
                    wandb.log({"agg_success/80": np.sort(np.array(success_rates))[int(len(success_rates)*0.8)]}, step=i)
                    wandb.log({"agg_success/20": np.sort(np.array(success_rates))[int(len(success_rates)*0.2)]}, step=i)
                    wandb.log({"agg_success/95": np.sort(np.array(success_rates))[int(len(success_rates)*0.95)]}, step=i)
                    wandb.log({"agg_success/05": np.sort(np.array(success_rates))[int(len(success_rates)*0.05)]}, step=i)

            t_wall = time.time() - t0
            wandb.log({"t_wall_tot": t_wall}, step=i)

            # update aggregate statistics
            for k, v in stats_traj.items():
                stats["agg_" + k + "/mean"] = np.array(v).mean()
                stats["agg_" + k + "/std"] = np.array(v).std()
                stats["agg_" + k + "/80"] = np.sort(np.array(v))[int(len(v)*0.8)]
                stats["agg_" + k + "/20"] = np.sort(np.array(v))[int(len(v)*0.2)]
                stats["agg_" + k + "/95"] = np.sort(np.array(v))[int(len(v)*0.95)]
                stats["agg_" + k + "/05"] = np.sort(np.array(v))[int(len(v)*0.05)]


            wandb.log(stats, step=i)

            print("################ Finished iteration: {:d}".format(i))

        except Exception as e:
            print("{}FATAL ERROR (proc #{}):{}{}".format(bcolors.WARNING, _id, str(e), bcolors.ENDC))
            print(traceback.format_exc())
            print("Recovering...")

    print("RUN {} STOPPED!".format(_id))
    pass


class text_generator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.gen = Wikitext103Generator(seed=cfg.seed)

    def sample(self):
        return next(self.gen)["content"]


if __name__ == "__main__":
    set_start_method('spawn')

    # Retrieve config
    cfg_dct = parse_args()
    CfgTuple = namedtuple('CfgTuple', cfg_dct)
    cfg = CfgTuple(**cfg_dct)

    if cfg_dct["seed"] == -1:
        cfg_dct["seed"] = int.from_bytes(os.urandom(4), sys.byteorder)
        print("SEED set to {}".format(cfg_dct["seed"]))

    # Init wandb
    proc_id = uuid.uuid4().hex[:6]
    cfg_dct["wandb_dct"] = {"project": cfg_dct.get("wandb_project", "PLEASE SET"),
                            "entity": cfg_dct.get("wandb_entity", "PLEASE SET"),
                            "group": cfg_dct["group_name"],
                            "name": cfg_dct["name"],
                            "config": deepcopy(cfg_dct)}
    wandb.init(**cfg_dct["wandb_dct"])

    #####################################################
    # FREEZE CONFIG
    #####################################################

    CfgTuple = namedtuple('CfgTuple', cfg_dct)
    cfg = CfgTuple(**cfg_dct)

    # set up model, enc, and text_generator
    enc, model = get_model(seed=cfg.seed,
                           model_name=cfg.model_name,
                           device=cfg.model_device)
    context_generator = text_generator(cfg)

    run(cfg,
        model=model,
        enc=enc,
        context_generator=context_generator,
        _id=proc_id
        )

    print("Finished!")
    pass
