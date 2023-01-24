import numpy as np
import os
from scipy.stats import chisquare
import torch as th
import torch.nn.functional as F

from utils import get_model, is_sent_finish, limit_past, kl2, entropy2

class Medium:

    def __init__(self):
        pass

    def reset(self, context=None):
        if context is None:
            context = self._sample_context()
        return context

    def step(self, action):
        pass

class ActionLabelException(Exception):
    """Exception raised for errors in the dehumanify function.

    Attributes:
        pre -- pre string
        after -- after string
        messages -- message explanation
    """

    def __init__(self, action_labels, msg_token, msg_tokens, message):
        self.action_labels = action_labels
        self.msg_token = msg_token
        self.msg_tokens = msg_tokens
        message += str(self)
        super().__init__(message)

    def __str__(self):
        msg = "\n action_labels: ||{}|| \n msg_token: ||{}|| \n msg_tokens: ||{}||".format(self.action_labels,
                                                                                           self.msg_token,
                                                                                           self.msg_tokens)
        return msg


class METEORMedium(Medium):

    def __init__(self, **kwargs):
        super().__init__()
        self.prev = None
        self.device = kwargs.get("device", "cpu")
        self.seed = kwargs.get("seed", None)
        if self.seed is None:
            random_data = os.urandom(4)
            self.seed = int.from_bytes(random_data, byteorder="big")
            self.seed = int.from_bytes(random_data, byteorder="big")
            print("METEORMedium random seed was set to {}".format(self.seed))

        self.model_name = kwargs.get("model_name", "gpt2-large")  # "gpt2-medium")
        self.logit_temp = kwargs.get("logit_temp", 1.0)
        self.model = kwargs.get("model", None)
        self.enc = kwargs.get("enc", None)
        if self.model is None:
            self.enc, self.model = get_model(self.seed, self.model_name, self.device)

        self.output = []
        self.log_probs = None
        self.indices = None
        self.probs_top_k = kwargs.get("probs_top_k", 0)
        self.entropy_loss_threshold = kwargs.get("entropy_loss_threshold", 0)
        self.precision = kwargs.get("precision", 0)
        self.temp = kwargs.get("temp", 1.0)

    def encode_context(self, raw_text):
        context_tokens = [self.enc.encoder['<|endoftext|>']]
        context_tokens += self.enc.encode(raw_text)
        return context_tokens

    def reset(self, context=None):
        self.past = None
        self.output = []
        if context is None:
            context = self._sample_context()
        context_tokens = self.encode_context(context)
        context = th.tensor(context_tokens[-1022:], device=self.device, dtype=th.long)
        self.prev = context
        return self.step(None)

    def _sample_context(self):
        assert False, "Not yet implemented!"

    def step(self, action):
        info = {}
        if action is not None:
            if action == -1:
                raise Exception("Action -1 should never be selected!")
            self.prev = th.zeros(1, device="cpu", dtype=th.int32) + action.to("cpu")
            self.output.append(action.item())
            info = {"end_of_text": False,
                    "end_of_context": False,
                    "end_of_line": is_sent_finish(action.item(),
                                                  self.enc)}  # End of context / EOL status detection here
        with th.no_grad():
            logits, past = self.model(self.prev.unsqueeze(0).to(self.device), past=[p.to(self.device) for p in self.past] if self.past is not None else None)
            self.past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_all = logits.clone()
            probs_all = F.softmax(logits / self.temp, dim=0)
            orig_entropy = entropy2(probs_all.cpu().numpy())

            criteria = []
            if self.entropy_loss_threshold:
                ent = -th.log2(probs_all) * probs_all
                ent[probs_all == 0.0] = 0.0
                ent_full = ent.sum()
                cum_ent = th.cumsum(ent, dim=0)
                i = 0
                while i < cum_ent.shape[0]:
                    if cum_ent[i] / ent_full >= self.entropy_loss_threshold:
                        break
                    i += 1
                criteria.append(i)
                info["k_elt"] = i
            if self.probs_top_k:
                criteria.append(self.probs_top_k)
            if self.precision:
                # Meteor
                max_val = 2 ** self.precision
                cur_interval = [0, max_val]
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = max(2, (F.softmax(logits / self.temp, dim=0) < cur_threshold).nonzero()[0].item())
                criteria.append(k)
                info["k_at"] = k
            k = min(criteria) if len(criteria) else None  # Select the strongest criterium
            info["k"] = k
            logits = logits[:k]
            indices = indices[:k]
            self.probs = F.softmax(logits / self.temp, dim=0)
            info["logits"] = logits.cpu().numpy().astype(np.longdouble)
            info["log_probs"] = F.log_softmax(logits / self.temp, dim=0).cpu().numpy().astype(np.longdouble)
            info["log_probs_T=1"] = F.log_softmax(logits, dim=0).cpu().numpy().astype(np.longdouble)
            info["indices"] = indices.cpu().numpy()
            self.action_labels = indices

            trunc_entropy = entropy2(self.probs.cpu().numpy())
            info["medium_entropy_raw"] = orig_entropy
            info["medium_entropy"] = trunc_entropy

        self.info = info
        return self.probs.cpu().numpy().astype(np.longdouble), self.info

    def get_output(self, clean_up=False):
        """
         Returns the actual output of the session.
         :param clean_up: If True, then we modify the output such that it finishes on an EoL.
         :return:
         """
        if clean_up:
            # Ensure the last word in output is EOL.
            for i, o in enumerate(reversed(self.output)):
                if o == "EOL":  # TODO!
                    self.output = self.output[:-i]
                    break
        return self.output

    def humanify(self, output: list):
        """
        Converts a list of outputs into a human-readable output
        :return:
        """
        return "".join([self.enc.decode([o]) for o in output])

class RandomMedium(METEORMedium):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev = None
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", None)
        if self.seed is None:
            random_data = os.urandom(4)
            self.seed = int.from_bytes(random_data, byteorder="big")
            self.seed = int.from_bytes(random_data, byteorder="big")
            print("METEORMedium random seed was set to {}".format(self.seed))

        self.model_name = kwargs.get("model_name", "gpt2-large")  # "gpt2-medium")
        self.logit_temp = kwargs.get("logit_temp", 1.0)
        self.model = kwargs.get("model", None)
        self.enc = kwargs.get("enc", None)
        if self.model is None:
            self.enc, self.model = get_model(self.seed, self.model_name, self.device)

        self.output = []
        self.log_probs = None
        self.indices = None
        self.probs_top_k = kwargs.get("probs_top_k", 0)
        self.entropy_loss_threshold = kwargs.get("entropy_loss_threshold", 0)
        self.precision = kwargs.get("precision", 0)
        self.temp = kwargs.get("temp", 1.0)

    def step(self, action):
        info = {}
        if action is not None:
            self.prev = th.zeros(1, device=self.device, dtype=th.int32) + action.to(self.device)
            self.output.append(action.item())
            info = {"end_of_text": False,
                    "end_of_context": False,
                    "end_of_line": is_sent_finish(action.item(),
                                                  self.enc)}  # End of context / EOL status detection here
        with th.no_grad():
            logits, past = th.ones((1, 213, 50257), device="cpu"), None
            if past is not None:
                self.past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            probs_all = F.softmax(logits / self.temp, dim=0)
            orig_entropy = entropy2(probs_all.cpu().numpy())

            criteria = []
            if self.entropy_loss_threshold:
                ent = -th.log2(probs_all) * probs_all
                ent[probs_all == 0.0] = 0.0
                ent_full = ent.sum()
                cum_ent = th.cumsum(ent, dim=0)
                i = 0
                while i < cum_ent.shape[0]:
                    if cum_ent[i] / ent_full >= self.entropy_loss_threshold:
                        break
                    i += 1
                criteria.append(i)
                info["k_elt"] = i
            if self.precision:
                # Meteor
                max_val = 2 ** self.precision
                cur_interval = [0, max_val]
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = max(2, (F.softmax(logits / self.temp, dim=0) < cur_threshold).nonzero()[0].item())
                criteria.append(k)
                info["k_at"] = k
            if self.probs_top_k:
                criteria.append(self.probs_top_k)
            k = min(criteria) if len(criteria) else None  # Select the strongest criterium
            # print("Effective k: {} out of {}".format(k, criteria))
            info["k"] = k
            logits = logits[:k]
            indices = indices[:k]
            self.probs = F.softmax(logits / self.temp, dim=0)
            info["logits"] = logits.cpu().numpy().astype(np.longdouble)
            info["log_probs"] = F.log_softmax(logits / self.temp, dim=0).cpu().numpy().astype(np.longdouble)
            info["log_probs_T=1"] = F.log_softmax(logits, dim=0).cpu().numpy().astype(np.longdouble)
            info["indices"] = indices.cpu().numpy()
            self.action_labels = indices
            trunc_entropy = entropy2(self.probs.cpu().numpy().astype(np.longdouble))
            info["medium_entropy_raw"] = orig_entropy
            info["medium_entropy"] = trunc_entropy

        self.info = info
        return self.probs.cpu().numpy().astype(np.longdouble), self.info

class VariRandomMedium(METEORMedium):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev = None
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", None)
        if self.seed is None:
            random_data = os.urandom(4)
            self.seed = int.from_bytes(random_data, byteorder="big")
            self.seed = int.from_bytes(random_data, byteorder="big")
            print("VariRandomMedium random seed was set to {}".format(self.seed))

        self.model_name = kwargs.get("model_name", "gpt2-large")  # "gpt2-medium")
        self.logit_temp = kwargs.get("logit_temp", 1.0)
        self.model = kwargs.get("model", None)
        self.enc = kwargs.get("enc", None)
        if self.model is None:
            self.enc, self.model = get_model(self.seed, self.model_name, self.device)

        self.output = []
        self.log_probs = None
        self.indices = None
        self.probs_top_k = kwargs.get("probs_top_k", 0)
        self.entropy_loss_threshold = kwargs.get("entropy_loss_threshold", 0)
        self.precision = kwargs.get("precision", 0)
        self.temp = kwargs.get("temp", 1.0)

    def reset(self, context=None):
        self.past = None
        self.output = []
        if context is None:
            context = self._sample_context()
        context_tokens = self.encode_context(context)
        context = th.tensor(context_tokens[-1022:], device=self.device, dtype=th.long)
        self.prev = context

        # Set distribution parameters for the trajectory
        # self.g_cpu = th.Generator()
        from numpy.random import default_rng
        rng = default_rng(context.tolist())
        probs = th.from_numpy(rng.uniform(rng.random(self.probs_top_k))).double()
        self.probs = probs / probs.sum()

        self.hist_uniform = np.zeros((self.probs_top_k,))
        self.hist_sampled = np.zeros((self.probs_top_k,))
        self.n_hist_samples = 0

        return self.step(None)

    def step(self, action):
        info = {}
        if action is not None:
            self.prev = th.zeros(1, device=self.device, dtype=th.int32) + action.to(self.device)
            self.output.append(action.item())
            info = {"end_of_text": False,
                    "end_of_context": False,
                    "end_of_line": is_sent_finish(action.item(),
                                                  self.enc)}  # End of context / EOL status detection here

            # add to histogram
            self.hist_sampled[action.item()] += 1
            uniform_action = th.multinomial(self.probs, 1).item()
            self.hist_uniform[uniform_action] += 1
            self.n_hist_samples += 1
            info["kl(sampled|true)"] = kl2(th.from_numpy(self.hist_sampled/np.sum(self.hist_sampled)),
                                           self.probs.cpu())
            info["kl(uniform|true)"] = kl2(th.from_numpy(self.hist_uniform/np.sum(self.hist_uniform),
                                           self.probs.cpu().numpy()))
            info["kl(sampled|uniform)"] = kl2(th.from_numpy(self.hist_sampled/np.sum(self.hist_sampled)),
                                              th.from_numpy(self.hist_uniform/np.sum(self.hist_uniform)))
            info["kl(uniform|sampled)"] = kl2(th.from_numpy(self.hist_uniform/np.sum(self.hist_uniform)),
                                              th.from_numpy(self.hist_sampled/np.sum(self.hist_sampled)))
            info["chisquare_p(sampled|true)"] = chisquare(self.hist_sampled, self.probs*self.n_hist_samples).pvalue
            info["chisquare_p(uniform|true)"] = chisquare(self.hist_uniform, self.probs*self.n_hist_samples).pvalue


        indices = th.arange(self.probs_top_k).cpu()
        info["indices"] = indices.numpy()
        self.action_labels = indices
        trunc_entropy = entropy(self.probs.squeeze().cpu().numpy().astype(np.longdouble), 2)
        info["medium_entropy_raw"] = trunc_entropy
        info["medium_entropy"] = trunc_entropy
        info["log_probs"] = th.log(self.probs).numpy()
        info["log_probs_T=1"] = info["log_probs"]

        self.info = info
        return self.probs.cpu().numpy().astype(np.longdouble), self.info

import torchaudio
class WaveRNNMedium(Medium):

    def __init__(self, **kwargs):
        self.prev = None
        self.device = kwargs.get("device", "cpu")
        self.seed = kwargs.get("seed", None)
        if self.seed is None:
            random_data = os.urandom(4)
            self.seed = int.from_bytes(random_data, byteorder="big")
            self.seed = int.from_bytes(random_data, byteorder="big")
            print("METEORMedium random seed was set to {}".format(self.seed))

        self.model_name = kwargs.get("model_name", "gpt2-large")  # "gpt2-medium")
        self.logit_temp = 1.0 #kwargs.get("logit_temp", 1.0)
        self.model = kwargs.get("model", None)
        self.enc = kwargs.get("enc", None)
        if self.model is None:
            self.enc, self.model = get_model(self.seed, self.model_name, self.device)

        self.output = []
        self.log_probs = None
        self.indices = None
        self.probs_top_k = kwargs.get("probs_top_k", 0)
        self.entropy_loss_threshold = kwargs.get("entropy_loss_threshold", 0)
        self.precision = kwargs.get("precision", 0)
        self.temp = 1.0 #kwargs.get("temp", 1.0)

        # Create WaveRNN model
        from torchaudio.models import WaveRNN
        self.wavernn = WaveRNN(upsample_scales=[5, 5, 11],
                               n_classes=256,
                               hop_length=275,
                               n_res_block=10,
                               n_rnn=512,
                               n_fc=512,
                               kernel_size=5,
                               n_freq=80,
                               n_hidden=128,
                               n_output=128).to(self.device)
        url = f"https://download.pytorch.org/torchaudio/models/wavernn_10k_epochs_8bits_ljspeech.pth"
        state_dict = th.hub.load_state_dict_from_url(url, **{})
        self.wavernn.load_state_dict(state_dict)
        self.wavernn.eval()

    def encode_context(self, raw_text):
        context_tokens = [self.enc.encoder['<|endoftext|>']]
        context_tokens += self.enc.encode(raw_text)
        return context_tokens

    def reset(self, context=None):
        self.past = None
        self.output = []
        if context is None:
            context = self._sample_context()

        # Spectogram reset
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        processor = bundle.get_text_processor()
        bundle._tacotron2_params["decoder_max_step"] = 10000
        tacotron2 = bundle.get_tacotron2().to(self.device)
        with th.inference_mode():
            processed, lengths = processor(context)
            processed = processed.to(self.device)
            lengths = lengths.to(self.device)
            specgram, spec_lengths, _ = tacotron2.infer(processed, lengths)
            self.lengths = spec_lengths

            # WaveRNN reset
            device = self.device
            dtype = specgram.dtype

            self.specgram = th.nn.functional.pad(specgram, (self.wavernn._pad, self.wavernn._pad)).to(self.device)
            self.specgram, aux = self.wavernn.upsample(self.specgram)
            if self.lengths is not None:
                self.lengths = self.lengths * self.wavernn.upsample.total_scale

            self.output = []
            self.b_size, _, self.seq_len = specgram.size()

            self.h1 = th.zeros((1, self.b_size, self.wavernn.n_rnn), device=device, dtype=dtype)
            self.h2 = th.zeros((1, self.b_size, self.wavernn.n_rnn), device=device, dtype=dtype)
            self.x = th.zeros((self.b_size, 1), device=device, dtype=dtype)

            self.aux_split = [aux[:, self.wavernn.n_aux * i : self.wavernn.n_aux * (i + 1), :] for i in range(4)]
            self.i = 0

        return self.step(None)

    def _sample_context(self):
        assert False, "Not yet implemented!"


    def step(self, action):
        info = {}
        if action is not None:
            if action.item() == -1:
                raise Exception("Action -1 should never be selected!")
            self.prev = th.zeros(1, device="cpu", dtype=th.int32) + action.to("cpu")
            self.x = 2 * (action.to(self.device).unsqueeze(0).unsqueeze(0) if not len(action.shape) else action.to(self.device).unsqueeze(0) )/ (2 ** self.wavernn.n_bits - 1.0) - 1.0

            self.output.append(self.x.item())
            info = {"end_of_text": False,
                    "end_of_context": False,
                    "end_of_line": is_sent_finish(action.item(),
                                                  self.enc)}  # End of context / EOL status detection here
        with th.no_grad():

            if self.i >= self.seq_len:
                raise Exception("WaveRNN medium ran out of spectogram!")

            # WaveRNN start
            m_t = self.specgram[:, :, self.i]

            a1_t, a2_t, a3_t, a4_t = [a[:, :, self.i] for a in self.aux_split]

            x = th.cat([self.x, m_t, a1_t], dim=1)
            x = self.wavernn.fc(x)
            _, h1 = self.wavernn.rnn1(x.unsqueeze(1), self.h1)

            x = x + h1[0]
            inp = th.cat([x, a2_t], dim=1)
            _, h2 = self.wavernn.rnn2(inp.unsqueeze(1), self.h2)

            x = x + h2[0]
            x = th.cat([x, a3_t], dim=1)
            x = F.relu(self.wavernn.fc1(x))

            x = th.cat([x, a4_t], dim=1)
            x = F.relu(self.wavernn.fc2(x))

            logits = self.wavernn.fc3(x)
            logits, indices = logits[0, :].sort(descending=True)

            self.i += 1

            # WaveRNN stop
            probs_all = F.softmax(logits / self.temp, dim=0)
            orig_entropy = entropy2(probs_all.cpu().numpy())

            k = None
            logits = logits[:k]
            indices = indices[:k]
            self.probs = F.softmax(logits / self.temp, dim=0)
            info["logits"] = logits.cpu().numpy().astype(np.longdouble)
            info["log_probs"] = F.log_softmax(logits / self.temp, dim=0).cpu().numpy().astype(np.longdouble)
            info["log_probs_T=1"] = F.log_softmax(logits, dim=0).cpu().numpy().astype(np.longdouble)
            info["indices"] = indices.cpu().numpy()
            self.action_labels = indices
            trunc_entropy = entropy2(self.probs.cpu().numpy())
            info["medium_entropy_raw"] = orig_entropy
            info["medium_entropy"] = trunc_entropy

        self.info = info
        return self.probs.cpu().numpy().astype(np.longdouble)/self.probs.cpu().numpy().astype(np.longdouble).sum(), self.info

    def get_output(self, clean_up=False):
        """
         Returns the actual output of the session.
         :param clean_up: If True, then we modify the output such that it finishes on an EoL.
         :return:
         """
        if clean_up:
            # Ensure the last word in output is EOL.
            for i, o in enumerate(reversed(self.output)):
                if o == "EOL":  # TODO!
                    self.output = self.output[:-i]
                    break
        return self.output

    def humanify(self, output: list):
        """
        Converts a list of outputs into a human-readable output
        :return:
        """
        return "".join([self.enc.decode([o]) for o in output])
