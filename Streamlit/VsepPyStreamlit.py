# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:29:24 2021

@author: Daniel Perez, Ernesto Horne
"""
import streamlit as st
import pygame
from PIL import Image
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import os
import IPython.display as ipd
import librosa
import librosa.display
from pydub import AudioSegment
from pyvis import network as net
from IPython.core.display import display, HTML
import streamlit.components.v1 as components

import argparse
import sys
from pathlib import Path
import subprocess

import julius
import torch as th
import torchaudio as ta
from torch import nn

#################################################################################################### UTILS

import errno
import functools
import hashlib
import inspect
import io
import os
import random
import socket
import tempfile
import warnings
import zlib
from contextlib import contextmanager

from diffq import UniformQuantizer, DiffQuantizer
import torch as th
import tqdm
from torch import distributed
from torch.nn import functional as F


def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def average_metric(metric, count=1.):
    """
    Average `metric` which should be a float across all hosts. `count` should be
    the weight for this particular host (i.e. number of examples).
    """
    metric = th.tensor([count, count * metric], dtype=th.float32, device='cuda')
    distributed.all_reduce(metric, op=distributed.ReduceOp.SUM)
    return metric[1].item() / metric[0].item()


def free_port(host='', low=20000, high=40000):
    """
    Return a port number that is most likely free.
    This could suffer from a race condition although
    it should be quite rare.
    """
    sock = socket.socket()
    while True:
        port = random.randint(low, high)
        try:
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        return port


def sizeof_fmt(num, suffix='B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def human_seconds(seconds, display='.2f'):
    """
    Given `seconds` seconds, return human readable duration.
    """
    value = seconds * 1e6
    ratios = [1e3, 1e3, 60, 60, 24]
    names = ['us', 'ms', 's', 'min', 'hrs', 'days']
    last = names.pop(0)
    for name, ratio in zip(names, ratios):
        if value / ratio < 0.3:
            break
        value /= ratio
        last = name
    return f"{format(value, display)} {last}"


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model(model, mix, shifts=None, split=False,
                overlap=0.25, transition_power=1., progress=False):
    """
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
    """
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    device = mix.device
    channels, length = mix.shape
    if split:
        out = th.zeros(len(model.sources), channels, length, device=device)
        sum_weight = th.zeros(length, device=device)
        segment = model.segment_length
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / model.samplerate
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = th.cat([th.arange(1, segment // 2 + 1),
                         th.arange(segment - segment // 2, 0, -1)]).to(device)
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max())**transition_power
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            chunk_out = apply_model(model, chunk, shifts=shifts)
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset:offset + segment] += weight[:chunk_length]
            offset += segment
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    elif shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted)
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        return out
    else:
        valid_length = model.valid_length(length)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length)
        with th.no_grad():
            out = model(padded_mix.unsqueeze(0))[0]
        return center_trim(out, length)


@contextmanager
def temp_filenames(count, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def get_quantizer(model, args, optimizer=None):
    quantizer = None
    if args.diffq:
        quantizer = DiffQuantizer(
            model, min_size=args.q_min_size, group_size=8)
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        quantizer = UniformQuantizer(
                model, bits=args.qat, min_size=args.q_min_size)
    return quantizer


def load_model(path, strict=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        package = th.load(load_from, 'cpu')

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]
    training_args = package["training_args"]
    quantizer = get_quantizer(model, training_args)

    set_state(model, quantizer, state)
    return model


def get_state(model, quantizer, half=False):
    if quantizer is None:
        dtype = th.half if half else None
        state = {k: p.data.to(device='cpu', dtype=dtype) for k, p in model.state_dict().items()}
    else:
        state = quantizer.get_quantized_state()
        buf = io.BytesIO()
        th.save(state, buf)
        state = {'compressed': zlib.compress(buf.getvalue())}
    return state


def set_state(model, quantizer, state):
    if quantizer is None:
        model.load_state_dict(state)
    else:
        buf = io.BytesIO(zlib.decompress(state["compressed"]))
        state = th.load(buf, "cpu")
        quantizer.restore_quantized_state(state)

    return state


def save_state(state, path):
    buf = io.BytesIO()
    th.save(state, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())


def save_model(model, quantizer, training_args, path):
    args, kwargs = model._init_args_kwargs
    klass = model.__class__

    state = get_state(model, quantizer, half=training_args.half)

    save_to = path
    package = {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': training_args,
    }
    th.save(package, save_to)


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)
    return __init__
####################################################################################################MODELS

import math

import julius
from torch import nn



class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    @capture_init
    def __init__(self,
                 sources,
                 audio_channels=2,
                 channels=64,
                 depth=6,
                 rewrite=True,
                 glu=True,
                 rescale=0.1,
                 resample=True,
                 kernel_size=8,
                 stride=4,
                 growth=2.,
                 lstm_layers=2,
                 context=3,
                 normalize=False,
                 samplerate=44100,
                 segment_length=4 * 10 * 44100):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            resample_input (bool): upsample x2 the input and downsample /2 the output.
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment_length (int): stored as meta information for easing
                future evaluations of the model. Length of the segments on which
                the model was trained.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment_length = segment_length

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        in_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.

        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        if self.resample:
            length *= 2
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        x = mix

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
        else:
            mean = 0
            std = 1

        x = (x - mean) / (1e-5 + std)

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
        if self.lstm:
            x = self.lstm(x)
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x

#########################################################################################PRETRAINED
import logging

from diffq import DiffQuantizer
import torch.hub




logger = logging.getLogger(__name__)
ROOT = "https://dl.fbaipublicfiles.com/demucs/v3.0/"

PRETRAINED_MODELS = {
    'demucs': 'e07c671f',
    'demucs48_hq': '28a1282c',
    'demucs_extra': '3646af93',
    'demucs_quantized': '07afea75',
    'tasnet': 'beb46fac',
    'tasnet_extra': 'df3777b2',
    'demucs_unittest': '09ebc15f',
}

SOURCES = ["drums", "bass", "other", "vocals"]


def get_url(name):
    sig = PRETRAINED_MODELS[name]
    return ROOT + name + "-" + sig[:8] + ".th"


def is_pretrained(name):
    return name in PRETRAINED_MODELS


def load_pretrained(name):
    if name == "demucs":
        return demucs(pretrained=True)
    elif name == "demucs48_hq":
        return demucs(pretrained=True, hq=True, channels=48)
    elif name == "demucs_extra":
        return demucs(pretrained=True, extra=True)
    elif name == "demucs_quantized":
        return demucs(pretrained=True, quantized=True)
    elif name == "demucs_unittest":
        return demucs_unittest(pretrained=True)
    elif name == "tasnet":
        return tasnet(pretrained=True)
    elif name == "tasnet_extra":
        return tasnet(pretrained=True, extra=True)
    else:
        raise ValueError(f"Invalid pretrained name {name}")


def _load_state(name, model, quantizer=None):
    url = get_url(name)
    state = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
    set_state(model, quantizer, state)
    if quantizer:
        quantizer.detach()


def demucs_unittest(pretrained=True):
    model = Demucs(channels=4, sources=SOURCES)
    if pretrained:
        _load_state('demucs_unittest', model)
    return model


def demucs(pretrained=True, extra=False, quantized=False, hq=False, channels=64):
    if not pretrained and (extra or quantized or hq):
        raise ValueError("if extra or quantized is True, pretrained must be True.")
    model = Demucs(sources=SOURCES, channels=channels)
    if pretrained:
        name = 'demucs'
        if channels != 64:
            name += str(channels)
        quantizer = None
        if sum([extra, quantized, hq]) > 1:
            raise ValueError("Only one of extra, quantized, hq, can be True.")
        if quantized:
            quantizer = DiffQuantizer(model, group_size=8, min_size=1)
            name += '_quantized'
        if extra:
            name += '_extra'
        if hq:
            name += '_hq'
        _load_state(name, model, quantizer)
    return model


def tasnet(pretrained=True, extra=False):
    if not pretrained and extra:
        raise ValueError("if extra is True, pretrained must be True.")
    model = ConvTasNet(X=10, sources=SOURCES)
    if pretrained:
        name = 'tasnet'
        if extra:
            name = 'tasnet_extra'
        _load_state(name, model)
    return model



########################################################################################################### AUDIO 
import json
import subprocess as sp
from pathlib import Path

import julius
import numpy as np
import torch




def _read_info(path):
    stdout_data = sp.check_output([
        'ffprobe', "-loglevel", "panic",
        str(path), '-print_format', 'json', '-show_format', '-show_streams'
    ])
    return json.loads(stdout_data.decode('utf-8'))


class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None,
             temp_folder=None):
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.
            temp_folder (str or Path or None): temporary folder to use for decoding.


        """
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))

        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]

            sp.run(command, check=True)
            wavs = []
            for filename in filenames:
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels()).t()
                if channels is not None:
                    wav = convert_audio_channels(wav, channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels):
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)

############################################################################################################3 SEPARATE

import argparse
import sys
from pathlib import Path
import subprocess

import julius
import torch as th
import torchaudio as ta



def load_track(track, device, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels).to(device)
    except FileNotFoundError:
        errors['ffmpeg'] = 'Ffmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio_channels(wav, audio_channels)
            wav = wav.to(device)
            wav = julius.resample_frac(wav, sr, samplerate)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


def encode_mp3(wav, path, bitrate=320, samplerate=44100, channels=2, verbose=False):
    try:
        import lameenc
    except ImportError:
        print("Failed to call lame encoder. Maybe it is not installed? "
              "On windows, run `python.exe -m pip install -U lameenc`, "
              "on OSX/Linux, run `python3 -m pip install -U lameenc`, "
              "then try again.", file=sys.stderr)
        sys.exit(1)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav = wav.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


################################################################################################################STREAMLIT
def main():
    st.title('V-SepPy, split-mix-loop your music')


    st.sidebar.header('Menu\'')
    choice = st.sidebar.radio(" ",["V-SepPy" ,"Dataset" ,'Models', 'Demo'])
    
    
    if choice == 'V-SepPy':
        st.header('What you can do') #Space
        st.write(
            """
            Separate the instruments of your own track to obtain a high quality audio. Then mix it up with 
            other instruments from your favorite songs and if you like it loop until you cant hear ir anymore.
                  
""")
        st.header('Why you can do it') #Space
        st.write(
            """
            Thanks to demucs, a deep learning model that directly operates on the raw input waveform and 
            generates a waveform for each source, you can get the best source separation. Demucs is today 
            the state-of-the-art model for separating individual instruments from a music track.
                  
""")
    if choice == 'Dataset':
        st.header('\n') #Space
        st.write(
            """
            Our model is conditioned by the quality of the data used to train it (garbage in garbage out 
            principle); in this sense the data must be varied enough to apply the model to any type of music.
""")

        st.header('Dataset utilisé : MUSDB 18')
        st.write(
            """
            MUSDB18 est un ensemble de données de 150 pistes musicales complètes (durée totale d'environ 10h) de genres variés.
            Pour chaque piste, il fournit:
    
            - Le mixture
            - Tambours
            - Basse
            - Voix
            - Autres
            
            Comme son nom l'indique, la tige « autres » contient toutes les autres sources du mixture qui ne sont pas les tambours,
            la basse ou la voix (étiquetées « accompagnement » dans le schéma ci-dessous) :
""")
        image = Image.open('musdb18.png')

        st.image(image, caption='Illustration of the stems comprising the mixture in a MUSDB18 track. Source: https://sigsep.github.io/')
        st.write(
            """
            On retrouve les informations correspondant à chaque chanson dans le dataframe suivant :
""")
        df = pd.read_csv('musdb18.csv')
        st.write(df)
        
        st.header('Visualization de chaque stem d’une chanson dans le dataset MUSDB18')
        option = st.selectbox(
            "Choissisez le type d'affichage",
            ('None','Waveform', 'Spectrogramme'))
        
        if option =='None':
            pass
        
        if option == 'Waveform':
            path_mix = os.path.join( 'train', 'A Classic Education - NightOwl', 'mixture.wav')
            path_vocal = os.path.join( 'train', 'A Classic Education - NightOwl', 'vocals.wav')
            path_bass = os.path.join( 'train', 'A Classic Education - NightOwl', 'bass.wav')
            path_drums = os.path.join( 'train', 'A Classic Education - NightOwl', 'drums.wav')
            sigmix, ratemix = librosa.load(path_mix, sr=22000, duration=40)
            sigvocal, ratevocal = librosa.load(path_vocal, sr=22000, duration=40)
            sigbass, ratebass = librosa.load(path_bass, sr=22000, duration=40)
            sigdrums, ratedrums = librosa.load(path_drums, sr=22000, duration=40)
            
            st.header("Wavefrom (Amplitude = f(temps)) d'une chanson et ses stems ")
            
            fig= plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigmix, sr=22000, color='teal', alpha=0.5)
            plt.title('Mixture - A Classic Education - NightOw')
            plt.ylim(-0.7,0.7)
            plt.subplot(222)
            librosa.display.waveplot(sigvocal, sr=22000, color='teal', alpha=0.5)
            plt.title('Vocals - A Classic Education - NightOw')
            plt.ylim(-0.7,0.7)
            plt.subplot(223)
            librosa.display.waveplot(sigbass, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass - A Classic Education - NightOw')
            plt.ylim(-0.7,0.7)
            plt.subplot(224)
            librosa.display.waveplot(sigdrums, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums - A Classic Education - NightOw')
            plt.ylim(-0.7,0.7);
            st.pyplot(fig)
            
            st.write(
                """
                Pour cette chanson particulière (A Classic Education - NightOw), nous pouvons distinguer certaines caractéristiques 
                intrinsèques de chaque instrument, par exemple, l'amplitude de l'onde et la durée des paquets d'ondes.
                Globalement, la forme d'onde de la somme des instruments est égale à celle du morceau original. 
    
    """)
            fig= plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigmix, sr=22000, color='teal', alpha=0.5)
            plt.ylim(-0.7,0.7)
            plt.title('Mixture - A Classic Education - NightOw')
            plt.subplot(222)
            librosa.display.waveplot(sigvocal, sr=22000, color='teal', alpha=0.5)
            librosa.display.waveplot(sigbass, sr=22000, color='teal', alpha=0.5)
            librosa.display.waveplot(sigdrums, sr=22000, color='teal', alpha=0.5)
            plt.ylim(-0.7,0.7)
            plt.title('Vocals + Drums + Bass - A Classic Education - NightOw')
            st.pyplot(fig)
            
        if option == 'Spectrogramme':
            
            path_mix = os.path.join( 'train', 'A Classic Education - NightOwl', 'mixture.wav')
            path_vocal = os.path.join( 'train', 'A Classic Education - NightOwl', 'vocals.wav')
            path_bass = os.path.join( 'train', 'A Classic Education - NightOwl', 'bass.wav')
            path_drums = os.path.join( 'train', 'A Classic Education - NightOwl', 'drums.wav')
            
            sigmix, ratemix = librosa.load(path_mix, sr=22000, duration=40)
            sigvocal, ratevocal = librosa.load(path_vocal, sr=22000, duration=40)
            sigbass, ratebass = librosa.load(path_bass, sr=22000, duration=40)
            sigdrums, ratedrums = librosa.load(path_drums, sr=22000, duration=40)
            
            specmix = librosa.stft(sigmix)
            specvocal = librosa.stft(sigvocal)
            specbass = librosa.stft(sigbass)  
            specdrums = librosa.stft(sigdrums)
            
            spec_db_mix = librosa.amplitude_to_db(specmix, ref=np.max)
            spec_db_vocal = librosa.amplitude_to_db(specvocal, ref=np.max)
            spec_db_bass = librosa.amplitude_to_db(specbass, ref=np.max)
            spec_db_drums = librosa.amplitude_to_db(specdrums, ref=np.max)
            st.header("Spectrogramme (Amplitude = f(temps et fréquence) le colormap indique l’amplitude de spectre) d'une chanson et ses stems")
            
            fig= plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_mix, 
                         sr=100000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Mixture - A Classic Education - NightOw')

            plt.subplot(222)
            librosa.display.specshow(spec_db_vocal, 
                         sr=100000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Vocals - A Classic Education - NightOw')

            plt.subplot(223)
            librosa.display.specshow(spec_db_bass, 
                         sr=100000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Bass - A Classic Education - NightOw')

            plt.subplot(224)
            librosa.display.specshow(spec_db_drums, 
                         sr=100000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Drums - A Classic Education - NightOw');
            st.pyplot(fig)
            
            st.write(
                """
                Ici, nous pouvons aussi distinguer des caractéristiques de chaque instrument: La basse présente des fréquences 
                plus basses que le reste des instruments et on peut distinguer la dissipation sélective des hautes fréquences. 
                La voix présente des impulsions plus longues et la présence d'harmoniques peut être observée. 
                La batterie semble être hyper représentée dans l'amplitude de certaines fréquences, comme on peut le constater 
                en comparant le morceau original et la superposition. La chanson originale présente une plus grande continuité 
                dans les basses fréquences par rapport à la superposition.
 
    
    """)
            fig2 = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_mix, 
                                     sr=100000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Mixture - A Classic Education - NightOw')
            plt.subplot(222)
            librosa.display.specshow(spec_db_vocal, 
                                     sr=100000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            librosa.display.specshow(spec_db_bass, 
                                     sr=100000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            librosa.display.specshow(spec_db_drums, 
                                     sr=100000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Vocals + Drums + Bass - A Classic Education - NightOw')
            st.pyplot(fig2)
            
            
        st.header('Proportion des genres dans les voix des chansons du dataset MUSDB18')
        
        Homme = df['Voix'].value_counts()[0] / (df['Voix'].value_counts()[0] + df['Voix'].value_counts()[1] + df['Voix'].value_counts()[2]) 
        Femme = df['Voix'].value_counts()[1] / (df['Voix'].value_counts()[0] + df['Voix'].value_counts()[1] + df['Voix'].value_counts()[2])
        Mixte = df['Voix'].value_counts()[2] / (df['Voix'].value_counts()[0] + df['Voix'].value_counts()[1] + df['Voix'].value_counts()[2])
        
        fig = plt.figure(figsize=(6,6))
        sns.countplot(df['Voix'], palette="hls")
        plt.text(-0.1, 104, f'{100*round(Homme,4)}%', size= 'large' )
        plt.text(0.9, 38, f'{100*round(Femme,4)}%', size= 'large')
        plt.text(1.9, 11, f'{100*round(Mixte,4)}%', size= 'large')
        plt.ylabel('Quantite')
        plt.xlabel('Genre de voix')
        plt.title('Proportion de voix selon son genre');
        st.pyplot(fig)
        
        
        st.header('Visualization des genres des voix')
        op = st.selectbox(
            "Choissisez le type d'affichage",
            ('None', 'Waveforme', 'Spectrogramme'))
        
        path_vocalh = os.path.join( 'train', 'Triviul - Dorothy', 'vocals.wav')
        path_vocalf = os.path.join( 'train', 'Actions - One Minute Smile', 'vocals.wav')
        path_vocalm = os.path.join( 'train', 'Fergessen - Back From The Start', 'vocals.wav')


        if op == 'None':
            pass
        
        if op == 'Waveforme':
            sigvocalh, ratevocalh = librosa.load(path_vocalh, sr=22000, duration=30)
            sigvocalf, ratevocalf = librosa.load(path_vocalf, sr=22000, duration=30)
            sigvocalm, ratevocalm = librosa.load(path_vocalm, sr=22000, duration=30)

            fig= plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigvocalh, sr=22000, color='teal', alpha=0.5)
            plt.title('Voix de homme - Triviul - Dorothy')
            plt.ylim(-0.8,0.8)
            plt.xlim(5,30)
            plt.subplot(222)
            librosa.display.waveplot(sigvocalf, sr=22000, color='teal', alpha=0.5)
            plt.title('Voix de femme - Actions - One Minute Smile')
            plt.ylim(-0.8,0.8)
            plt.xlim(2,22)
            plt.subplot(223)
            librosa.display.waveplot(sigvocalm, sr=22000, color='teal', alpha=0.5)
            plt.title('Voix Mixte - Fergessen - Back From The Start')
            plt.ylim(-0.8,0.8)
            plt.xlim(6,30)
            st.pyplot(fig)
            st.write(
                """
                La seule différence que l'on peut observer est une légère augmentation des amplitudes de la voix 
                féminine par rapport à la voix masculine dans les hautes fréquences.  
    """)
        if op == 'Spectrogramme':
            
            sigvocalh, ratevocalh = librosa.load(path_vocalh, sr=22000, duration=30)
            sigvocalf, ratevocalf = librosa.load(path_vocalf, sr=22000, duration=30)
            sigvocalm, ratevocalm = librosa.load(path_vocalm, sr=22000, duration=30)
            
            # First, compute the spectrogram using the "short-time Fourier transform" (stft)
            specvocalh = librosa.stft(sigvocalh)
            specvocalf = librosa.stft(sigvocalf)
            specvocalm = librosa.stft(sigvocalm)
    
            # Scale the amplitudes according to the decibel scale
            spec_db_vocalh = librosa.amplitude_to_db(specvocalh, ref=np.max)
            spec_db_vocalf = librosa.amplitude_to_db(specvocalf, ref=np.max)
            spec_db_vocalm = librosa.amplitude_to_db(specvocalm, ref=np.max)

            fig= plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_vocalh, 
                         sr=22000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Voix de homme - Triviul - Dorothy')
            plt.xlim(6,30)
            plt.subplot(222)
            librosa.display.specshow(spec_db_vocalf, 
                         sr=22000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Voix de femme - Actions - One Minute Smile')

            plt.subplot(223)
            librosa.display.specshow(spec_db_vocalm, 
                         sr=22000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Voix Mixte - Fergessen - Back From The Start')
            plt.xlim(6,30);
            st.pyplot(fig)
            st.write(
                """
                La seule différence que l'on peut observer est une légère augmentation des amplitudes de la voix 
                féminine par rapport à la voix masculine dans les hautes fréquences.  
    """)
        
        st.header('Repartition des genres musicaux dans le dataset MUSDB18')
        
        fig = Figure()
        ax = fig.subplots()
        sns.countplot(y='Genre', data= df, palette="hls", order=['Rock','Alternative','Pop','Country','Heavy metal','Hip-hop','Reggae','Electro'], ax=ax)
        ax.set_xlabel('Quantite')
        ax.set_ylabel('Genre')
        st.pyplot(fig)
        
        st.header('Drums par genre musical')
        opt = st.selectbox(
            "Choissisez le type d'affichage",
            ('None', 'Waveforme', 'Spectrogram'))
        
    
        if opt == 'None':
            pass
        
        if opt == 'Waveforme':
            path_drumsr = os.path.join( 'train', 'The So So Glos - Emergency', 'drums.wav')
            path_drumsa = os.path.join( 'train', 'Tim Taler - Stalker', 'drums.wav')
            path_drumsp = os.path.join( 'test', 'Cristina Vane - So Easy', 'drums.wav')
            path_drumsc = os.path.join( 'train', 'James May - Dont Let Go', 'drums.wav')
            path_drumsh = os.path.join( 'train', 'Wall Of Death - Femme', 'drums.wav')
            path_drumshh = os.path.join( 'train', 'Grants - PunchDrunk', 'drums.wav')
            path_drumsre = os.path.join( 'train', 'Music Delta - Reggae', 'drums.wav')
            path_drumse = os.path.join( 'test', 'PR - Happy Daze', 'drums.wav')
    
            sigdrumsr, ratevocalh = librosa.load(path_drumsr, sr=22000, duration=30)
            sigdrumsa, ratevocalf = librosa.load(path_drumsa, sr=22000,  duration=30)
            sigdrumsp, ratevocalm = librosa.load(path_drumsp, sr=22000,  duration=30)
            sigdrumsc, ratevocalh = librosa.load(path_drumsc, sr=22000, duration=30)
            sigdrumsh, ratevocalf = librosa.load(path_drumsh, sr=22000, duration=30)
            sigdrumshh, ratevocalm = librosa.load(path_drumshh, sr=22000, duration=30)
            sigdrumsre, ratevocalf = librosa.load(path_drumsre, sr=22000,  duration=30)
            sigdrumse, ratevocalm = librosa.load(path_drumse, sr=22000, duration=30)
            
            fig = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigdrumsr, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Rock - The So So Glos - Emergency')
            plt.ylim(-0.8,0.8)
            plt.subplot(222)
            librosa.display.waveplot(sigdrumsa, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Alternative - Tim Taler - Stalker')
            plt.ylim(-0.8,0.8)
            plt.subplot(223)
            librosa.display.waveplot(sigdrumsp, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Pop - Cristina Vane - So Easy')
            plt.ylim(-0.8,0.8)
            plt.xlim(4)
            plt.subplot(224)
            librosa.display.waveplot(sigdrumsc, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Country - James May - Dont Let Go')
            plt.ylim(-0.8,0.8)
            plt.xlim(4);
            st.pyplot(fig)
            
            fig2 = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigdrumsh, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Heavymetal - Wall Of Death - Femme')
            plt.ylim(-0.8,0.8)
            plt.subplot(222)
            librosa.display.waveplot(sigdrumshh, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Hip-hop - Grants - PunchDrunk')
            plt.ylim(-0.8,0.8)
            plt.xlim(5)
            plt.subplot(223)
            librosa.display.waveplot(sigdrumsre, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Reggae - Music Delta - Reggae')
            plt.ylim(-0.8,0.8)
            plt.subplot(224)
            librosa.display.waveplot(sigdrumse, sr=22000, color='teal', alpha=0.5)
            plt.title('Drums Electro - PR - Happy Daze')
            plt.ylim(-0.8,0.8);
            st.pyplot(fig2)
            
            st.write(
                """
                Dans l'espace des ondes, il n'est pas evident de trouver des caracteristiques distinctives
                du tambour parmi tous les styles musicaux  
    """)
        
        if opt == 'Spectrogram':
            path_drumsr = os.path.join( 'train', 'The So So Glos - Emergency', 'drums.wav')
            path_drumsa = os.path.join( 'train', 'Tim Taler - Stalker', 'drums.wav')
            path_drumsp = os.path.join( 'test', 'Cristina Vane - So Easy', 'drums.wav')
            path_drumsc = os.path.join( 'train', 'James May - Dont Let Go', 'drums.wav')
            path_drumsh = os.path.join( 'train', 'Wall Of Death - Femme', 'drums.wav')
            path_drumshh = os.path.join( 'train', 'Grants - PunchDrunk', 'drums.wav')
            path_drumsre = os.path.join( 'train', 'Music Delta - Reggae', 'drums.wav')
            path_drumse = os.path.join( 'test', 'PR - Happy Daze', 'drums.wav')
    
            sigdrumsr, ratevocalh = librosa.load(path_drumsr, sr=22000, duration=30)
            sigdrumsa, ratevocalf = librosa.load(path_drumsa, sr=22000,  duration=30)
            sigdrumsp, ratevocalm = librosa.load(path_drumsp, sr=22000,  duration=30)
            sigdrumsc, ratevocalh = librosa.load(path_drumsc, sr=22000, duration=30)
            sigdrumsh, ratevocalf = librosa.load(path_drumsh, sr=22000, duration=30)
            sigdrumshh, ratevocalm = librosa.load(path_drumshh, sr=22000, duration=30)
            sigdrumsre, ratevocalf = librosa.load(path_drumsre, sr=22000,  duration=30)
            sigdrumse, ratevocalm = librosa.load(path_drumse, sr=22000, duration=30)
            # First, compute the spectrogram using the "short-time Fourier transform" (stft)
            specdrumsr = librosa.stft(sigdrumsr)
            specdrumsa = librosa.stft(sigdrumsa)
            specdrumsp = librosa.stft(sigdrumsp)
            specdrumsc = librosa.stft(sigdrumsc)
            specdrumsh = librosa.stft(sigdrumsh)
            specdrumshh = librosa.stft(sigdrumshh)
            specdrumsre = librosa.stft(sigdrumsre)
            specdrumse = librosa.stft(sigdrumse)


            # Scale the amplitudes according to the decibel scale
            spec_db_drumsr = librosa.amplitude_to_db(specdrumsr, ref=np.max)
            spec_db_drumsa = librosa.amplitude_to_db(specdrumsa, ref=np.max)
            spec_db_drumsp = librosa.amplitude_to_db(specdrumsp, ref=np.max)
            spec_db_drumsc = librosa.amplitude_to_db(specdrumsc, ref=np.max)
            spec_db_drumsh = librosa.amplitude_to_db(specdrumsh, ref=np.max)
            spec_db_drumshh = librosa.amplitude_to_db(specdrumshh, ref=np.max)
            spec_db_drumsre = librosa.amplitude_to_db(specdrumsre, ref=np.max)
            spec_db_drumse = librosa.amplitude_to_db(specdrumse, ref=np.max)
            
            fig = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_drumsr, 
                         sr=22000, 
                         x_axis='time', 
                         y_axis='hz', 
                         cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Rock - The So So Glos - Emergency')
            plt.subplot(222)
            librosa.display.specshow(spec_db_drumsa, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Alternative - Tim Taler - Stalker')
            plt.xlim(4)
            
            plt.subplot(223)
            librosa.display.specshow(spec_db_drumsp, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Pop - Cristina Vane - So Easy')
            plt.xlim(4)
            
            plt.subplot(224)
            librosa.display.specshow(spec_db_drumsc, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Country - James May - Dont Let Go')
            plt.xlim(4);
            st.pyplot(fig)
            
            fig2 = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_drumsh, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Heavymetal - Wall Of Death - Femme')
            
            plt.subplot(222)
            librosa.display.specshow(spec_db_drumshh, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Hip-hop - Grants - PunchDrunk')
            
            plt.subplot(223)
            librosa.display.specshow(spec_db_drumsre, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Reggae - Music Delta - Reggae')
            
            plt.subplot(224)
            librosa.display.specshow(spec_db_drumse, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Drums Electro - PR - Happy Daze');
            st.pyplot(fig2)
            
            st.write(
                """
                Dans le spectre du tambour, nous pouvons trouver des similitudes pour tous les styles musicaux.
                En particulier, que chaque impulsion du tambour couvre toute la gamme de frequences de maniere 
                presque plate.  
    """)
    
        st.header('Bass par genre musical')
        
        opti = st.selectbox(
            "Choissisez le type d'affichage.",
            ('None', 'Waveforme', 'Spectrogramme'))    
        
        if opti == 'None':
            pass
        
        if opti == 'Waveforme':
            path_bassr = os.path.join( 'train', 'Dark Ride - Burning Bridges', 'bass.wav')
            path_bassa = os.path.join( 'train', 'Drumtracks - Ghost Bitch', 'bass.wav')
            path_bassp = os.path.join( 'train', 'Leaf - Wicked', 'bass.wav')
            path_bassc = os.path.join( 'train', 'Music Delta - Country1', 'bass.wav')
            path_bassh = os.path.join( 'test', 'Timboz - Pony', 'bass.wav')
            path_basshh = os.path.join( 'test', 'Side Effects Project - Sing With Me', 'bass.wav')
            path_bassre = os.path.join( 'test', 'Arise - Run Run Run', 'bass.wav')
            path_basse = os.path.join( 'test', 'PR - Oh No', 'bass.wav')
            
            sigbassr, ratevocalh = librosa.load(path_bassr, sr=22000, duration=30)
            sigbassa, ratevocalf = librosa.load(path_bassa, sr=22000, duration=30)
            sigbassp, ratevocalm = librosa.load(path_bassp, sr=22000,  duration=30)
            sigbassc, ratevocalh = librosa.load(path_bassc, sr=22000,  duration=30)
            sigbassh, ratevocalf = librosa.load(path_bassh, sr=22000, duration=30)
            sigbasshh, ratevocalm = librosa.load(path_basshh, sr=22000, duration=30)
            sigbassre, ratevocalf = librosa.load(path_bassre, sr=22000, duration=30)
            sigbasse, ratevocalm = librosa.load(path_basse, sr=22000, duration=30)
            
            fig = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigbassr, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Rock - Dark Ride - Burning Bridges')
            plt.ylim(-0.8,0.8)
            plt.xlim(7)
            plt.subplot(222)
            librosa.display.waveplot(sigbassa, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Alternative - Drumtracks - Ghost Bitch')
            plt.ylim(-0.8,0.8)
            plt.subplot(223)
            librosa.display.waveplot(sigbassp, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Pop - Leaf - Wicked')
            plt.ylim(-0.8,0.8)
            plt.xlim(3)
            plt.subplot(224)
            librosa.display.waveplot(sigbassc, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Country - Music Delta - Country1')
            plt.ylim(-0.8,0.8);
            st.pyplot(fig)
            
            fig2 = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.waveplot(sigbassh, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Heavymetal - Timboz - Pony')
            plt.ylim(-0.8,0.8)
            plt.xlim(4)
            plt.subplot(222)
            librosa.display.waveplot(sigbasshh, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Hip-hop - Side Effects Project - Sing With Me')
            plt.ylim(-0.8,0.8)
            plt.xlim(7)
            plt.subplot(223)
            librosa.display.waveplot(sigbassre, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Reggae - Arise - Run Run Run')
            plt.ylim(-0.8,0.8)
            plt.subplot(224)
            librosa.display.waveplot(sigbasse, sr=22000, color='teal', alpha=0.5)
            plt.title('Bass Electro - PR - Oh No')
            plt.ylim(-0.8,0.8);
            st.pyplot(fig2)
            
            st.write(
                """
                Il semble qu'en general, l'amplitud de la basse soit faible par rapport aux autres 
                instruments et cela se repete dans tous les styles musicaux  
    """)
        if opti == 'Spectrogramme':
                
            path_bassr = os.path.join( 'train', 'Dark Ride - Burning Bridges', 'bass.wav')
            path_bassa = os.path.join( 'train', 'Drumtracks - Ghost Bitch', 'bass.wav')
            path_bassp = os.path.join( 'train', 'Leaf - Wicked', 'bass.wav')
            path_bassc = os.path.join( 'train', 'Music Delta - Country1', 'bass.wav')
            path_bassh = os.path.join( 'test', 'Timboz - Pony', 'bass.wav')
            path_basshh = os.path.join( 'test', 'Side Effects Project - Sing With Me', 'bass.wav')
            path_bassre = os.path.join( 'test', 'Arise - Run Run Run', 'bass.wav')
            path_basse = os.path.join( 'test', 'PR - Oh No', 'bass.wav')
            
            sigbassr, ratevocalh = librosa.load(path_bassr, sr=22000, duration=30)
            sigbassa, ratevocalf = librosa.load(path_bassa, sr=22000, duration=30)
            sigbassp, ratevocalm = librosa.load(path_bassp, sr=22000,  duration=30)
            sigbassc, ratevocalh = librosa.load(path_bassc, sr=22000,  duration=30)
            sigbassh, ratevocalf = librosa.load(path_bassh, sr=22000, duration=30)
            sigbasshh, ratevocalm = librosa.load(path_basshh, sr=22000, duration=30)
            sigbassre, ratevocalf = librosa.load(path_bassre, sr=22000, duration=30)
            sigbasse, ratevocalm = librosa.load(path_basse, sr=22000, duration=30)
            
           # First, compute the spectrogram using the "short-time Fourier transform" (stft)
            specbassr = librosa.stft(sigbassr)
            specbassa = librosa.stft(sigbassa)
            specbassp = librosa.stft(sigbassp)
            specbassc = librosa.stft(sigbassc)
            specbassh = librosa.stft(sigbassh)
            specbasshh = librosa.stft(sigbasshh)
            specbassre = librosa.stft(sigbassre)
            specbasse = librosa.stft(sigbasse)
            
            
            # Scale the amplitudes according to the decibel scale
            spec_db_bassr = librosa.amplitude_to_db(specbassr, ref=np.max)
            spec_db_bassa = librosa.amplitude_to_db(specbassa, ref=np.max)
            spec_db_bassp = librosa.amplitude_to_db(specbassp, ref=np.max)
            spec_db_bassc = librosa.amplitude_to_db(specbassc, ref=np.max)
            spec_db_bassh = librosa.amplitude_to_db(specbassh, ref=np.max)
            spec_db_basshh = librosa.amplitude_to_db(specbasshh, ref=np.max)
            spec_db_bassre = librosa.amplitude_to_db(specbassre, ref=np.max)
            spec_db_basse = librosa.amplitude_to_db(specbasse, ref=np.max)

            
            fig = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_bassr, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Rock - Dark Ride - Burning Bridges')
            plt.xlim(7)
            plt.subplot(222)
            librosa.display.specshow(spec_db_bassa, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Alternative - Drumtracks - Ghost Bitch')
            
            plt.subplot(223)
            librosa.display.specshow(spec_db_bassp, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Pop - Leaf - Wicked')
            plt.xlim(2)
            
            plt.subplot(224)
            librosa.display.specshow(spec_db_bassc, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Country - Music Delta - Country1');
            st.pyplot(fig)
            
            fig2 = plt.figure(figsize=(13,13))
            plt.subplot(221)
            librosa.display.specshow(spec_db_bassh, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Heavymetal - Timboz - Pony')
            plt.xlim(3)
            
            plt.subplot(222)
            librosa.display.specshow(spec_db_basshh, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Hip-hop - Side Effects Project - Sing With Me')
            plt.xlim(6)
            
            plt.subplot(223)
            librosa.display.specshow(spec_db_bassre, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Reggae - Arise - Run Run Run')
            
            plt.subplot(224)
            librosa.display.specshow(spec_db_basse, 
                                     sr=22000, 
                                     x_axis='time', 
                                     y_axis='hz', 
                                     cmap=plt.get_cmap('Spectral'))
            plt.title('Bass Electro - PR - Oh No');
            st.pyplot(fig2)
            
            st.write(
                """
                Bien que certains styles fassent un usage plus fréquent de la guitare basse que d'autres,
                dans le spectre de la basse, nous pouvons trouver des similitudes pour tous 
                les styles musicaux : basses fréquences et dissipation sélective des hautes fréquences.
 
    """)








            
    if choice == 'Models':
      #  st.sidebar.header('Track separation')
        #choice = st.sidebar.selectbox(' ', ('','Music source separation'))
        
        
        
        
        
        st.header('Model types')
        st.write(
            """
            Source separation for music is the task of isolating contributions, or stems, 
            from different instruments recorded individually and arranged together to form a song. 
            These components include voice, bass, drums and any other accompaniments. Unlike 
            many audio synthesis tasks where the best performances are achieved by models that 
            directly generate the waveform, the state-of-the-art in source separation for music 
            is to compute masks on the magnitude spectrum.
            - Spectra
            - Waveform
""")

        st.header('Evaluation models: metrics')
        
        st.write(
            """
            An estimate of a Source s_i is assumed to actually be composed of four separate components,
            
            $$
            s_i = 𝑠_{target}+𝑒_{interf}+𝑒_{noise}+𝑒_{artif},
            $$
            where 𝑠_{target} is the true source, and 𝑒_{interf}, 𝑒_{noise}, and 𝑒_{artif} are error terms 
            for interference, noise, and added artifacts, respectively.

 """)

        st.write(
            """
            - Source-to-Distortion Ratio (SDR)
 """)
        st.latex(r'''
        SDR = 10 \log_{10}\left(\frac{|𝑠_{target}+𝑒_{interf}+𝑒_{noise}|^2}{|𝑒_{artif}|^2}\right)
        ''') 
        
        st.write(
            """
            This is usually interpreted as the amount of unwanted artifacts a source estimate has with relation to the true source.
 """)
        st.write(
            """
            - Mean Opinion Scores (MOS) evaluating the quality and absence of artifacts of the 
            separated audio. 38 people rated 20 samples each, randomly sample from one of the 3 
            models or the groundtruth. There is one sample per track in the MusDB test set and 
            each is 8 seconds long. Ratings of 5 means that the quality is perfect (no artifacts).
            
 """)

        
        st.write(
           """
            Comparison of accuracy: Overall SDR is the mean of the SDR for each of the 4 sources, 
            MOS Quality is a rating from 1 to 5 of the naturalness and absence of artifacts given by 
            human listeners (5 = no artifacts), MOS Contamination is a rating from 1 to 5 with 5 being 
            zero contamination by other sources.

            
            | Model         | Domain     | Extra data?  | Overall SDR | MOS Quality | MOS Contamination |
            | ------------- |-------------| -----:|------:|----:|----:|
            | [Open-Unmix][openunmix]      | spectrogram | no | 5.3 | 3.0 | 3.3 |
            | [D3Net][d3net]  | spectrogram | no | 6.0 | - | - |
            | [Wave-U-Net][waveunet]      | waveform | no | 3.2 | - | - |
            | Demucs (this)      | waveform | no | **6.3** | **3.2** | 3.3 |
            | Conv-Tasnet (this)     | waveform | no | 5.7 | 2.9 | **3.4** |
            | Demucs  (this)    | waveform | 150 songs | **6.8** | - | - |
            | Conv-Tasnet  (this)    | waveform | 150 songs | 6.3 | - | - |
            | [MMDenseLSTM][mmdenselstm]      | spectrogram | 804 songs | 6.0 | - | - |
            | [D3Net][d3net]  | spectrogram | 1.5k songs | 6.7 | - | - |
            | [Spleeter][spleeter]  | spectrogram | 25k songs | 5.9 | - | - |
            | **Our Model**  | **spectrogram** | **30s songs** | **4.04** | - | - |

""")
        
        st.header('Selected model : demucs')
        st.write(
            """
            Demucs, a new waveform to waveform model, with a U-Net structure and a bidirectional LSTM. 
            Experiments on the MusDB show that, with appropriate data augmentation, Demucs beats all 
            existing existing state-of-the-art architectures.
""")
        image = Image.open('demucs_model.png')

        st.image(image, caption='(a) Demucs architecture. (b) Detailed view of the layers Decoder. Source: A. Défossez et al.  Music Source Separation in the Waveform Domain. 2021.')

        st.header('Try it! Separate your track')
        
     
        #if choice == 'Music source separation':
            
        #    st.header('Separate track')

        uploaded_file = st.file_uploader("Upload your song", type=["wav", "mp3"])

        if uploaded_file is not None:
                
            audio_bytes = uploaded_file.read()
            st.write(uploaded_file.name)
            st.audio(audio_bytes, format='audio/wav')
            model = load_pretrained('demucs_quantized')
            track = Path(uploaded_file.name)
            wav = load_track(track, 'cpu', 2, 44100)
            out = Path("separated") / "demucs_quantized"
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            sources = apply_model(model, wav, shifts= 0, split=True, overlap=0.25, progress=True)
            sources = sources * ref.std() + ref.mean()

            track_folder = out / track.name.rsplit(".", 1)[0]
            track_folder.mkdir(exist_ok=True)

            for source, name in zip(sources, model.sources):
                wavname = str(track_folder / f"{name}.wav")
                ta.save(wavname, source, sample_rate=model.samplerate)
                
            text = "\\"
            #Affichage d'audio
            st.header(f'{uploaded_file.name[:-4]}')
            st.header('Resultats: ')
            st.write('Drums: ')
            st.text({uploaded_file.name[:-4]})
            #path_tmp = './separated/demucs_quantized'
            audio = open(f"./separated/demucs_quantized/{uploaded_file.name[:-4]}/drums.wav", ('rb'))
            #audio = open(f'./separated\demucs_quantizeds{uploaded_file.name[:-4]}\\drums.wav', ('rb'))
            audio_bytes = audio.read()
            st.audio(audio_bytes)
            #Affichage d'audio
            st.write('Bass: ')
            audio = open(f'./separated/demucs_quantized/{uploaded_file.name[:-4]}/bass.wav', ('rb'))
            audio_bytes = audio.read()
            st.audio(audio_bytes)
            #Affichage d'audio
            st.write('Voice: ')
            audio = open(f'./separated/demucs_quantized/{uploaded_file.name[:-4]}/vocals.wav', ('rb'))
            audio_bytes = audio.read()
            st.audio(audio_bytes)
            #Affichage d'audio
            st.write('Other: ')
            audio = open(f'./separated/demucs_quantized/{uploaded_file.name[:-4]}/other.wav', ('rb'))
            audio_bytes = audio.read()
            st.audio(audio_bytes)
            
        else:
            st.header('Example: Daft Punk - Get Lucky')
            st.write('Oiriginal mixutre: ')
            #Affichage d'audio
            audio_file = open( 'daft-punk-get-lucky.wav', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.header('Resultats: ')
            st.write('Drums: ')
            #Affichage d'audio
            audio_file = open( 'streamlit-getlucky-drums.wav', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.write('Bass: ')
            #Affichage d'audio
            audio_file = open( 'streamlit-getlucky-bass.wav', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.write('Voice: ')
            #Affichage d'audio
            audio_file = open( 'streamlit-getlucky-vocals.wav', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.write('Other: ')
            #Affichage d'audio
            audio_file = open( 'streamlit-getlucky-other.wav', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
        
        
        
     
        
     
    if choice == 'Demo':
        st.sidebar.header('Demo')
        choice = st.sidebar.selectbox(' ', ('Mixer', 'Looper'))
        
        if choice == 'Mixer':

            st.header('Mixer')
            st.write('\nThis mixer is MixingBear a package for automatic beat-mixing of music files in Python to get two audio files and to mix them perfectly.')
            
            st.write('In our case we use MixingBear to mix two stems of two different songs as in the example below: ')

            st.header('Before mixing: ')
            st.write('Drums: Daft Punk - Around the World')
            #Affichage d'audio
            audio_file = open( 'Daftpunk_Around_the_world_drums.mp3', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            
            st.write('Bass: Red hot chilli papers - By the way')
            #Affichage d'audio
            audio_file = open( 'Red_Hot_Chilli_by_the_way_bass.wav', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            
            st.write('\n')
            code = '''mixer(Daftpunk_Around_the_world_drums, Red_Hot_Chilli_by_the_way_bass, Mixture)'''

            st.code(code, language='python')

            st.header('After mixing: ')
            st.write('Drums: Daft Punk - Around the World + Bass: Red hot chilli papers - By the way')
            #Affichage d'audio
            audio_file = open( 'mix1.mp3', ('rb'))
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)

            st.header("How it's work?")

            code = '''import librosa
import madmom
from madmom.features.beats import *

rnn_processor = RNNBeatProcessor(post_processor=None)
predictions = rnn_processor(y)
mm_processor = MultiModelSelectionProcessor(num_ref_predictions=None)
beats = mm_processor(predictions)

data['beat_samples'] = peak_picking(beats, len(y), 5, 0.01)

    if len(data['beat_samples']) < 3:
        data['beat_samples'] = peak_picking(beats, len(y), 25, 0.01)

    if data['beat_samples'] == []:
        data['beat_samples'] = [0]

    data['number_of_beats'] = len(data['beat_samples'])

    # tempo
    data['tempo_float'] = (len(data['beat_samples'])-1)*60/data['duration']
    data['tempo_int'] = int(data['tempo_float'])


    # noisiness featues
    data['zero_crossing'] = librosa.feature.zero_crossing_rate(y)[0].tolist()
    data['noisiness_median'] = float(np.median(data['zero_crossing']))
    data['noisiness_sum'] = sum( librosa.zero_crossings(y)/y.shape[0] )

    # spectral features
    notes = []

    try:
        chroma = librosa.feature.chroma_cqt(y, n_chroma=12, bins_per_octave=12, n_octaves=8, hop_length=512)

        for col in range(chroma.shape[1]):
            notes.append(int(np.argmax(chroma[:,col])))

        data['notes'] = notes
        data['dominant_note'] = int(np.argmax(np.bincount(np.array(notes))))
    except:
        data['notes'] = [0]
        data['dominant_note'] = 0

    return data'''

            st.code(code, language='python')

            code = '''def find_best_sync_point(bottom_file_beats, top_file_beats, max_mix_sample, offset, mode):

    offset = offset
    matches_per_round = []

    # turning args to numpy arrays
    bottom_file_beats = np.array(bottom_file_beats)
    top_file_beats = np.array(top_file_beats)

    for rn in range(bottom_file_beats.shape[0]):

        try:

            zero_sync_samples = bottom_file_beats[rn] - top_file_beats[0]
            slider = top_file_beats + (zero_sync_samples)'''

            st.code(code, language='python')
        if choice == 'Looper':

            st.header('Looper')
            st.write('This looper use a python script for extracting loops from audio files.')
            st.write('The script is a way to identify and extract the loops that occur in a song, and to provide a map of the piece, all in a single optimization, to do so, the script use nonnegative tensor factorization.')
            st.caption('Smith, Jordan B. L., and Goto, Masataka. 2018. "Nonnegative tensor factorization for source separation of loops in audio." Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (IEEE ICASSP 2018). Calgary, AB, Canada. pp. 171--175.')
            st.write('In our case we use this looper to extract loops of different stems and songs as in the example below:')
            st.header('Select your rhytm!\n')
            select1, select2 = st.columns(2)
            with select1:
                 select1 = st.checkbox('Daft Punk - Get Lucky - other', value=False) 
                 if select1 == True:
                    with select2:
                        select2 = st.checkbox('DNCE - Cake By The Ocean - drums', value=False)
    
            select3, select4 = st.columns(2)
            with select3:
                 select3 = st.checkbox('Michael Jackson - Billie Jean - drums', value=False) 
                 if select3 == True:
                    with select4:
                        select4 = st.checkbox('Daft Punk - Around the world - other', value=False)    
        
            select5, select6 = st.columns(2)
            with select5:
                 select5 = st.checkbox('Arctic Monkeys - Arabella - other', value=False) 
                 if select5 == True:
                    with select6:
                        select6 = st.checkbox('DNCE - Cake By The Ocean - voice', value=False) 

            

            if select1 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound.play(-1)

            if select3 == True:
                pygame.quit()
                pygame.mixer.init()
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound3.play(-1)

            if select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound5= pygame.mixer.Sound("extracted_mix10_2.wav")
                sound5.play(1)
                
            if select1 == True and select2 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound.play(-1)
                sound2.play(-1)

            if select1 == True and select3 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound.play(-1)
                sound3.play(-1)
            
            if select1 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound.play(-1)
                sound5.play(-1)

            if select3 == True and select4 == True:
                pygame.quit()
                pygame.mixer.init()
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound3.play(-1)
                sound4.play(-1)

            if select3 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound3.play(-1)
                sound5.play(-1)

            if select5 == True and select6 == True:
                pygame.quit()
                pygame.mixer.init()
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound5.play(-1)
                sound6.play(-1)

            if select1 == True and select2 == True and select3 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound.play(-1)
                sound2.play(-1)
                sound3.play(-1)  

            if select1 == True and select2 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound.play(-1)
                sound2.play(-1)
                sound5.play(-1)
                
            if select1 == True and select3 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound.play(-1)
                sound3.play(-1)
                sound5.play(-1)  

            if select1 == True and select3 == True and select4 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound.play(-1)
                sound3.play(-1)
                sound4.play(-1) 

            if select3 == True and select4 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound3.play(-1)
                sound4.play(-1)  
                sound5.play(-1)

            if select1 == True and select5 == True and select6 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound.play(-1)
                sound5.play(-1)
                sound6.play(-1)

            if select3 == True and select5 == True and select6 == True:
                pygame.quit()
                pygame.mixer.init()
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound3.play(-1)
                sound5.play(-1)
                sound6.play(-1)

            if select1 == True and select2 == True and select3 == True and select4 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound.play(-1)
                sound2.play(-1)
                sound3.play(-1) 
                sound4.play(-1) 

            if select1 == True and select2 == True and select3 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound.play(-1)
                sound2.play(-1)
                sound3.play(-1) 
                sound5.play(-1) 

            if select1 == True and select2 == True and select5 == True and select6 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound.play(-1)
                sound2.play(-1)
                sound5.play(-1) 
                sound6.play(-1)
            
            if select1 == True and select3 == True and select5 == True and select6 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound.play(-1)
                sound3.play(-1)
                sound5.play(-1) 
                sound6.play(-1)

            if select1 == True and select3 == True and select4 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound.play(-1)
                sound3.play(-1) 
                sound4.play(-1)
                sound5.play(-1) 
            
            if select3 == True and select4 == True and select5 == True and select6 == True:
                pygame.quit()
                pygame.mixer.init()
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound3.play(-1) 
                sound4.play(-1)
                sound5.play(-1) 
                sound6.play(-1)
            
            if select1 == True and select2 == True and select3 == True and select4 == True and select5 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound.play(-1)
                sound2.play(-1)
                sound3.play(-1) 
                sound4.play(-1)
                sound5.play(-1)

            if select1 == True and select2 == True and select5 == True and select6 == True and select3 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound.play(-1)
                sound2.play(-1)
                sound3.play(-1)
                sound5.play(-1) 
                sound6.play(-1)  

            if select3 == True and select4 == True and select5 == True and select6 == True and select1 == True:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound.play(-1)
                sound3.play(-1) 
                sound4.play(-1)
                sound5.play(-1) 
                sound6.play(-1)

            if select1 == True and select2 == True and select3 == True and select4 == True and select5 == True and select6:
                pygame.quit()
                pygame.mixer.init()
                sound = pygame.mixer.Sound("extracted_mix2_14.wav")
                sound2 = pygame.mixer.Sound("extracted_mix4_6.wav")
                sound3 = pygame.mixer.Sound("extracted_mix3_0.wav")
                sound4 = pygame.mixer.Sound("extracted_mix1_3.wav")
                sound5 = pygame.mixer.Sound("extracted_mix10_2.wav")
                sound6 = pygame.mixer.Sound("extracted_mix11_3.wav")
                sound.play(-1)
                sound2.play(-1)
                sound3.play(-1) 
                sound4.play(-1)
                sound5.play(-1)
                sound6.play(-1)

            if select1 == False and select3 == False and select5 == False:
                pygame.quit()


            if st.button('Stop'):
                pygame.quit()
            
            st.header('Follow the rhytm!')
            g3=net.Network(height='400px', width='80%',heading='')
            ids=['V', 'SEP', 'PY', 'V', 'SEP', 'PY', 'V', 'SEP', 'PY', 'V', 'SEP', 'PY', 'V', 'SEP']
            xs=[2.776, 1.276, 0.3943, -1.0323, -1.0323, 0.3943, 0.7062, 2.1328, -0.4086, -1.8351, -2.9499, -2.147, -3.5736, -0.0967]
            ys=[0.0, 0.0, 1.2135, 0.75, -0.75, -1.2135, -2.6807, -3.1443, -3.6844, -3.2209, -4.2246, -1.7537, -1.2902, -5.1517]
            bonds=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [8, 9], [9, 10], [9, 11], [11, 12], [8, 13], [5, 1], [11, 4]]
            g3.set_options('''
            var options = {
                "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4
            },
            "edges":{
                "width":24
            },
            "physics": {
                "barnesHut": {
                "gravitationalConstant":-2000,
                "centralGravity": 0,
                "springLength": 60,
                "springConstant": 0.545,
                "damping": 0.1,
                "avoidOverlap": 0.52
                },
                "maxVelocity:":50,
                "minVelocity": 0.75,
                "timestep": 0.5
            }
            }
            ''')
            for atomo in range(14):
                g3.add_node(atomo,label=ids[atomo],x=int(100*xs[atomo]),y=int(100*ys[atomo]),physics=True,size=30)
            g3.add_edges(bonds)
            HtmlFile = open("g3.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height = 900,width=900)

    expander = st.sidebar.expander('Contact')
   # , [Twitter] (https://twitter.com/lopezyse) and [Medium] (https://lopezyse.medium.com/)"
    expander.write("Tawfa OUISSI: Find me on [LinkedIn] (https://www.linkedin.com/in/tawfa-ouissi-621492108/)")
    expander.write("Ernesto HORNE: Find me on [LinkedIn] (https://www.linkedin.com/in/ernesto-horne-a8552a23/)")    
    expander.write("Daniel PEREZ: Find me on [LinkedIn] (https://www.linkedin.com/in/dan-prz/)")   
    expander.write("Michel DJIMASSE: michel djimasse.sapbw@gmail.com")   

if __name__ == '__main__':
    main()
    
    
    
#        V-SepPy Team
#                                     __                                        __ 
#                |--|                                      |--|
#     .._       o' o'                     (())))     _    o' o'
#    //\\\    |  __                      )) _ _))  ,' ; |  __  
#   ((-.-\)  o' |--|  ,;::::;.          (C    )   / /^ o' |--| 
#  _))'='(\-.  o' o' ,:;;;;;::.         )\   -'( / /     o' o'
# (          \       :' o o `::       ,-)()  /_.')/                 .
# | | .)(. |\ \      (  (_    )      /  (  `'  /\_)    .:izf:,_  .  |
# | | _   _| \ \     :| ,==. |:     /  ,   _  / 1  \ .:q568Glip-, \ |
# \ \/ '-' (__\_\____::\`--'/::    /  /   / \/ /|\  \-38'^"^`8k='  \L,
#  \__\\[][]____(_\_|::,`--',::   /  /   /__/ <(  \  \8) o o 18-'_ ( /
#   :\o*.-.(     '-,':   _    :`.|  L----' _)/ ))-..__)(  J  498:- /]
#   :   [   \     |     |=|   '  |\_____|,/.' //.   -38, 7~ P88;-'/ /
#   :  | \   \    |  |  |_|   |  |    ||  :: (( :   :  ,`""'`-._,' /
#   :  |  \   \   ;  |   |    |  |    \ \_::_)) |  :  ,     ,_    /
#   :( |   /  )) /  /|   |    |  |    |    [    |   \_\      _;--==--._ 
#OT :  |  /  /  /  / |   |    |  |    |    Y    |DP  (_\____:_        _:
#   :  | /  / _/  /  \   |MD  |  |  EH|    |    | ,--==--.  |_`--==--'_|
#                                                         "   `--==--'
