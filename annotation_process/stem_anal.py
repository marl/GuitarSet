import vamp
import librosa
import csv
import argparse
import numpy as np
import jams


def offset_strength_multi(y=None, sr=22050, S=None, lag=1, max_size=1,
                         detrend=False, center=True, feature=None,
                         aggregate=None, channels=None, **kwargs):
    """Compute a spectral flux onset strength envelope across multiple channels.

    Onset strength for channel `i` at time `t` is determined by:

    `mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])`


    Parameters
    ----------
    y        : np.ndarray [shape=(n,)]
        audio time-series

    sr       : number > 0 [scalar]
        sampling rate of `y`

    S        : np.ndarray [shape=(d, m)]
        pre-computed (log-power) spectrogram

    lag      : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by `n_fft / (2 * hop_length)` frames

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with `fmax=11025.0`

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.

    kwargs : additional keyword arguments
        Additional parameters to `feature()`, if `S` is not provided.


    Returns
    -------
    onset_envelope   : np.ndarray [shape=(n_channels, m)]
        array containing the onset strength envelope for each specified channel


    Raises
    ------
    ParameterError
        if neither `(y, sr)` nor `S` are provided


    See Also
    --------
    onset_strength

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=10.0)
    >>> D = librosa.stft(y)
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log')
    >>> plt.title('Power spectrogram')

    Construct a standard onset function over four sub-bands

    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    ...                                                     channels=[0, 32, 64, 96, 128])
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(onset_subbands, x_axis='time')
    >>> plt.ylabel('Sub-bands')
    >>> plt.title('Sub-band onset strength')

    """

    if feature is None:
        feature = librosa.feature.melspectrogram
        kwargs.setdefault('fmax', 11025.0)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, int):
        raise ParameterError('lag must be a positive integer')

    if max_size < 1 or not isinstance(max_size, int):
        raise ParameterError('max_size must be a positive integer')

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, **kwargs))

        # Convert to dBs
        S = librosa.core.power_to_db(S)

    # Retrieve the n_fft and hop_length,
    # or default values for onsets if not provided
    n_fft = kwargs.get('n_fft', 2048)
    hop_length = kwargs.get('hop_length', 512)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if max_size == 1:
        ref_spec = S
    else:
        ref_spec = scipy.ndimage.maximum_filter1d(S, max_size, axis=0)

    # Compute difference to the reference, spaced by lag
    onset_env = S[:, lag:] - ref_spec[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, -1 * onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    onset_env = librosa.util.sync(onset_env, channels,
                          aggregate=aggregate,
                          pad=pad,
                          axis=0)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]),
                       mode='constant')

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99],
                                         onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[:, :S.shape[1]]

    return onset_env

def segment_offset(seg, sr, seg_start_time, verbose=False):
    if len(seg) > sr*4:
        seg = seg[0:int(sr*4)]

    seg, perc = librosa.effects.hpss(y=seg)

    t_step, pt = pt_anal(seg, sr)
    hop_len = 256


    rms = librosa.feature.rmse(y=seg, hop_length=hop_len)[0].reshape(-1,1)  + 1e-15
    silence_f = rms.shape[0]-1
    for i, e in enumerate(rms):
        if i < 5:
            continue
        if e < 1e-14:
            silence_f = i
            break
    rms_db = np.log(rms)
    rms_diff = -np.diff(rms_db, axis=0)
    offset_str = offset_strength_multi(y=seg, hop_length=hop_len, sr=sr)[0].reshape(-1,1)
    min_idx = min(rms_diff.shape[0], offset_str.shape[0])
    rms_diff = rms_diff[:min_idx, :]
    offset_str = offset_str[:min_idx, :]

    to_peak_pick = np.maximum(rms_diff * offset_str / rms[:-1], 0.0)
    t_vec = (np.arange(to_peak_pick.shape[0]) * float(t_step)).reshape(-1,1)



    offset_frames_raw = librosa.util.peak_pick(to_peak_pick.ravel(), 5, 5, 5, 7, 0.5, 10)
    offset_frames_raw = [f for f in offset_frames_raw if to_peak_pick[f] > 2]
    offset_frames = np.asarray([f for f in offset_frames_raw if t_vec[f] > 0.06])
    if offset_frames.shape[0] == 0:
        offset_frames = [silence_f]
    offset_times = librosa.frames_to_time(offset_frames, hop_length=hop_len, sr=sr)
    
#     print(offset_frames[0])



    if verbose:
        plt.figure()
        ax = plt.subplot(3,1,1)
        librosa.display.waveplot(y=seg, sr=sr)
        plt.subplot(3,1,2, sharex=ax)
        plt.plot(t_vec, offset_str)
        plt.plot(t_vec, rms_diff)
        plt.vlines(offset_times[0], 0, 1)
        plt.xlabel('sec')
        plt.show()

        plt.subplot(3,1,3, sharex=ax)
        plt.plot(t_vec, to_peak_pick)
        plt.vlines(offset_times, 0, max(to_peak_pick), linestyles='dashed')
        plt.xlabel('sec')
        plt.show()
    return offset_times[0] + seg_start_time, pt[0:offset_frames[0]], t_step


def pt_anal(y, fs):
    """Run pyin on an audio signal y.

    Parameters
    ----------
    y : np.array
        audio signal
    fs : float
        audio sample rate
    param : dict default=None
        threshdistr : int, default=2
            Yin threshold distribution identifier.
                - 0 : uniform
                - 1 : Beta (mean 0.10)
                - 2 : Beta (mean 0.15)
                - 3 : Beta (mean 0.20)
                - 4 : Beta (mean 0.30)
                - 5 : Single Value (0.10)
                - 6 : Single Value (0.15)
                - 7 : Single Value (0.20)
        outputunvoiced : int, default=0
            Output estimates classified as unvoiced?
                - 0 : No
                - 1 : Yes
                - 2 : Yes, as negative frequencies
        precisetime : int, default=0
            If 1, use non-standard precise YIN timing (slow)
        lowampsuppression : float, default=0.005
            Threshold between 0 and 1 to supress low amplitude pitch estimates.


    """

    param = {"threshdistr": 2,
             "lowampsuppression": 0.08,
             "outputunvoiced": 1,
             "precisetime": 0,
             "prunethresh": 0.05,
             "onsetsensitivity": 0.8}

    pt_output = vamp.collect(y, fs, 'pyin:pyin', output='smoothedpitchtrack', parameters=param)

    return pt_output['vector']

def chop_sig(y, adj_on_samps):
    y_chopt = []
    for (i, on) in enumerate(adj_on_samps):
        if i+1 == len(adj_on_samps):
            end = len(y)
        else:
            end = adj_on_samps[i+1]

        y_chopt.append(y[on:end])
    return y_chopt

def note_anal(y, fs, seg_start_time, outname):
    offset_time, pitch_track, t_step = segment_offset(y, fs, seg_start_time)

    with open(outname+'_pt.csv', 'a') as pt:
        writer = csv.writer(pt, delimiter=',')
        for i, f in enumerate(pitch_track):
            writer.writerow([seg_start_time + i*float(t_step) , f])

    with open(outname+'_onoff.csv', 'a') as onoff:
        writer = csv.writer(onoff, delimiter=',')
        writer.writerow([seg_start_time, offset_time])
    return 0

def main(args):
    outname = args.outdir
    y, sr = librosa.load(args.wavpath, sr=None)
    adjusted_onset_jam = jams.load(args.onsetjams)
    ann = adjusted_onset_jam.search(namespace='onset')[0]
    adjusted_onset_times = ann.to_event_values()[0]
    adj_on_samps = librosa.time_to_samples(adjusted_onset_times, sr=sr)
    y_chopt = chop_sig(y, adj_on_samps)
    print("about to clear csv files")

    with open(outname+'_pt.csv', 'w') as csvfile: pass
    with open(outname+'_onoff.csv', 'w') as csvfile: pass

    k=0
    for seg, seg_start_time in zip(y_chopt, adjusted_onset_times):
        if k%20 == 0:
            print(k, len(y_chopt))
        k+=1

        offset_time, pitch_track, t_step = segment_offset(seg, sr, seg_start_time)
        with open(outname+'_pt.csv', 'a') as pt:
            writer = csv.writer(pt, delimiter=',')
            for i, f in enumerate(pitch_track):
                if f > 0:
                    writer.writerow([seg_start_time + i*float(t_step) , f])
        with open(outname+'_onoff.csv', 'a') as onoff:
            writer = csv.writer(onoff, delimiter=',')
            writer.writerow([seg_start_time, offset_time])

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='a note using pYin. appending the result to a existing csv file.')
    parser.add_argument(
        'wavpath', type=str, help='path to the stem of interest')
    parser.add_argument(
        'outdir', type=str, help='path to the output directory + name')
    parser.add_argument(
        'onsetjams', type=str, help='onsetjams')


    main(parser.parse_args())
