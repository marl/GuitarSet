import vamp
import librosa
import csv
import argparse
import numpy as np

def offset_anal(y, fs):
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
             "precisetime": 0,
             "prunethresh": 0.05,
             "onsetsensitivity": 0.8}

    vp = vamp.collect(y, fs, 'pyin:pyin', output='voicedprob', parameters=param)

    return vp['vector']

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

def segment_offset(seg, sr, seg_start_time, verbose=False):
    t_step, vp = offset_anal(seg, sr)
    t_step, pt = pt_anal(seg, sr)
    rms = librosa.feature.rmse(y=seg, hop_length=256)[0]
    min_idx = min(len(vp), len(rms))
    vp = vp[0:min_idx] + 1e-12
    rms = rms[0:min_idx]
    
    log_offset_prob = -np.log(rms * vp)
    
    
    offset_frame = 10 # start checking at around 58 ms.
    while offset_frame < len(log_offset_prob) and log_offset_prob[offset_frame] < 8.5:
        offset_frame += 1;
    
    offset_time = librosa.frames_to_time(offset_frame, hop_length=256, sr=sr) + seg_start_time

    t_vec = (np.arange(len(vp)) * float(t_step) + seg_start_time)
    
    if verbose:
        plt.figure()
        ax = plt.subplot(3,1,1)
        plt.plot(t_vec, vp)
        plt.plot(t_vec, rms)
        plt.xlabel('sec')

        plt.subplot(3,1,2, sharex=ax)
        plt.plot(t_vec, log_offset_prob)
        plt.vlines(offset_time, 0, 12)
        plt.xlabel('sec')

        plt.subplot(3,1,3, sharex=ax)
        plt.plot(t_vec[0:offset_frame], librosa.hz_to_midi(pt[0:offset_frame]))
        plt.xlabel('sec')

        plt.show()
    
    return offset_time, pt[0:offset_frame], t_step


def main(args):
    y, sr = librosa.load(args.wav_path, sr=None)
    offset_time, pitch_track, t_step = segment_offset(y, sr, args.seg_start_time)
    
    with open(args.pt_path, 'a') as pt:
        writer = csv.writer(pt, delimiter=',')
        for i, f in enumerate(pitch_track):
            writer.writerow([args.seg_start_time + i*float(t_step) , f])
    

    with open(args.onoff_path, 'a') as onoff:
        writer = csv.writer(onoff, delimiter=',')
        writer.writerow([args.seg_start_time,offset_time])
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='a note using pYin. appending the result to a existing csv file.')
    parser.add_argument(
        'wav_path', type=str, help='path to the stem of interest')
    parser.add_argument(
        'seg_start_time', type=float,
        help='time for the start of each element')
    parser.add_argument(
        'pt_path', type=str,
        help='path for pitch_track. ready to import for Tony')
    parser.add_argument(
        'onoff_path', type=str,
        help='path for NoteOn NoteOff time pairs csv.')
        
    main(parser.parse_args())