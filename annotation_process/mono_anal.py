import vamp
import librosa
import csv
import argparse
import numpy as np


def mono_anal(y, fs):
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
             "outputunvoiced": 2,
             "precisetime": 0,
             "prunethresh": 0.05,
             "onsetsensitivity": 0.8}

    output = vamp.collect(y, fs, 'pyin:pyin', output='smoothedpitchtrack', parameters=param)

    return output['vector']

def trim_pitch_track_end(pitch_track):
    i=0
    #find first positive
    while (pitch_track[i] < 0):
        if i == len(pitch_track) - 1:
            break
        i += 1
    #then find first negative
    st = i
    while (pitch_track[i] > 0):
        if i == len(pitch_track) - 1:
            break
        i += 1
    end = i
    return pitch_track[st:end], st

def main(args):
    y, fs = librosa.load(args.wav_path, sr=None)
    t_step, pitch_track = mono_anal(y, fs)
    pitch_track_trimed, st = trim_pitch_track_end(pitch_track)
    p_time = np.arange(len(pitch_track_trimed)) * float(t_step) + args.seg_start_time
    if len(p_time) == 0:
        print('trimmed pitch_track is too short, droped note.')
        return 0
    on_time = p_time[0]
    off_time = p_time[-1]
    
    with open(args.pt_path, 'a') as pt:
        writer = csv.writer(pt, delimiter=',')
        for t, f in zip(p_time, pitch_track_trimed[st:-1]):
            writer.writerow([t,f])
    

    with open(args.onoff_path, 'a') as onoff:
        writer = csv.writer(onoff, delimiter=',')
        writer.writerow([on_time,off_time])
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
