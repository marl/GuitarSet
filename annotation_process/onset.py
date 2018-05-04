import vamp
import librosa
import numpy as np
import pretty_midi
import jams
import os
import argparse

def rough_midi(args):
    # gets a rough midi trasncription using pYin Note from inpath.wav to outpath.mid
    if os.path.exists(args.outpath):
        print('file already exist')
        return 0
        #check if outpath exist, return 0.
        
    y, fs = librosa.load(args.inpath, sr=None)
#     tempo = int(args.inpath.split('_')[1].split('-')[1])
    print('finished loading')
    
    param = {"threshdistr": 2,
             "lowampsuppression": 0.08,
             "outputunvoiced": 2,
             "precisetime": 0,
             "prunethresh": 0.05,
             "onsetsensitivity": 0.8}

    pyin_note_output = vamp.collect(y, fs, 'pyin:pyin', output='notes', parameters=param)['list']
    print('finished pYin')
    midi = build_midi_from_output(pyin_note_output)
    
    midi.write(args.outpath)
    return 0


def build_midi_from_output(pyin_note_output):
    midi = pretty_midi.PrettyMIDI('tempo_template.mid')
    ch = pretty_midi.Instrument(program=25)
    for note in pyin_note_output:
        pitch = int(round(librosa.hz_to_midi(note['values'])[0]))
        st = float(note['timestamp'])
        dur = float(note['duration'])
#         print(pitch, st, dur )
        n = pretty_midi.Note(
            velocity=100,
            pitch=pitch, start=st,
            end=st+dur
        )

        ch.notes.append(n)
#         bend_amount = int(round((note.value - pitch) * 4096))
#         pb = pretty_midi.PitchBend(pitch=bend_amount*q, time=st)
#         ch.pitch_bends.append(pb)
    midi.instruments.append(ch)
    return midi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='analyze whole stems.')
    parser.add_argument(
        'inpath', type=str, help='path to the stem of interest')
    parser.add_argument(
        'outpath', type=str, help='path to the stem of interest')
        
    rough_midi(parser.parse_args())
