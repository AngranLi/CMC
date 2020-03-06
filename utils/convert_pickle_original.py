import pickle
import pandas as pd
import glob
import os
import argparse
import numpy as np

channel_list = [
        "S11", "S12", "S22"
    ]

def convert(path, save_path, ext):
    """Converts pickle files to another format
    
    Arguments:
        path {str} -- pickle file location
        save_path {str} -- new file location
        format {str} -- file extention (csv, npy, npz)
    """
    if isinstance(path, str):
        with open(path, "rb") as f:
            raw = pickle.load(f)
    else:
        raw = pickle.loads(path)
    label = raw['metadata']['threat_type']['target_model']
    traces = raw['traces']

    # raise Exception(traces)

    if ext=='csv':
        csv = None
        for trace in traces:
            df = pd.DataFrame.from_dict(trace)
            if csv is None:
                csv = df
            else:
                csv = pd.concat([csv,df])
        csv.to_csv(save_path, index=False)
    
    elif ext=='npy':
        img = []
        for trace in traces:
            img.append(np.stack(([(trace[channel]) for channel in channel_list]), axis=1))
        img = np.array(img)
        np.save(save_path, img, allow_pickle=False)
    
    elif ext=='npz':
        channels = list(traces[0].keys())
        channels.sort()
        img = {}
        for c in channels:
            arr = np.stack([trace[c] for trace in traces])
            img[c]=arr
        np.savez(save_path, **img)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Changes pickle to another format (csv, npy, npz)')
    parser.add_argument('--root_path', '-r', type=str, help='Root path with .pickle files')
    parser.add_argument('--save_path', '-s', type=str, help='Destination for new files')
    parser.add_argument('--format', '-f', type=str, help='extension for saved files (csv, npy, npz)')
    args = parser.parse_args()

    orig_paths = glob.glob(os.path.join(args.root_path, "**/*.pickle"), recursive=True)
    new_paths = [x.replace(args.root_path, args.save_path, 1)[:-6] + args.format for x in orig_paths]

    for i in range(len(orig_paths)):
        try:
            convert(orig_paths[i], new_paths[i], args.format)
        except:
            os.makedirs(os.path.dirname(new_paths[i]))
            convert(orig_paths[i], new_paths[i], args.format)
