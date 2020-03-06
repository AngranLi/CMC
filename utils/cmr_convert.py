from zipfile import ZipFile
import json
import struct
import numpy as np
import pickle

ARRAY_DATA_TYPE_INT64 = 8
ARRAY_DATA_TYPE_FLOAT64 = 16
ARRAY_DATA_TYPE_COMPLEX128 = 19


def get_hashes(sensor, hashes):
    if isinstance(sensor, dict):
        for k, v in sensor.items():
            if k == "hash":
                hashes.append(v)
            elif isinstance(v, dict):
                get_hashes(v, hashes)
            elif isinstance(v, list):
                for j in v:
                    get_hashes(j, hashes)


def bytes_to_array(bytes_array):
    offset = 0
    id, endian, dim = struct.unpack_from("3Q", bytes_array, offset)
    
    offset += 24
    shape = struct.unpack_from(f'{dim}Q', bytes_array, 24)

    offset += dim * 8
    N = np.prod(shape)
    data_type, byte_size = struct.unpack_from("2Q", bytes_array, offset)
    
    offset += 2 * 8
    if data_type == ARRAY_DATA_TYPE_FLOAT64:
        array = struct.unpack_from(f'{N}d', bytes_array, offset)
        offset += N * 8
    elif data_type == ARRAY_DATA_TYPE_INT64:
        array = struct.unpack_from(f'{N}q', bytes_array, offset)
        offset += N * 16
    elif data_type == ARRAY_DATA_TYPE_INT64:
        real_view = struct.unpack_from(f'{2 * N}d', bytes_array, offset)
        offset += N * 16
        real_view = np.array(real_view)
        array = real_view.view(np.complex128)
    array = np.reshape(array, shape)

    return array


def convert_p1m(path, save_path=None):
    zfile = ZipFile(path, 'r')
    name_list = zfile.namelist()
    schema_name = [n for n in name_list if n.endswith('schema.json')][0]
    schema = json.loads(zfile.read(schema_name).decode("utf-8"))
    sensor = schema['sensors']['radar']['default']

    hashes = []
    get_hashes(sensor, hashes)
    hashes = list(set(hashes))

    hash_bytes = {}
    for h in hashes:
        hash_bytes[h] = zfile.read(f'array/{h}.array')
    
    hash_objs = {}
    traces = []
    for sample in sensor['samples']:
        stacks = sample['stacks']
        datasource_id = sample['datasource_id']
        trace = {}
        for k, v in sample['values'].items():
            x_hash = v['x']['hash']
            y_hash = v['y']['hash']
            
            x = None
            y = None
            
            if x_hash in hash_objs:
                x = hash_objs[x_hash]
            elif x_hash in hash_bytes:
                bytes_array = hash_bytes[x_hash]
                x = bytes_to_array(bytes_array)
                hash_objs[x_hash] = x
            else:
                raise LookupError(f'Hash not found for x array: hash = {x_hash}')
            
            if y_hash in hash_objs:
                y = hash_objs[y_hash]
            elif y_hash in hash_bytes:
                bytes_array = hash_bytes[y_hash]
                y = bytes_to_array(bytes_array)
                hash_objs[y_hash] = y
            else:
                raise LookupError(f'Hash not found for y array: hash = {y_hash}')
        
            trace[k] = y.copy()
        
        traces.append(trace)

    data = {'metadata': schema, 'traces': traces}

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    return data
