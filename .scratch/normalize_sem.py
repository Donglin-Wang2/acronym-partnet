import json
import os
import shutil
import numpy as np

ID = "6cb6373befd7e1a7aa918b099faddaba"
IN = f'./temp_meshes/{ID}_sem.obj'
OUT = f"./temp_meshes/{ID}_sem_normalized.obj"
STAT = {"max": [0.79121, 0.933076, 0.9144], "centroid": [0.3977983202067827, 0.40019718530994164, 0.5265005023087718], "id": "6cb6373befd7e1a7aa918b099faddaba", "numVertices": 855, "min": [-0.00635, -0.00302059, -0.0065486]}

def obj2stats(obj):
    """
    Computes statistics of OBJ vertices and returns as {num,min,max,centroid}
    """
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    with open(obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                aggVertices += v
                numVertices += 1
                minVertex = np.minimum(v, minVertex)
                maxVertex = np.maximum(v, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['numVertices'] = numVertices
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info

def normalizeOBJ(obj, out, stats=None):
    """
    Normalizes OBJ to be centered at origin and fit in unit cube
    """
    if not stats:
        stats = obj2stats(obj)
    diag = np.array(stats['max']) - np.array(stats['min'])
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    outmtl = os.path.splitext(out)[0] + '.mtl'
    with open(obj, 'r') as f, open(out, 'w') as fo:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                vNorm = (v - c) * norm
                vNormString = 'v %f %f %f\n' % (vNorm[0], vNorm[1], vNorm[2])
                fo.write(vNormString)
            elif line.startswith('mtllib '):
                fo.write('mtllib ' + os.path.basename(outmtl) + '\n')
            else:
                fo.write(line)
    return stats

normalizeOBJ(IN, OUT)