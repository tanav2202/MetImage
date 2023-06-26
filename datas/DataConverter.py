"""
Convert raw MS data into MetImage

"""
import pandas as pd
import numpy as np
from pyteomics import mzxml
import glob
from scipy import sparse
import os
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from tqdm import tqdm


def binningMZ(mzList, intList, mzmin=100, mzmax=1000, binSize=0.01):
    """
    Binning m/z from .mzml

    :param mzList: a list of m/z value
    :param intList: a list of intensity (same length as mzlist)
    :param mzmin: the minimum value of m/z bin (custom)
    :param mzmax: the maximum value of m/z bin (custom)
    :param binSize: the Da of every bin in m/z binning (custom)
    :return: A table of binned m/z and intensity
    """
    BinNumber = (mzmax - mzmin) / binSize
    BinIndex = round((pd.Series(mzList.tolist()) - mzmin) / binSize)
    BinTable = pd.DataFrame({'index': BinIndex, 'intensity': intList.tolist()})
    BinTable['index'] = BinTable['index'].astype(int)
    BinTable = BinTable.groupby('index').sum()
    full_index = range(int(BinNumber))
    BinTable = BinTable.reindex(full_index)
    BinTable = BinTable.fillna(value=0)
    return BinTable.iloc[:, 0]


def ParseSpec(spec, mzmin=100, mzmax=1000, binSize=0.01):
    """
    Convert the spec data into MetImage matrix

    :param spec: spec data generated from
    :return: MetImage matrix
    """
    scan = spec['id']
    mzList = spec['m/z array']
    intList = spec['intensity array']
    BinTable = binningMZ(mzList, intList, mzmin=mzmin, mzmax=mzmax, binSize=binSize)
    BinTable.columns = [scan]
    return BinTable


def ConvertMetImage(file_path, mzmin=100, mzmax=1000, binSize=0.01, Threads=6, save_path="."):
    """
    Generate MetImage from a raw MS data (.mzml) file

    :param file_path: pathway of the input file
    :param mzmin: the minimum value of m/z bin (custom)
    :param mzmax: the maximum value of m/z bin (custom)
    :param binSize: the Da of every bin in m/z binning (custom)
    :param Threads: Threads used for multiprocessing (custom)
    :param save_path: pathway of output data (custom)
    :return: MetImage, the whole metabolome profiling image (.npz)
    """
    reader = mzml.read(file_path)
    filename = os.path.splitext(os.path.basename(file_path))[0]

    pool = ThreadPool(Threads)
    ParseSpec_partial = partial(ParseSpec, mzmin=mzmin, mzmax=mzmax, binSize=binSize)
    MetImage = pool.map(ParseSpec_partial, list(reader))
    MetImage = np.array(MetImage).T
    pool.close()
    pool.join()

    SparseTable = sparse.csr_matrix(MetImage, dtype=np.float32)
    sparse.save_npz(os.path.join(save_path, filename + ".npz"), SparseTable)

    del MetImage
    del reader


def ConvertDataset(rawdata_dir,pattern="mzXML",mzmin=60, mzmax=1200, binSize=0.01, Threads=6, save_path = "."):
    print(rawdata_dir)
    """
    Convert whole dataset into whole metabolome profiling images (.npz).

    :param rawdata_dir: pathway of input dataset dir. (custom)
    :param pattern: MS data format. (custom, default mzXML)
    :param Threads: Threads used for multiprocessing. (custom)
    :param save_path: pathway of output data. (custom)
    :return: MetImage, whole metabolome profiling image (.npz)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Groups = os.listdir(rawdata_dir)
    print(Groups)
    if Groups is None:
        ConvertMetImage(rawdata_dir, pattern=pattern, mzmin=mzmin, mzmax=mzmax, binSize=binSize,
                        Threads=Threads, save_path=save_path)
    else:
        for Group in Groups:
            print(Group)
            wd = rawdata_dir + "/" + Group
            if not os.path.exists(wd):
                os.makedirs(wd)
            ConvertMetImage(wd, pattern=pattern, mzmin=mzmin, mzmax=mzmax, binSize=binSize,
                            Threads=Threads, save_path=save_path+"/"+Group)
