import numpy as np
import h5py


f = h5py.File(r"C:\Users\14380\Desktop\scRNA\scRNA参考数据数据\Dataset.Human_Chrom8_train_test.h5")

with h5py.File(r"C:\Users\14380\Desktop\scRNA\scRNA参考数据数据\Dataset.Human_Chrom8_train_test.h5", 'r') as f:
    print(list(f.keys()))
    # 读入数据集
    dset1 = f['celltype']
    dset2 = f['test_gene']
    dset3 = f['test_label']
    dset4 = f['train_data']
    dset5 = f['train_gene']

    print(dset1)

    # 转化为numpy格式
    np_dset1 = np.array(dset1)
    np_dset2 = np.array(dset2)
    np_dset3 = np.array(dset3)
    np_dset4 = np.array(dset4)
    np_dset5 = np.array(dset5)
    print(np_dset1)
    print(np_dset2)
    print(np_dset3)
    print(np_dset4)
    print(np_dset5)