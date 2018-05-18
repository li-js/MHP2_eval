import pickle
import numpy as np
from tqdm import tqdm
import cv2
import os, sys
import argparse

parser = argparse.ArgumentParser(description='Eval MHP-v2')

parser.add_argument('--plot', dest='plot', default=False, help='Whether to plot the results')
parser.add_argument('--sparse', dest='sparse', default=False, help='Use sparse matrix to save memory footprint')
parser.add_argument('--cache_pkl', dest='cache_pkl', default=False, help='Cache results to save memory footprint')

args = parser.parse_args()

data_root = '/home/lijianshu/MultiPerson/data/LV-MHP-v2/'   # Path to your MHP-V2 dataset
cache_pkl_path = './tmp'                                    # Path to the cached_pkl

if args.cache_pkl and (not os.path.isdir(cache_pkl_path)):
    os.makedirs(cache_pkl_path)

#set_list = ['test_all', 'test_inter_top20', 'test_inter_top10']
set_list = ['val']
skip_header = 0

meta_results = {}
for f in open('results.txt').readlines()[skip_header:]:
    items=f.strip().split()
    key = items[0]
    items = items[1:]
    assert(len(items)%2==0)
    num = len(items)/2
    persons = []
    for k in xrange(num):
        persons.append([int(items[2*k]), float(items[2*k+1])])

    meta_results[key] = persons


results_all = {}
for key in tqdm(meta_results, desc='Generating results ..'):
    persons = meta_results[key]

    global_seg = cv2.imread('global_seg/{}.png'.format(key), cv2.IMREAD_UNCHANGED) # make sure this is a 2d np array
    global_tag = cv2.imread('global_tag/{}.png'.format(key), cv2.IMREAD_UNCHANGED) # make sure this is a 2d np array

    results = {} 
    dets, masks = [], []
    for p_id, score in persons:
        mask = (global_tag == p_id)
        if np.sum(mask)==0: continue
        seg = mask*global_seg
        ys, xs = np.where(mask>0)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        #x1,y1,x2,y2=0,0,0,0
        dets.append((x1, y1, x2, y2, score))
        masks.append(seg)            

    # Reuiqred Field of each result: a list of masks, each is a multi-class masks for one person.
        # It can also be sparsified to [scipy.sparse.csr_matrix(mask) for mask in masks] to save memory cost
    results['MASKS']= masks if not args.sparse else [scipy.sparse.csr_matrix(mask) for mask in masks]
    # Reuiqred Field of each result, a list of detections corresponding to results['MASKS']. 
    results['DETS'] = dets    

    if args.cache_pkl:
        results_cache_add = cache_pkl_path + key + '.pklz'
        pickle.dump(results, gzip.open(results_cache_add, 'w'))
        results_all[key] = results_cache_add
    else:
        results_all[key]=results

    if args.plot:
        import pylab as plt
        plt.figure('seg')
        plt.imshow(global_seg)
        print('Seg unique:'+str(np.unique(global_seg)))
        plt.figure('tag')
        plt.imshow(global_tag)      
        print('Tag unique:'+str(np.unique(global_tag)))
        plt.show()


import eval_mhp
import mhp_data
final_results = {}

for set_ in set_list:    
    final_results[set_] = {}
    dat_list = mhp_data.get_data(data_root, set_)
    results_all_i = {}
    for dat in dat_list:
        key = os.path.basename(dat['filepath']).split('.')[0]
        results_all_i[key] = results_all[key]

    final_results[set_]['ap_list'], final_results[set_]['pcp_list'] = [], []
    for thres in [float(i)/10 for i in range(1,10)]:
        ap_seg, pcp = eval_mhp.eval_seg_ap(results_all_i, dat_list, nb_class=59, ovthresh_seg=thres, task_id=set_, Sparse=args.sparse, From_pkl=args.cache_pkl)
        final_results[set_]['ap_list'].append(ap_seg)
        final_results[set_]['pcp_list'].append(pcp)


pickle.dump(final_results, open('metrics.pkl', 'w'))
