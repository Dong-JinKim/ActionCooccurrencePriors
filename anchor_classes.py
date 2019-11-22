import numpy as np
import pickle
import torch
import os
import utils.io as io
import pdb


proc_dir=os.path.join(os.getcwd(),'data_symlinks/hico_processed')
object_list_json = os.path.join(proc_dir,'object_list.json')
verb_list_json = os.path.join(proc_dir,'verb_list.json')
hoi_list_json = os.path.join(proc_dir,'hoi_list.json')

object_list = io.load_json_object(object_list_json)
verb_list = io.load_json_object(verb_list_json)
hoi_list = io.load_json_object(hoi_list_json)

obj_list = {obj['name']:oid for oid,obj in enumerate(object_list)}
v_list = {verb['name']:vid for vid,verb in enumerate(verb_list)}


weight = torch.load('co-occurrence_verb.pkl')    
weight = torch.Tensor(weight)  
weight=weight.sum(0) 
mask = torch.eye(117,117).byte()
weight.masked_fill_(mask,0.0)

## Code for cluster HOI
'''
Groups = []
for cid in range(600):
    neighbors = [ii.item() for ii in torch.where(weight[cid]>0)[0]]  
    
    assigned=0
    for gg in Groups:
        if len(set(neighbors) & set(gg))>0:
            gg.append(cid)
            assigned=1
            break
    if assigned==0:
        Groups.append([cid])
'''

## Code for cluster verb

count = sum((weight>0).float())
_ , YY = count.sort()  
 
########################################################### -----start ----- ##########################################  
MAX_anchors = 10

Groups = []
occupied = []
location = [0]*117
anchors = []
## STEP1
for cid in YY:#[:36]:# :11 2  :20 3
    if len(weight[cid,anchors].nonzero())>0:
        ind = weight[anchors,cid].argmax()
        neighbor = anchors[ind]
        #current_location = location[neighbor]
        #print(current_location==ind.item())
        #gg = Groups[current_location]
        gg = Groups[ind.item()]

        gg.append(cid.item())
        #location[cid.item()]=current_location

            #print(neighbor)
    else:
        #location[cid.item()]=len(Groups)
        if len(Groups)<MAX_anchors:
            Groups.append([cid.item()])
            anchors.append(cid.item())
        else:
            gg = Groups[-1]
            gg.append(cid.item())
    #occupied.append(cid.item())


for gg in Groups:#visualize
    print([verb_list[cid]['name']for cid in gg])
    


cid2gid = {cid:gid  for gid,gg in enumerate(Groups) for cid in gg}
#hoi2gid = {hid:cid2gid[v_list[hoi['verb']]]+1 for hid,hoi in enumerate(hoi_list)}#---!! for group
hoi2gid = {hid:cid2gid[v_list[hoi['verb']]]+1 if v_list[hoi['verb']] in anchors else len(Groups)-1 for hid,hoi in enumerate(hoi_list)}#for anchor

io.dump_json_object(hoi2gid,f'hoi2anchor{MAX_anchors}.json') 


## save {(groups+regular) to typical verb list}
regulars = [ii for ii in range(117) if ii not in anchors[:-1]]
cid2gid = {cid:iid for iid, cid in enumerate(anchors[:-1]+regulars)}
gid_list = [cid2gid[cid] for cid in range(117)]
io.dump_json_object(gid_list,f'gid2cid{MAX_anchors}.json') 







