

import torch
import pandas as pd

p1 = "/mnt/8T/hou/multicard_CSFEL/weights/binclass/net-WholeNet_traindb-ff-c23-720-140-140_face-scale_size-299_seed-43_note-DF_v3/last.pth"
p2 = "/mnt/8T/hou/multicard_gen/weights/binclass/net-WholeNet_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-Gtest2_02_2/it001000.pth"
p3 = "/mnt/8T/hou/multicard_gen/weights/binclass/net-WholeNet_traindb-ff-c23-500-140-140_face-scale_size-299_seed-22_note-Gtest2_01_4/it000001.pth"
model1 = torch.load(p1)
model2 = torch.load(p2)


# df = pd.DataFrame(columns=['part', 'change'])
# count = 0
# for item1, item2 in zip(model1['model'].items(), model2['model'].items()):

#     part = item1[0]
#     if("module.judge.j_linear.weight" not in part):
#         continue
#     change = abs((item1[1]-item2[1])/item1[1])
#     df.loc[count] = [part, change.numpy()]
#     count = count + 1

# df.to_csv("2.csv")
# print(1)


total = sum([param.nelement() for param in model1.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))
