import  h5py
import numpy as np
import pdb

with h5py.File("test_vis_ques_att.h5", "r") as f:
    images_id = f.get("images_id").value
    questions_id= f.get("questions_id").value
    vis1 = f.get("vis1").value
    ques1 = f.get("ques1").value
images_box = np.load("test_image_box_top_36.npy").item()
pdb.set_trace()
bbox = []
# f.create_dataset("vis1",data=vis1)
# f.create_dataset("ques1", data=ques1)
# f.create_dataset("images_id", data=images_id)
# f.create_dataset("questions_id", data=questions_id)
for i in range(len(images_id)):
    img_id = images_id[i]
    ques_id = questions_id[i]
    box = images_box[img_id]
    bbox.append(box)

with h5py.File("test_vis_ques_att_box.h5", "w") as f:
    f.create_dataset("vis1", data=vis1)
    f.create_dataset("ques1", data=ques1)
    f.create_dataset("images_id", data=images_id)
    f.create_dataset("questions_id", data=questions_id)
    f.create_dataset("boxes", data=bbox)