# visda2017
## SEED 0
########################## CAF-A ResNet-101
python train_CAF_A.py --net ResNet101 --dset visda2017 --test_interval 5000 --output_dir results/res101 --s_dset_path data/list/visda2017/train_list.txt --t_dset_path data/list/visda2017/val_list.txt --t_test_path data/list/visda2017/val_list.txt --K 4 --SEED 0

########################## CAF-D ResNet-101
python train_CAF_D.py --net ResNet101 --dset visda2017 --test_interval 5000 --output_dir results/res101 --s_dset_path data/list/visda2017/train_list.txt --t_dset_path data/list/visda2017/val_list.txt --t_test_path data/list/visda2017/val_list.txt --K 4 --SEED 0
