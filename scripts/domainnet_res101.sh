# DomainNet
########################## CAF-A ResNet-101
# source domain clipart_train
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domian infograph_train
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain painting_train
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain quickdraw_train
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain real_train
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain sketch_train
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_A.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0


########################## CAF-D ResNet-101
# source domain clipart_train
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/clipart_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domian infograph_train
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/infograph_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain painting_train
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/painting_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain quickdraw_train
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/quickdraw_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain real_train
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/real_train.txt --t_dset_path data/list/domainnet/sketch_train.txt --t_test_path data/list/domainnet/sketch_test.txt --SEED 0

# source domain sketch_train
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/clipart_train.txt --t_test_path data/list/domainnet/clipart_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/infograph_train.txt --t_test_path data/list/domainnet/infograph_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/painting_train.txt --t_test_path data/list/domainnet/painting_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/quickdraw_train.txt --t_test_path data/list/domainnet/quickdraw_test.txt --SEED 0
python train_CAF_D.py --net ResNet101 --dset domainnet --max_interval 50001 --output_dir results/res101 --s_dset_path data/list/domainnet/sketch_train.txt --t_dset_path data/list/domainnet/real_train.txt --t_test_path data/list/domainnet/real_test.txt --SEED 0
