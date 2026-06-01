# training (og parameters)
python train_heatmap.py -b 32 -ps 25.0 -cd 10 -lr 0.001 -exp 0 -e 60 ../BoxCars116k
# ((Load the model trained in experiment 0 at epoch 60, then continue training for 15 more epochs with a smaller LR, and save it as experiment 1.))
python train_heatmap.py -b 32 -ps 25.0 -cd 10 -lr 0.0001 -expr 0 -r 60 -exp 1 -e 75 ../BoxCars116k

# the same but with full argument names
python train_heatmap.py --batch_size 32 --perspective_sigma 25.0 --crop_delta 10 --learning_rate 0.001 --experiment 0 --epochs 60 ../BoxCars116k
python train_heatmap.py --batch_size 32 --perspective_sigma 25.0 --crop_delta 10 --learning_rate 0.0001 --experiment-resume 0 --resume 60 --experiment 1 --epochs 75 ../BoxCars116k


# training (custom parameters to see if it works)
# adding the gpu parameter
python train_heatmap.py --gpu 0 --batch_size 2 -ps 25.0 -cd 10 -lr 0.001 -exp 0 --epochs 2 ../BoxCars116k


# find my python PID:
ps -u $michalikova36 | grep python
# see if its here
nvidia-smi | grep python

# the full gpu table
watch -n 1 nvidia-smi


# view the logs in logs/
# 1) from the linux, in the directory above logs:
tensorboard --logdir logs
# 2) from my machine, from anywhere:
ssh -L 6006:localhost:6006 michalikova36@jupiter.dai.fmph.uniba.sk
# 3) open http://localhost:6006 on my machine


# nohup utils
# watch the logs live
tail -f nohup.out
# stop it - get the pid and kill it
pgrep -af train_heatmap.py
kill  pid


# training (my first modification - 128 x 128 heatmaps, 4 scales) (smaller batch size and less epochs for testing)
python train_heatmap.py \
  --batch_size 2 \
  --epochs 2 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 2 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  ../BoxCars116k

  # ^^^ running is expected to take almost an hour?! (print below)
    460/46466 ━━━━━━━━━━━━━━━━━━━━ 54:38 71ms/step - 0_out_loss: 0.1878 - 0_out_mean_acc: 0.0000e+00 - 1_out_loss: 0.1513 - 1_out_mean_acc: 0.0000e+00 - loss: 0.3391

# training (my first modification - 128 x 128 heatmaps, 4 scales)
python train_heatmap.py \
  --batch_size 32 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 3 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  ../BoxCars116k

  # unsuccessful run, ran out of memory:
  2026-01-11 00:02:21.219090: W external/local_xla/xla/tsl/framework/bfc_allocator.cc:501] Allocator (GPU_0_bfc) ran out of memory trying to allocate 18.39GiB (rounded to 19749918464)requested by op
  2026-01-11 00:02:21.221684: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 19749918304 bytes.
  Out of memory while trying to allocate 19749918304 bytes.

# training (my first modification - 128 x 128 heatmaps, 4 scales) (reducing the batch size to prevent OOM error)
## other suggested actions - reduce --num_stacks, reduce number of scales
## included nohup & to prevent the process from stopping upon logout
# GOOD ONE
nohup python train_heatmap.py \
  --batch_size 8 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 3 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  ../BoxCars116k &

  # started training ^^^ 20:25 ish, sat, finished 20h ish later

# training (my 2nd modification - 256 x 256 heatmaps, 4 scales)
# (reducing the batch size even more to prevent OOM error)
# GOOD ONE
nohup python train_heatmap.py \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 4 \
  --input_size 256 \
  --heatmap_size 256 \
  --gpu 0 \
  ../BoxCars116k &

  # started training ^^^ 1:05 ish, thu

# training (my 3rd modification - 256 x 256 heatmaps, 1 scale)
# (the grid of sie 256 x 256 is not that coarse, so 1 not too sharp but not too broad scale might be enough)
# GOOD ONE
nohup python train_heatmap.py \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.1 \
  --experiment 5 \
  --input_size 256 \
  --heatmap_size 256 \
  --gpu 0 \
  ../BoxCars116k &

  # started training ^^^ 20:40 ish, wed


# running the Model evaluation - extracting vanishing points
# --resume 75 is used to work with a trained version of the model, not untrained (at epoch 0)
# added the module flag to run it as a module to avoid python setting its internal search path to the eval folder, ignoring other folders
python -m eval.extract_vp_bcp_heatmap \
  --batch_size 32 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --experiment 1 \
  --resume 75 \
  ../BrnoCarPark

# i realised i need to match the parameters for the correct model to be chosen
# same parameters but -learning rate and +resume
# also reducing resume to 60 because that is the maximum i have

# 1st modification
# Model evaluation - extracting vanishing points
python -m eval.extract_vp_bcp_heatmap \
  --batch_size 8 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 3 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# 2nd modifications
python -m eval.extract_vp_bcp_heatmap \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 4 \
  --input_size 256 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# 3rd modification
# running 59 here because of the unfinished training
python -m eval.extract_vp_bcp_heatmap \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.1 \
  --experiment 5 \
  --input_size 256 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 58 \
  ../BrnoCarPark


# 1st modification
# Model evaluation - extracting vanishing points
# preserving the skip argument, adjusting the rest
python -m eval.extract_vp_bcs_heatmap \
  --skip 10 \
  --batch_size 8 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 3 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed


# 2nd modification
python -m eval.extract_vp_bcs_heatmap \
  --skip 10 \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 4 \
  --input_size 256 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

# 3rd modification
python -m eval.extract_vp_bcs_heatmap \
  --skip 10 \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.1 \
  --experiment 5 \
  --input_size 256 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 58 \
  ../2016-ITS-BrnoCompSpeed

# Model evaluation - extracting the camera calibration file
# 1st modification
python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_128out_0.03-0.1-0.3-1.0s_2n_8b_256c_3_r60

# 2nd modification
python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.03-0.1-0.3-1.0s_2n_2b_256c_4_r60

# 3rd modification
python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.1s_2n_2b_256c_5_r58

# second part (the same command for all 3 modifications)
python -m eval.eval_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark

# FINAL RESULTS
**************************
Eval BrnoCompSpeed
**************************
For system_VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.03-0.1-0.3-1.0s_2n_2b_256c_4_r60_0.1c.json mean rel err: 321.49546876297353, median rel err 49.109653871683946, mean abs err 2.983194555992494, median abs err 0.4487845313890681
For system_VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_128out_0.03-0.1-0.3-1.0s_2n_8b_256c_3_r60_0.1c.json mean rel err: 22.027559588609165, median rel err 17.239224367753295, mean abs err 0.25619601375737616, median abs err 0.1660280505683765
For system_VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.1s_2n_2b_256c_5_r58_0.1c.json mean rel err: 83.0127757494037, median rel err 46.542880378723225, mean abs err 0.915505249227244, median abs err 0.418471597753573
**************************
Eval BrnoCarPark
**************************
For system_VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.03-0.1-0.3-1.0s_2n_2b_256c_4_r60_0.1c.json mean rel err: 0.35456478659345547, median rel err 0.32286122328633104, mean abs err 5.381366157184036, median abs err 4.624973889356742
For system_VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_128out_0.03-0.1-0.3-1.0s_2n_8b_256c_3_r60_0.1c.json mean rel err: 0.18055680816136463, median rel err 0.15845322754146646, mean abs err 2.5421490554805666, median abs err 2.078881187831315
For system_VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.1s_2n_2b_256c_5_r58_0.1c.json mean rel err: 0.6074633152735058, median rel err 0.6416890244464539, mean abs err 9.17023156143448, median abs err 7.546752409446318

---

python -m eval.extract_calib \
  --debug \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_256in_256out_0.1s_2n_2b_256c_5_r58

- go to the server
- make a directory
- use python to start a webserver


# ---
# --- NEW EXPERIMENTS in the other file
# ---