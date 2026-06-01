# ---
# --- NEW EXPERIMENTS
# ---

### 1ST MODIFICATION ###

# input < output (input 128, output 256), 1 scale only, ...yes

# training

## (reducing the batch size to prevent OOM error)
## (included nohup & to prevent the process from stopping upon logout, and logging to a custom file)

nohup python train_heatmap.py \
  --modification 1 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 1.0 \
  --experiment 6 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 1 \
  ../BoxCars116k \
  >> modif1_exp6.txt 2>&1 &

  # started training 2026-04-15 20:22:17
  ## that failed on the 17th, the server was back up on the 18th
  # started training 2026-04-19 09:55:32
  ## finished 20th in the evening / 21st in the morning
  # tail -f modif1_exp6.txt
  ## to see the logs

# Model evaluation - extraction of the vanishing points

# using the CUDA variable to use gpu 1, but then it has to be 0 in the parameters because tensorflow remapped gpu 1 to gpu 0

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcs_heatmap \
  --modification 1 \
  --skip 10 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 1.0 \
  --experiment 6 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcp_heatmap \
  --modification 1 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 1.0 \
  --experiment 6 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# Model evaluation - extracting the camera calibration file

python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_256out_1.0s_2n_4b_256c_6_r60

# second part

python -m eval.eval_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark

### PREVIEWING THE RESULTS ###

CUDA_VISIBLE_DEVICES=1 python preview_heatmap.py \
  --modification 1 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 1.0 \
  --experiment 6 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark/frames/S01/000

# it stopped cuz of an error, but i have 69M of data, so good enough
# run the command below, but NOT from jupiter lol
scp -r michalikova36@jupiter.dai.fmph.uniba.sk:/home/m/michalikova36/deep_vp/preview_heatmap_visualise /Users/lejka/deep_vp_extra
# the heatmap was a blurry yellow highlighted car, not good
# changes for att2 mostly include changing heatmap_pred[-1] to heatmap_pred[0]
# att3 is me including --resume 60 into the command, it fixed stuff, breh

### 2ND MODIFICATION ###

# training

nohup python train_heatmap.py \
  --modification 2 \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 1.0 \
  --experiment 7 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 1 \
  --resume 8 \
  ../BoxCars116k \
  >> modif2_exp7.txt 2>&1 &

# started training 2026-04-22 16:28:25
# training crashed during epoch 9/60 at 2026-04-24 02:49:48
# gonna continue the training from epoch 8 (to train the 9th epoch and onwards) - adding the resume parameter
# restarted training 2026-04-24 10:31:56
# finished on the 30th ?? myb earlier, not later

# Model evaluation - extraction of the vanishing points

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcs_heatmap \
  --modification 2 \
  --skip 10 \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 1.0 \
  --experiment 7 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcp_heatmap \
  --modification 2 \
  --batch_size 2 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 1.0 \
  --experiment 7 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# Model evaluation - extracting the camera calibration file

python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_256out_1.0s_2n_2b_256c_7_r60

# second part

python -m eval.eval_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark

### 2.1 MODIFICATION ### ((kinda))

# same as experiment 7, but the scale is 0.03 instead of 1.0

# training

nohup python train_heatmap.py \
  --modification 2 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 \
  --experiment 8 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 1 \
  ../BoxCars116k \
  >> modif2-1_exp8.txt 2>&1 &

  # started training 2026-05-01 22:16:50
  # finished sometime in 2026-05-06

# Model evaluation - extraction of the vanishing points

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcs_heatmap \
  --modification 2 \
  --skip 10 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 \
  --experiment 8 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcp_heatmap \
  --modification 2 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 \
  --experiment 8 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# Model evaluation - extracting the camera calibration file

python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_256out_0.03s_2n_4b_256c_8_r60

# second part

python -m eval.eval_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark

### 3RD MODIFICATION ###

# training

nohup python train_heatmap.py \
  --modification 3 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 9 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 1 \
  ../BoxCars116k \
  >> modif3_exp9.txt 2>&1 &

  # started training 2026-05-11 13:16
  # finished 2026-05-14 shortly after midnight

# Model evaluation - extraction of the vanishing points

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcs_heatmap \
  --modification 3 \
  --skip 10 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 9 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcp_heatmap \
  --modification 3 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 9 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# Model evaluation - extracting the camera calibration file

python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_128out_0.03-0.1-0.3-1.0s_2n_4b_256c_9_r60

# second part

python -m eval.eval_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark

# previewing the results

CUDA_VISIBLE_DEVICES=1 python preview_heatmap.py \
  --modification 3 \
  --batch_size 4 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 9 \
  --input_size 128 \
  --heatmap_size 128 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed/dataset/session0_center/video.avi

scp -r michalikova36@jupiter.dai.fmph.uniba.sk:/home/m/michalikova36/deep_vp/preview_heatmap_visualise/att4_bcs/second_part /Users/lejka/deep_vp_extra/preview_heatmap_visualise/att4_bcs/second_part
scp -r michalikova36@jupiter.dai.fmph.uniba.sk:/home/m/michalikova36/deep_vp/preview_heatmap_visualise/att4_bcs/fourth_part /Users/lejka/deep_vp_extra/preview_heatmap_visualise/att4_bcs/fourth_part

### 4TH MODIFICATION ###

# training

# ran it with batch size 4 at first, that one is not finished

nohup python train_heatmap.py \
  --modification 4 \
  --batch_size 8 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 10 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 1 \
  ../BoxCars116k \
  >> modif4_exp10.txt 2>&1 &

  # started training 2026-05-22 12:53:12
  # finished 20206-05-23 kinda before midnight

# Model evaluation - extraction of the vanishing points

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcs_heatmap \
  --modification 4 \
  --skip 10 \
  --batch_size 8 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 10 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

  # veryyyy slow, the GPU is barely being used, meaning the CPU
    # is doing most of the work, meaning the heatmaps are
    # probably problematic, and the vanishing point were not
    # detected very precisely
  # did not finish running

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcp_heatmap \
  --modification 4 \
  --batch_size 8 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 10 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

  # again, very slow, did not finish running


### 5TH MODIFICATION ###

# training

nohup python train_heatmap.py \
  --modification 5 \
  --batch_size 16 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --learning_rate 0.001 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 11 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 1 \
  ../BoxCars116k \
  >> modif5_exp11.txt 2>&1 &

  # started training 2026-05-24 02:34:52
  # finished training 2026-05-25 13:00 cca

# Model evaluation - extraction of the vanishing points

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcs_heatmap \
  --modification 5 \
  --skip 10 \
  --batch_size 16 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 11 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../2016-ITS-BrnoCompSpeed

CUDA_VISIBLE_DEVICES=1 python -m eval.extract_vp_bcp_heatmap \
  --modification 5 \
  --batch_size 16 \
  --epochs 60 \
  --perspective_sigma 25.0 \
  --crop_delta 10 \
  --scales 0.03 0.1 0.3 1.0 \
  --experiment 11 \
  --input_size 128 \
  --heatmap_size 256 \
  --gpu 0 \
  --resume 60 \
  ../BrnoCarPark

# Model evaluation - extracting the camera calibration file

python -m eval.extract_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark \
  VPout_VP1VP2b_pd_aug_25.0ps_10cd_128in_256out_0.03-0.1-0.3-1.0s_2n_16b_256c_11_r60

# second part

python -m eval.eval_calib \
  ../2016-ITS-BrnoCompSpeed \
  ../BrnoCarPark