python train.py \
--dataroot datasets \
--name test2 \
--model pix2pixHD \
--which_model_netG unet_256 \
--which_direction BtoA \
--lambda_A 100 \
--dataset_mode aligned \
--use_spp \
--no_lsgan \
--norm batch