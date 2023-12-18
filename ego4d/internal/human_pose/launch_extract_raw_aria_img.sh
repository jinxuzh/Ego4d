method=automatic # MODIFY
batch_job_name=Ego4D_preprocess
cfg=dev_release_base
min_subclip_length=1
cache_root_dir=/mnt/volume2/Data/Ego4D/aria_raw_images
data_dir=/mnt/volume2/Data/Ego4D
partition=learnaccel
run_type=local
steps=preprocess_aria
selected_frames_dir=/mnt/volume2/Data/Ego4D/annotations/ego_pose/hand/selected_frames_info_${method}
TAKES=""

i=0
for take_name in $TAKES
do
    echo "============================ [$((++i))] $take_name starts ============================"
    taskset --cpu-list 16-31 python launch_extract_raw_aria_img.py \
    --base_work_dir /mnt/volume2/Data/Ego4D \
    --batch_job_name ${batch_job_name} \
    --config-name ${cfg} \
    --partition ${partition} \
    --run_type ${run_type} \
    --steps ${steps} \
    --take_name ${take_name} \
    cache_root_dir=${cache_root_dir} \
    data_dir=${data_dir} \
    inputs.min_subclip_length=${min_subclip_length} \
    inputs.selected_frames_dir=${selected_frames_dir} \
    repo_root_dir=/home/jinxu/code/Ego4d
    echo ============================ $take_name finished ============================ $'\n\n'
done
