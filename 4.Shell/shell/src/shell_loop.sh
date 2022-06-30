#!/bin/bash
input_dir=${1}
output_dir=${2}

bhw=(0.8 1.0 1.2)
asca=(1.0 1.1 1.2 1.3 1.4)  # default 1.2
gamma=(0.01 0.02 0.03 0.04)          # default 0.01, 0.06
lambda=(0.5 0.55 0.6 0.65)                    # default 0.6
bhw_for_num=$[${#bhw[@]}-1]
asca_for_num=$[${#asca[@]}-1]
gamma_for_num=$[${#gamma[@]}-1]
lambda_for_num=$[${#lambda[@]}-1]

echo "harden coef table coef number: $bhw_for_num"
echo "scatter asca coef number: $asca_for_num"
echo "scatter gamma coef number: $gamma_for_num"
echo "scatter lambda coef number: $lambda_for_num"
echo "test number: $[$[$bhw_for_num+1]*$[$asca_for_num+1]*$[$gamma_for_num+1]*$[$lambda_for_num+1]]"

# copy harden calibration table
harden_table_src_dir="/home/nv/gaojunbao/data/20220630/References/KV120_Body_HardenCorrTable/*" 
harden_table_parent_dir="/home/nv/gaojunbao/data/20220630/BH_table/"
# for ((i=0;i<=$bhw_for_num;i++))
# do
#     harden_table_dst_path="$harden_table_parent_dir""${bhw[i]}""/"
#     if [ ! -d $harden_table_dst_path ]; then
#         mkdir -p $harden_table_dst_path
#     fi
    
#     cp $harden_table_src_dir $harden_table_dst_path 
# done

# correct_pipeline.xml file path
correct_pipeline_file_path="./Res/config/correct_pipeline.xml"
run_recon_file_path="./Res/config/runRecon.xml"
# 

# loop BHW coef
for ((i=0;i<=$bhw_for_num;i++))
do
    # loop scatter asca coef
    for ((j=0;j<=$asca_for_num;j++))
    do
        # loop scatter gamma coef
        for ((k=0;k<=$gamma_for_num;k++))
        do
            # loop scatter lambda coef
            for ((l=0;l<=$lambda_for_num;l++))
            do

                # combination the result directory according these coefs
                echo "current coef: bhw=${bhw[i]}, asca=${asca[j]}, gamma=${gamma[k]}, lammda=${lambda[l]}"

                # result directory
                result_dir_name="$input_dir""bhw_""${bhw[i]}""_asca_${asca[j]}""_gamma_""${gamma[k]}""_lambda_""${lambda[l]}""/"
                if [ ! -d $result_dir_name ]; then
                    mkdir -p $result_dir_name
                fi

                # harden table path
                harden_table_curr_dir="$harden_table_parent_dir""${bhw[i]}""/KV120_Body_HardenCorrTable/"

                # change the correct_pipeline.xml to this loop parameters
                # change the bhw table path
                sed -i '18s#<HardenImageDir>.*</HardenImageDir>#<HardenImageDir>'$harden_table_curr_dir'</HardenImageDir>#' $correct_pipeline_file_path
                # change the asca
                sed -i '51s#<asca>.*</asca>#<asca>'${asca[j]}'</asca>#' $correct_pipeline_file_path
                # change the lambda
                sed -i '49s#<lambda>.*</lambda>#<lambda>'${lambda[l]}'</lambda>#' $correct_pipeline_file_path
                # change the gamma
                sed -i '50s#<gamma>.*</gamma>#<gamma>'${gamma[k]}'</gamma>#' $correct_pipeline_file_path

                # change the result dir
                sed -i '5s#<OutputDir>.*</OutputDir>#<OutputDir>'$result_dir_name'</OutputDir>#' $run_recon_file_path
                sed -i '5s#<OutputDir>.*</OutputDir>#<OutputDir>'$result_dir_name'</OutputDir>#' $correct_pipeline_file_path

                # execute RunRecon
                ./RunRecon 
            done
        done
    done
done
