# bash create_sorted_data.sh
set -e
top_freqs=(64 128 256 512 1024 2048 4096)

data_process_type=shuffle_unique
for ((i=0;i<${#top_freqs[@]};++i));do
    top_freq=${top_freqs[i]}
    python3.6 create_sorted_data.py --data_process_type $data_process_type \
                                    --top_freq $top_freq
done