mkdir -p ./logs/
LOG_DIR=./logs/
CODE_DIR=.

arch=$1
echo "Arch: $arch"

# resnet	output X/Y
# N	CO	X	Y	CI	KH	KH              // strides
# 1	64	112	112	3	7	7           // 2   conv_expr_S2D1P0
# 1	64	56	56	64	1	1           // 1   conv_expr_S1D1P0
# 1	64	56	56	64	3	3           // 1   conv_expr_S1D1P0
# 1	256	56	56	64	1	1           // 1   conv_expr_S1D1P0
# 1	128	56	56	256	1	1           // 2   conv_expr_S2D1P0
# 1	128	28	28	128	3	3           // 1   conv_expr_S1D1P0
# 1	512	28	28	128	1	1           // 1   conv_expr_S1D1P0
# 1	256	14	14	512	1	1           // 2   conv_expr_S2D1P0
# 1	256	14	14	256	3	3           // 1   conv_expr_S1D1P0
# 1	1024	14	14	256	1	1           // 1   conv_expr_S1D1P0
# 1	512	14	14	1024	1	1           // 2   conv_expr_S2D1P0
# 1	512	7	7	512	3	3           // 1   conv_expr_S1D1P0
# 1	2048	7	7	512	1	1           // 1   conv_expr_S1D1P0

totalround=3

for ((i=0;i<$totalround;i++)); do
    echo "Round $i"

    # resnet0
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 0 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S2D1P0  \
    --shape 1	64	112	112	3	7	7 2>&1 |tee $LOG_DIR/resnet0.log

    # # resnet1
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 1 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	64	56	56	64	1	1 2>&1 |tee $LOG_DIR/resnet1.log

    # resnet2
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 2 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	64	56	56	64	3	3 2>&1 |tee $LOG_DIR/resnet2.log

    # resnet3
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 3 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	256	56	56	64	1	1 2>&1 |tee $LOG_DIR/resnet3.log

    # resnet4
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 4 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S2D1P0  \
    --shape 1	128	28	28	256	1	1 2>&1 |tee $LOG_DIR/resnet4.log

    # resnet5
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 5 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	128	28	28	128	3	3 2>&1 |tee $LOG_DIR/resnet5.log

    # resnet6
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 6 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	512	28	28	128	1	1 2>&1 |tee $LOG_DIR/resnet6.log

    # resnet7
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 7 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S2D1P0  \
    --shape 1	256	14	14	512	1	1 2>&1 |tee $LOG_DIR/resnet7.log

    # resnet8
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 8 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	256	14	14	256	3	3 2>&1 |tee $LOG_DIR/resnet8.log

    # resnet9
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 9 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	1024	14	14	256	1	1 2>&1 |tee $LOG_DIR/resnet9.log

    # resnet10
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 10 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S2D1P0  \
    --shape 1	512	7	7	1024	1	1 2>&1 |tee $LOG_DIR/resnet10.log

    # resnet11
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 11 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	512	7	7	512	3	3 2>&1 |tee $LOG_DIR/resnet11.log

    # resnet12
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network resnet --testcase 12 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P0  \
    --shape 1	2048	7	7	512	1	1 2>&1 |tee $LOG_DIR/resnet12.log
    
    mkdir -p $LOG_DIR/resnet_roller_$i
    mv $LOG_DIR/resnet*.log $LOG_DIR/resnet_roller_$i
done



# yolo

# test	N	CO	H	W	CI	KH	KH	strides
# 0	1	32	544	544	3	3	3	1
# 1	1	64	272	272	32	3	3	1
# 2	1	128	136	136	64	3	3	1
# 3	1	64	136	136	128	1	1	1
# 4	1	256	68	68	128	3	3	1
# 5	1	128	68	68	256	1	1	1
# 6	1	512	34	34	256	3	3	1
# 7	1	256	34	34	512	1	1	1
# 8	1	1024	17	17	512	3	3	1
# 9	1	512	17	17	1024	1	1	1

# yolo
for ((i=0;i<$totalround;i++)); do
    echo "Round $i"

    # yolo0
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 0 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1  \
    --shape 1	32	544	544	3	3	3 2>&1 |tee $LOG_DIR/yolo0.log

    # yolo1
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 1 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1  \
    --shape 1	64	272	272	32	3	3 2>&1 |tee $LOG_DIR/yolo1.log

    # yolo2
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 2 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1  \
    --shape 1	128	136	136	64	3	3 2>&1 |tee $LOG_DIR/yolo2.log

    # yolo3
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 3 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1  \
    --shape 1	64	136	136	128	1	1 2>&1 |tee $LOG_DIR/yolo3.log

    # yolo4
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 4 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1 \
    --shape 1	256	68	68	128	3	3 2>&1 |tee $LOG_DIR/yolo4.log

    # yolo5
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 5 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1 \
    --shape 1	128	68	68	256	1	1 2>&1 |tee $LOG_DIR/yolo5.log

    # yolo6
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 6 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1 \
    --shape 1	512	34	34	256	3	3 2>&1 |tee $LOG_DIR/yolo6.log

    # yolo7
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 7 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1 \
    --shape 1	256	34	34	512	1	1 2>&1 |tee $LOG_DIR/yolo7.log

    # yolo8
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 8 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1 \
    --shape 1	1024	17	17	512	3	3 2>&1 |tee $LOG_DIR/yolo8.log

    # yolo9
    FUSE_PAD=0 CONV_LAYOUT=NCHW  python3 -u $CODE_DIR/test_op_mp.py --runs $i --network yolo --testcase 9 --topk 50 --arch $1 --code_dir generated_source/conv --smem_tiling --reg_tiling --op conv_expr_S1D1P1 \
    --shape 1	512	17	17	1024	1	1 2>&1 |tee $LOG_DIR/yolo9.log

    mkdir -p $LOG_DIR/yolo_roller_$i
    mv $LOG_DIR/yolo*.log $LOG_DIR/yolo_roller_$i
done



# pz list:
# M	N	L
# 512	64	1024
# 512	4096	1024
# 512	1024	4096
# 512	64	768
# 512	3072	768
# 512	768	3072

# matmul
for ((i=0;i<$totalround;i++)); do
    echo "Round $i"

    python3 -u $CODE_DIR/test_op_mp.py --runs $i --network mm --testcase 0 --topk 50 --arch $1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 64   1024 2>&1 | tee $LOG_DIR/matmul0_512_64_1024.log
    python3 -u $CODE_DIR/test_op_mp.py --runs $i --network mm --testcase 1 --topk 50 --arch $1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 4096 1024 2>&1 | tee $LOG_DIR/matmul1_512_4096_1024.log
    python3 -u $CODE_DIR/test_op_mp.py --runs $i --network mm --testcase 2 --topk 50 --arch $1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 1024 4096 2>&1 | tee $LOG_DIR/matmul2_512_1024_4096.log

    python3 -u $CODE_DIR/test_op_mp.py --runs $i --network mm --testcase 3 --topk 50 --arch $1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 64   768  2>&1 | tee $LOG_DIR/matmul3_512_64_768.log
    python3 -u $CODE_DIR/test_op_mp.py --runs $i --network mm --testcase 4 --topk 50 --arch $1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 3072 768  2>&1 | tee $LOG_DIR/matmul4_512_3072_768.log
    python3 -u $CODE_DIR/test_op_mp.py --runs $i --network mm --testcase 5 --topk 50 --arch $1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 512 768  3072 2>&1 | tee $LOG_DIR/matmul5_512_768_3072.log

    mkdir -p $LOG_DIR/matmul_roller_$i
    mv $LOG_DIR/matmul*.log $LOG_DIR/matmul_roller_$i

done
