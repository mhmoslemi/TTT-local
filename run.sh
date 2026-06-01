# # python train.py --num-circles 10 --target 1.6294 \
# CUDA_VISIBLE_DEVICES=5 python train.py --num-circles 2 \
#                 --num-steps 5 \
#                 --groups-per-step 2 \
#                 --group-size 2 \
#                 --max-new-tokens 500 \
#                 --temperature 1.1 \
#                 --max-seq-length 40000 \
#                 --model-name LiquidAI/LFM2.5-350M
#             #    --model-name LiquidAI/LFM2.5-1.2B-Base \
# #                --model-name LiquidAI/LFM2.5-350M


# python train.py --num-circles 10 --target 1.6294 \
python train_multy.py --num-circles 26 \
                --num-steps 3 \
                --groups-per-step 8 \
                --group-size 64 \
                --max-new-tokens 6700 \
                --temperature 1 
                # \
                # --max-seq-length 40000 \
                # --model-name openai/gpt-oss-120b
            #    --model-name LiquidAI/LFM2.5-1.2B-Base \
#                --model-name LiquidAI/LFM2.5-350M
