# Solution-FP3-English-Language-Learning
I trained a family of deberta models, after which I predicted labels for Feedback Prize 2 data and re-labeled Feedback Prize 3 data. Then I trained 1 fold deberta v3 xsmall first on fp2 data, then on fp3. I used kaggle resources and my 1060 for training. The full training cycle was about 1 hour. Text prediction took about 15 minutes on CPU


0. Edit Settings
1. Label the dataset with a large ensemble or downloads https://drive.google.com/file/d/1rfBRN6m7lS4LjlWzmqe2U0WXzPWgjgyH/view?usp=sharing
2. Start pretraining on [FP2](https://www.kaggle.com/competitions/feedback-prize-effectiveness)
`python train.py --model microsoft/deberta-v3-xsmall --lr 5e-4 --seed 42 --typ all --mod psevdo_old --max_len 512 --epochs 2`
3. Start training
`python train.py --model microsoft/deberta-v3-xsmall --chk '42_512_all_microsoft-deberta-v3-xsmall/microsoft-deberta-v3-xsmall_fold5_best.pth' --lr 5e-4 --seed 42 --typ all --mod psevdo_train --max_len 512 --epochs 3`
4. Run test `python test.py`
