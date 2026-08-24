# DCB-CDR

Anonymous repository for paper submission.

---

## Environment

```bash
pip install -r requirements.txt
```

---

## Training

Example:

```bash
python entry.py \
  --gpu 0 \
  --experiment DCB_CDR \
  --exp_part dcb_CDR \
  --aggregation aggregation \
  --RQVAE True \
  --rq_num 0 \
  --set_aggr item_iu \
  --diff_lr 0.01 \
  --mapping_lambda 0.01 \
  --uniformity_loss 0.1 \
  --recon_loss 0.01 \
  --ratio '[0.2, 0.8]' \
  --task 1
```

---

## Logging

To save logs:

```bash
TODAY=$(date +%Y%m%d)
LOG_FILE=task1

python entry.py \
  --gpu 0 \
  --experiment DCB_CDR \
  --exp_part dcb_CDR \
  --aggregation aggregation \
  --RQVAE True \
  --rq_num 0 \
  --set_aggr item_iu \
  --diff_lr 0.01 \
  --mapping_lambda 0.01 \
  --uniformity_loss 0.1 \
  --recon_loss 0.01 \
  --ratio '[0.2, 0.8]' \
  --task 1 \
  > out/$TODAY/${LOG_FILE}.log 2>&1 &
```