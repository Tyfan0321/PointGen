# Full Evaluation Comparison: global_std vs data_aug

This comparison uses only the currently saved full evaluation summary files under
the two `eval` directories. Runs with `samples-*` are ignored.

## Shared-Step Comparison

For `global_std`, multiple checkpoints were evaluated. This table uses the best
`global_std` checkpoint by `RR` for each dataset and inference step. The
`data_aug` run has one evaluated checkpoint, `epoch-200`.

| Dataset | Steps | global_std ckpt | global_std RR | global_std RRE | global_std RTE | global_std dist | data_aug ckpt | data_aug RR | data_aug RRE | data_aug RTE | data_aug dist | Delta RR |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 3DMatch | 3 | epoch-134 | 0.7671 | 10.8582 | 0.3710 | 0.3094 | epoch-200 | 0.8417 | 7.9406 | 0.2696 | 0.2493 | +0.0746 |
| 3DMatch | 5 | epoch-134 | 0.7843 | 10.7781 | 0.3662 | 0.3347 | epoch-200 | 0.8447 | 7.8976 | 0.2617 | 0.3242 | +0.0604 |
| 3DMatch | 10 | epoch-124 | 0.8164 | 10.3589 | 0.3316 | 0.2781 | epoch-200 | 0.8718 | 7.4443 | 0.2354 | 0.1989 | +0.0555 |
| 3DLoMatch | 3 | epoch-134 | 0.3346 | 37.7303 | 1.1783 | 1.0137 | epoch-200 | 0.4217 | 30.7899 | 0.9394 | 0.8055 | +0.0870 |
| 3DLoMatch | 5 | epoch-134 | 0.3431 | 37.6316 | 1.1734 | 1.0185 | epoch-200 | 0.4262 | 30.9741 | 0.9311 | 0.8545 | +0.0831 |
| 3DLoMatch | 10 | epoch-134 | 0.3953 | 37.0160 | 1.1341 | 0.9819 | epoch-200 | 0.4789 | 30.0694 | 0.8876 | 0.7763 | +0.0837 |

## Best Full Result Per Run

| Dataset | Run | Best ckpt | Steps | RR | RRE | RTE | dist_error |
|---|---|---|---:|---:|---:|---:|---:|
| 3DMatch | global_std | epoch-124 | 10 | 0.8164 | 10.3589 | 0.3316 | 0.2781 |
| 3DMatch | data_aug | epoch-200 | 10 | 0.8718 | 7.4443 | 0.2354 | 0.1989 |
| 3DLoMatch | global_std | epoch-134 | 10 | 0.3953 | 37.0160 | 1.1341 | 0.9819 |
| 3DLoMatch | data_aug | epoch-200 | 10 | 0.4789 | 30.0694 | 0.8876 | 0.7763 |
