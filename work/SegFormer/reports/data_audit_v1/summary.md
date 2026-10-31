# Data audit summary

- created: 2026-10-31 22:04:48 +0800
- data root: `E:\桌面\AI\segformer_resources\data\visa`
- input size: 512, ignore index: 255
- overall: **ok** in 19.26 s

## Counts

| split | normal | anomaly | total |
| --- | ---: | ---: | ---: |
| train | 482 | 48 | 530 |
| val | 120 | 12 | 132 |
| test | 402 | 40 | 442 |

## Decode

- image sizes: `{'1404x1070': 1104}`
- image modes: `{'RGB': 1104}`
- mask modes: `{'L': 100}`
- decode failures: 0

## Mask semantics

- raw value union: `[0, 1, 2, 3, 4]`
- raw value sets: `{'{0,1}': 14, '{0,1,2}': 1, '{0,2}': 46, '{0,2,3}': 5, '{0,4}': 17, '{0,2,3,4}': 1, '{0,3}': 14, '{0,2,4}': 1, '{0,3,4}': 1}`

## Anomaly foreground (% of image)

| split | min | p10 | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.07655 | 0.112629 | 0.187648 | 0.993403 | 0.522339 | 12.725857 |
| val | 0.154964 | 0.159291 | 0.315787 | 1.89295 | 6.331443 | 7.893602 |
| test | 0.090529 | 0.107703 | 0.193939 | 0.568017 | 0.696208 | 4.955601 |

## 512 letterbox

- padding fraction: 23.828125% (median)
- example 1404x1070 -> 512x390 at scale 0.364672, pads L0 T61 R0 B61
- anomalies whose 512 foreground became empty: 0
- images with decreased/increased/unchanged component counts: 12/14/74
- original components retained/lost: 309/19 of 328; images with an actual lost region: 12
- original components split: 37; output components merging original ids: 20
- 512/original foreground ratio: median 99.93862% (min 94.67396%, max 104.26173%)

## Duplicates

- exact duplicate groups: 0 (cross-split: 0)
- near-duplicate method: every cross-split pair compared on a 64x64 grayscale thumbnail: Pearson correlation >= 0.985 and mean absolute difference <= 6.0; dHash distance is reported per candidate as supporting evidence
- dHash distance histogram (<=8): `{'0': 16, '1': 148, '2': 812, '3': 2544, '4': 6179, '5': 12097, '6': 18347, '7': 24104, '8': 26971}`
- cross-split pairs compared: 362564, candidates: 1 {'train-val': 1}

## Checks

- PASS — manifests_match_split_meta
- PASS — all_images_decoded
- PASS — single_image_size
- PASS — images_rgb
- PASS — masks_single_channel_L
- PASS — mask_sizes_match
- PASS — all_anomalies_have_mask
- PASS — no_empty_foreground
- PASS — normals_have_no_mask_field
- PASS — normal_512_labels_zero_inside_ignore_outside
- PASS — foreground_survives_512
- PASS — counts_match_protocol
