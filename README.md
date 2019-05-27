# StringMatching-Pytorch

Usage:

```python3 main.py --dataset_train=... --dataset_test=...```

Arguments:

```--dataset_train:``` one of [geonames, jrc_organization, jrc_person, historical_places]

```--dataset_test:``` one of [geonames, jrc_organization, jrc_person, historical_places]

```--batch_size:``` int, default 32

```--hidden_units:``` int, default 60

```--bidirectional:``` bool, default True

```--self_attention:``` bool, default False

```--max_pooling:``` bool, default False

```--alignment:``` bool, default False

```--shortcut:``` bool, default False
