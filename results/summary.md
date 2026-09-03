# Reproduction summary

| class | variants | ideas | PC1 share | in-sample days | hold-out days | reflected |
|---|---|---|---|---|---|---|
| 1 | 16 | 9 | 0.48 | 3,130 | 2,180 | ['f4', 'f12'] |
| 2 | 10 | 4 | 0.51 | 1,565 | 2,180 | [] |
| 3 | 6 | 2 | 0.53 | 1,565 | 2,180 | [] |
| 4 | 20 | 8 | 0.30 | 3,130 | 2,180 | ['f10', 'f13', 'f16', 'f19', 'f20'] |

## Purged-CV mean IC (in-sample)

| method | c1 | c2 | c3 | c4 |
|---|---|---|---|---|
| cluster_eq_tuned | 0.0314 | 0.0208 | 0.0125 | 0.0193 |
| cluster_ic_tuned | 0.0312 | 0.0184 | 0.0103 | 0.0201 |
| naive_averaged | 0.0310 | 0.0208 | 0.0125 | 0.0165 |
| eb_weight | 0.0309 | 0.0210 | 0.0124 | 0.0183 |
| ic_weight | 0.0309 | 0.0173 | 0.0108 | 0.0198 |
| ridge_tuned | 0.0308 | 0.0090 | 0.0141 | 0.0160 |
| pls_tuned | 0.0308 | 0.0133 | 0.0146 | 0.0163 |
| cluster_eq | 0.0308 | 0.0216 | 0.0124 | 0.0193 |
| cluster_eb | 0.0308 | 0.0216 | 0.0124 | 0.0197 |
| uniqueness_reg | 0.0306 | 0.0132 | 0.0114 | 0.0186 |
| pca_pc1 | 0.0305 | 0.0183 | 0.0125 | 0.0033 |
| uniqueness_tuned | 0.0287 | 0.0091 | 0.0145 | 0.0174 |

## Hold-out mean IC (frozen on in-sample)

| method | c1 | c2 | c3 | c4 |
|---|---|---|---|---|
| naive_averaged | 0.0118 (t 1.88) | 0.0146 (t 2.43) | 0.0133 (t 2.64) | 0.0192 (t 3.94) |
| ic_weight | 0.0119 (t 1.88) | 0.0145 (t 2.47) | 0.0135 (t 2.47) | 0.0154 (t 2.86) |
| pca_pc1 | 0.0112 (t 1.78) | 0.0123 (t 2.18) | 0.0133 (t 2.66) | 0.0063 (t 1.64) |
| cluster_eq | 0.0119 (t 1.89) | 0.0175 (t 2.66) | 0.0133 (t 2.64) | 0.0194 (t 3.69) |
| cluster_eb | 0.0119 (t 1.89) | 0.0175 (t 2.66) | 0.0133 (t 2.64) | 0.0163 (t 3.06) |
| cluster_eb_shaped | 0.0154 (t 2.45) | 0.0147 (t 3.06) | 0.0027 (t 0.62) | 0.0183 (t 3.59) |
| cluster_eq_shaped | 0.0155 (t 2.50) | 0.0151 (t 3.16) | 0.0027 (t 0.62) | 0.0188 (t 3.72) |
| selected (gated) | 0.0154 (t 2.45) | 0.0175 (t 2.66) | 0.0133 (t 2.64) | 0.0194 (t 3.69) |

### Gate, class 1

```
ADOPT shapes
  A  accuracy non-inferiority: oof diff -0.0003, NW t -0.10 (need > -2) -> pass
  B  tradeability win: ret_5d 0.728 vs 0.583, turn_5d 0.365 vs 0.602 -> pass
```

### Gate, class 2

```
REFUSE shapes (keep the linear composite)
  A  accuracy non-inferiority: oof diff -0.0218, NW t -3.33 (need > -2) -> fail
  B  tradeability win: ret_5d 0.960 vs 0.970, turn_5d 0.059 vs 0.033 -> fail
```

### Gate, class 3

```
REFUSE shapes (keep the linear composite)
  A  accuracy non-inferiority: oof diff -0.0108, NW t -1.70 (need > -2) -> pass
  B  tradeability win: ret_5d 0.951 vs 0.906, turn_5d 0.065 vs 0.026 -> fail
```

### Gate, class 4

```
REFUSE shapes (keep the linear composite)
  A  accuracy non-inferiority: oof diff -0.0074, NW t -2.97 (need > -2) -> fail
  B  tradeability win: ret_5d 0.928 vs 0.898, turn_5d 0.103 vs 0.159 -> pass
```
