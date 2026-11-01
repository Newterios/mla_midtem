# Endterm report — apartment asking-price prediction (SE-2424, Balgabekov & Kaiyrbekuly)

Full text, code and figures: `Endterm_SE-2424_Balgabekov_Kaiyrbekuly.ipynb`. This file collects the tables.

**Chosen by 5-fold CV MAE:** Gradient boosting (tuned) — test MAE **6.29 M KZT** (95 % bootstrap 5.56–7.14), test MdAPE **9.13 %** (8.31–9.88), R² 0.855. Midterm linear model on the same 963 test ads: 8.66 M, 11.27 %, R² 0.712. Registration goal (median error < 10 %): reached.
Settings: {'min_samples_leaf': 5, 'max_leaf_nodes': 7, 'max_iter': 600, 'max_depth': None, 'learning_rate': 0.12, 'l2_regularization': 0.0}.

## Every model, same folds, same test split
| model                                           | CV MAE (M KZT)   |   CV MdAPE % |   CV R² |   Test MAE (M KZT) |   Test RMSE (M KZT) |   Test MdAPE % |   Test R² | within ±10 %   | within ±20 %   |
|:------------------------------------------------|:-----------------|-------------:|--------:|-------------------:|--------------------:|---------------:|----------:|:---------------|:---------------|
| Gradient boosting (tuned)                       | 6.62 ± 0.35      |          8.9 |   0.857 |              6.291 |               13.89 |           9.13 |     0.855 | 54%            | 84%            |
| XGBoost (tuned)                                 | 6.64 ± 0.32      |          8.8 |   0.852 |              6.269 |               13.55 |           8.94 |     0.862 | 55%            | 85%            |
| Gradient boosting (tuned) + K-Means segment     | 6.65 ± 0.31      |          8.9 |   0.855 |              6.319 |               13.66 |           9.13 |     0.86  | 54%            | 83%            |
| LightGBM (tuned)                                | 6.75 ± 0.28      |          9   |   0.841 |              6.193 |               12.78 |           9    |     0.877 | 55%            | 84%            |
| Blend: tuned forest + tuned boosting            | 6.95 ± 0.33      |          9   |   0.83  |              6.462 |               13.92 |           8.68 |     0.854 | 55%            | 84%            |
| Hist. gradient boosting (default)               | 7.01 ± 0.38      |          9.2 |   0.823 |              6.624 |               14.15 |           9.06 |     0.849 | 54%            | 85%            |
| Gradient boosting (default: 100 trees, depth 3) | 7.87 ± 0.33      |         10.5 |   0.796 |              6.94  |               13.86 |          10    |     0.856 | 50%            | 80%            |
| Random forest (tuned)                           | 7.99 ± 0.38      |         10.2 |   0.774 |              7.222 |               14.85 |           9.98 |     0.834 | 50%            | 80%            |
| Random forest, 300 trees (default)              | 8.00 ± 0.37      |         10.2 |   0.772 |              7.206 |               14.93 |           9.78 |     0.832 | 51%            | 79%            |
| Extra trees (500, leaf 2)                       | 8.69 ± 0.43      |         11.5 |   0.75  |              7.642 |               16.6  |          10.43 |     0.793 | 49%            | 77%            |
| NN: 64                                          | 8.82 ± 0.60      |         11.4 |   0.75  |              8.327 |               18.78 |          10.99 |     0.735 | 47%            | 75%            |
| sklearn MLP 128-64 (tanh, alpha 0.1)            | 8.86 ± 0.53      |         12.2 |   0.765 |              8.266 |               16.18 |          11.85 |     0.803 | 44%            | 74%            |
| NN: 256-128-64                                  | 8.86 ± 0.41      |         11.9 |   0.761 |              8.535 |               15.07 |          11.91 |     0.829 | 43%            | 72%            |
| NN: 128-64                                      | 8.90 ± 0.60      |         11.9 |   0.757 |              8.498 |               15.92 |          11.54 |     0.81  | 44%            | 71%            |
| Midterm linear (log price)                      | 9.10 ± 0.61      |         12.1 |   0.753 |              8.66  |               19.58 |          11.27 |     0.712 | 45%            | 73%            |
| NN: no hidden layer (= linear on log price)     | 9.26 ± 0.57      |         11.9 |   0.711 |              8.794 |               20.37 |          11.58 |     0.688 | 44%            | 74%            |
| Midterm KNN size+location (k=10)                | 10.18 ± 0.46     |         13.7 |   0.679 |              9.754 |               19.49 |          13.7  |     0.715 | 37%            | 66%            |
| Midterm decision tree (depth 9)                 | 10.19 ± 0.50     |         13.3 |   0.662 |              9.421 |               18.19 |          13.46 |     0.751 | 41%            | 67%            |
| District median price/m² (realtor rule)         | 10.45 ± 0.39     |         14.6 |   0.688 |              9.945 |               20.29 |          14.35 |     0.691 | 37%            | 62%            |
| KNN (k=10, distance), no PCA                    | 11.13 ± 0.46     |         15.6 |   0.636 |             10.71  |               21.01 |          16.12 |     0.668 | 32%            | 60%            |
| Decision tree, fully grown                      | 11.25 ± 0.73     |         14.2 |   0.59  |             10.615 |               19.58 |          14.29 |     0.712 | 37%            | 64%            |
| KNN (k=10, distance) + PCA(0.9)                 | 11.48 ± 0.61     |         16   |   0.605 |             11.21  |               21.5  |          16.32 |     0.653 | 31%            | 59%            |
| Dummy (median)                                  | 20.81 ± 0.75     |         32.1 |  -0.082 |             20.402 |               37.68 |          33.06 |    -0.067 | 19%            | 32%            |

## K-Means segments (training rows; price columns computed after clustering)
| segment   |   ads |   area_m2 |   rooms |   floor |   floors_in_building |   year_built |   lat |   lon |   fresh_renovation_share |   price_m2_kKZT |   median_price_M |   share_above_80M |
|:----------|------:|----------:|--------:|--------:|---------------------:|-------------:|------:|------:|-------------------------:|----------------:|-----------------:|------------------:|
| Alm-0     |   385 |     86.96 |       3 |       3 |                    8 |         2020 | 43.22 | 76.87 |                     0.35 |          779.73 |            67.5  |              0.36 |
| Alm-1     |   331 |     57    |       2 |       4 |                    6 |         2022 | 43.31 | 76.99 |                     0.26 |          632.17 |            36.5  |              0.03 |
| Alm-2     |   531 |     47.5  |       2 |       5 |                    9 |         2024 | 43.24 | 76.82 |                     0.35 |          722.5  |            35.5  |              0.02 |
| Alm-3     |   319 |     51    |       2 |       3 |                    5 |         1976 | 43.23 | 76.89 |                     0.24 |          766.13 |            37    |              0.03 |
| Alm-4     |   283 |     80    |       3 |      12 |                   16 |         2022 | 43.23 | 76.89 |                     0.39 |          848.21 |            65    |              0.32 |
| Ast-0     |   182 |     56.1  |       2 |       4 |                    6 |         2006 | 51.16 | 71.44 |                     0.2  |          477.23 |            26    |              0    |
| Ast-1     |   584 |     92    |       3 |       6 |                   10 |         2022 | 51.12 | 71.42 |                     0.31 |          666.67 |            59.99 |              0.27 |
| Ast-2     |   325 |     69.74 |       2 |      13 |                   18 |         2022 | 51.13 | 71.42 |                     0.33 |          650    |            42.9  |              0.13 |
| Ast-3     |   912 |     45.12 |       2 |       6 |                   10 |         2024 | 51.12 | 71.42 |                     0.28 |          659.34 |            30.8  |              0    |

## Error of the chosen model by band, city, rooms and segment (test rows)
|                    |   n |   MAE_M |   MdAPE |   median_signed |   Midterm_MdAPE |
|:-------------------|----:|--------:|--------:|----------------:|----------------:|
| price_band=<20M    |  45 |     2.4 |    11.1 |             9.4 |            12.6 |
| price_band=20–35M  | 320 |     2.8 |     8.1 |             3.1 |            10   |
| price_band=35–50M  | 275 |     4.1 |     7.8 |            -1.4 |             9.6 |
| price_band=50–80M  | 220 |     7.3 |    10.7 |            -2.9 |            12.4 |
| price_band=80–150M |  88 |    16.2 |    12.9 |            -7.1 |            21.9 |
| price_band=>150M   |  15 |    58.4 |    16.6 |           -13.9 |            31.8 |
| city=Almaty        | 463 |     7   |     8.7 |            -0.7 |            11.2 |
| city=Astana        | 500 |     5.6 |     9.4 |             0.2 |            11.3 |
| rooms=1.0          | 200 |     2.8 |     8.9 |             1   |             9.9 |
| rooms=2.0          | 389 |     4.3 |     8.1 |             0.5 |            11.3 |
| rooms=3.0          | 307 |     7.4 |     9.7 |            -0.7 |            11.4 |
| rooms=4.0          |  57 |    16.8 |    12.4 |            -4.5 |            18.3 |
| rooms=5.0          |  10 |    59.6 |    21.8 |            -7.5 |            25.2 |
| segment=Alm-0      |  92 |    14.8 |     9.2 |             0.8 |            11.5 |
| segment=Alm-1      |  95 |     4.9 |     8.8 |             1.2 |            11.2 |
| segment=Alm-2      | 145 |     3.8 |     8.3 |            -1.3 |            11.3 |
| segment=Alm-3      |  65 |     4   |     8.4 |             1.4 |             9.2 |
| segment=Alm-4      |  66 |     9.4 |     9.8 |            -4.6 |            12.4 |
| segment=Ast-0      |  41 |     3.2 |    11.1 |             4.8 |            11   |
| segment=Ast-1      | 148 |     9.4 |    10.1 |            -1.2 |            13.6 |
| segment=Ast-2      |  85 |     6.4 |    10.9 |             3.6 |            13   |
| segment=Ast-3      | 226 |     3.3 |     8.1 |            -0.1 |            10.2 |

## Neural network: gradient flow with depth (Practice 8A measurement, fold 1)
Untrained networks of 64 units per hidden layer, one `backward()` on the inner-training rows; then trained with early stopping on the same fold.
|   hidden layers | activation   |   grad norm, first hidden layer |   grad norm, last hidden layer |   ratio first / last |   best inner-val MAE (M KZT), fold 1 |   best epoch |   epochs run |
|----------------:|:-------------|--------------------------------:|-------------------------------:|---------------------:|-------------------------------------:|-------------:|-------------:|
|               1 | relu         |                       0.415     |                        0.415   |            1         |                                8.683 |           25 |           55 |
|               1 | sigmoid      |                       0.1955    |                        0.1955  |            1         |                                8.323 |          264 |          294 |
|               3 | relu         |                       0.06789   |                        0.06159 |            1.102     |                                8.625 |           30 |           60 |
|               3 | sigmoid      |                       0.003935  |                        0.1255  |            0.03135   |                                8.35  |          285 |          300 |
|               6 | relu         |                       0.003063  |                        0.02476 |            0.1237    |                                8.631 |           28 |           58 |
|               6 | sigmoid      |                       9.394e-06 |                        0.1704  |            5.512e-05 |                               20.33  |           10 |           35 |

## Other numbers
- K-Means: k = {'Astana': 4, 'Almaty': 5}; DBSCAN eps = 1.40, 29 noise ads in training; PCA: 7 components for 90 % of the variance of the 10 numeric fields.
- Over-pricing flag (> 20 % above the estimate): 10% of the test ads.
- Features (24): area, log_area, rooms, kitchen_area, floor, total_floors, year_built, ceiling_height, n_photos, lat, lon, city, district, building_type, condition, bathroom, furnished, balcony, parking, former_dorm, seller_type, has_complex, mortgaged, is_urgent.
