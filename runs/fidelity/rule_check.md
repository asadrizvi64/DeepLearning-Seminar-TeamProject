# Rule check (radius 0.6 D)

## 1a. Detector dependence replicates on embryo 2
- s02 t150: JPEG-XL 0.987 vs Luxar 0.901 -> gap 8.6 points
- s02 t180: JPEG-XL 0.956 vs Luxar 0.893 -> gap 6.3 points
**Verdict: REPLICATES**

## 1b. Beyond 120x, Luxar keeps more nuclei than JPEG2000 (embryo 2, both detectors)
- s02 t150 LoG: Luxar 441x 0.85 vs JPEG2000 440x 1.01   <-- Luxar not better
- s02 t150 LoG: Luxar 377x 0.98 vs JPEG2000 377x 1.01   <-- Luxar not better
- s02 t150 LoG: Luxar 329x 1.01 vs JPEG2000 330x 1.00 
- s02 t150 LoG: Luxar 269x 1.03 vs JPEG2000 269x 1.00 
- s02 t150 LoG: Luxar 227x 1.03 vs JPEG2000 227x 1.00 
- s02 t150 LoG: Luxar 136x 1.03 vs JPEG2000 135x 1.00 
- s02 t150 Cellpose: Luxar 441x 0.74 vs JPEG2000 440x 0.90   <-- Luxar not better
- s02 t150 Cellpose: Luxar 377x 0.81 vs JPEG2000 377x 0.93   <-- Luxar not better
- s02 t150 Cellpose: Luxar 329x 0.84 vs JPEG2000 330x 0.94   <-- Luxar not better
- s02 t150 Cellpose: Luxar 269x 0.87 vs JPEG2000 269x 0.97   <-- Luxar not better
- s02 t150 Cellpose: Luxar 227x 0.88 vs JPEG2000 227x 0.95   <-- Luxar not better
- s02 t150 Cellpose: Luxar 136x 0.87 vs JPEG2000 135x 0.96   <-- Luxar not better
- s02 t180 LoG: Luxar 434x 0.66 vs JPEG2000 434x 1.01   <-- Luxar not better
- s02 t180 LoG: Luxar 370x 0.87 vs JPEG2000 370x 1.01   <-- Luxar not better
- s02 t180 LoG: Luxar 321x 0.93 vs JPEG2000 321x 1.01   <-- Luxar not better
- s02 t180 LoG: Luxar 260x 0.99 vs JPEG2000 260x 1.01   <-- Luxar not better
- s02 t180 LoG: Luxar 218x 1.01 vs JPEG2000 219x 1.00 
- s02 t180 LoG: Luxar 131x 1.02 vs JPEG2000 131x 1.01 
- s02 t180 Cellpose: Luxar 434x 0.62 vs JPEG2000 434x 0.85   <-- Luxar not better
- s02 t180 Cellpose: Luxar 370x 0.78 vs JPEG2000 370x 0.84   <-- Luxar not better
- s02 t180 Cellpose: Luxar 321x 0.83 vs JPEG2000 321x 0.87   <-- Luxar not better
- s02 t180 Cellpose: Luxar 260x 0.82 vs JPEG2000 260x 0.92   <-- Luxar not better
- s02 t180 Cellpose: Luxar 218x 0.84 vs JPEG2000 219x 0.91   <-- Luxar not better
- s02 t180 Cellpose: Luxar 131x 0.87 vs JPEG2000 131x 0.92   <-- Luxar not better
**Verdict: DOES NOT REPLICATE** (24 size-matched pairs within 1.25x)

## 1c. JPEG-XL keeps >= 97% at every size it reaches (embryo 2, both detectors)
- s02 t150 LoG: min 0.994 over 5 sizes (14-118x)
- s02 t150 Cellpose: min 0.967 over 5 sizes (14-118x)
- s02 t180 LoG: min 1.000 over 4 sizes (14-106x)
- s02 t180 Cellpose: min 0.925 over 4 sizes (14-106x)
**Verdict: QUALIFY** (worst 0.925)

## 2. Cliff sits below ~5 splats per labelled nucleus (LoG)
- s01 t100: 3.1/nuc 1.10, 6.5/nuc 1.10, 10.4/nuc 1.08, 17.2/nuc 1.07, 26.0/nuc 1.07, 58.7/nuc 1.06
- s01 t150: 1.7/nuc 0.87, 5.5/nuc 0.97, 13.3/nuc 1.01, 30.2/nuc 1.00, 56.4/nuc 1.01, 103.7/nuc 1.00, 209.1/nuc 1.01
- s01 t194: 1.0/nuc 0.67, 3.2/nuc 0.91, 7.3/nuc 0.96, 16.4/nuc 0.97, 30.9/nuc 0.98, 56.6/nuc 1.00, 114.9/nuc 1.01
- s02 t150: 2.3/nuc 0.85, 4.1/nuc 0.98, 6.0/nuc 1.01, 9.6/nuc 1.03, 13.3/nuc 1.03, 30.3/nuc 1.03, 53.6/nuc 1.04, 115.0/nuc 1.04
- s02 t180: 1.3/nuc 0.66, 2.3/nuc 0.87, 3.4/nuc 0.93, 5.4/nuc 0.99, 7.5/nuc 1.01, 16.9/nuc 1.02, 29.7/nuc 1.03, 63.9/nuc 1.04
**Verdict (frame 100): NOT CONTRADICTED -- never below 90% (lowest budget 3.1/nuc)**

## 3. Luxar's own budget K*
- s01 t150: K* = 64000 (signal_limited, still climbing), ~30x: LoG 1.005, Cellpose 0.965
- s01 t194: K* = 64000 (signal_limited, still climbing), ~28x: LoG 1.006, Cellpose 0.983
**Verdict: SAFE for blob detection -- limit the PSNR-blind claim to hand-chosen budgets**

## 5. Luxar vs JPEG2000 at the same bytes (paired 95% CI over nuclei)
- cellpose: 36 pairs on 5 frames -- JPEG2000 better 25, no difference 11, Luxar better 0
- log: 36 pairs on 5 frames -- JPEG2000 better 11, no difference 14, Luxar better 11
    - Luxar better: E1 t100 luxar_K8000 159x, +8.3 points [+2.4, +15.2]
    - Luxar better: E1 t100 luxar_K4000 263x, +10.7 points [+4.5, +18.3]
    - Luxar better: E1 t100 luxar_K3000 323x, +9.5 points [+3.4, +16.7]
    - Luxar better: E1 t100 luxar_K2000 395x, +9.5 points [+3.5, +16.9]
    - Luxar better: E1 t100 luxar_K1500 459x, +8.3 points [+2.4, +15.4]
    - Luxar better: E1 t100 luxar_K1000 537x, +9.5 points [+3.4, +16.9]
    - Luxar better: E2 t150 luxar_K32000 47x, +3.4 points [+1.1, +6.2]
    - Luxar better: E2 t150 luxar_K16000 90x, +4.5 points [+1.1, +8.4]
    - Luxar better: E2 t150 luxar_K8000 136x, +3.4 points [+0.5, +6.8]
    - Luxar better: E2 t150 luxar_K4000 227x, +3.4 points [+0.5, +6.9]
    - Luxar better: E2 t180 luxar_K32000 46x, +2.7 points [+0.9, +4.9]
- watershed: 36 pairs on 5 frames -- JPEG2000 better 23, no difference 13, Luxar better 0
**Verdict: Luxar better somewhere -- the paper names these combinations**
