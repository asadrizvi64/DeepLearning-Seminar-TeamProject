# Rule check (radius 0.6 D)

## 1a. Detector dependence replicates on embryo 2
- s02 t150: NOT EVALUABLE -- Luxar fits in 55-94x: 0, JPEG-XL: 1
- s02 t180: NOT EVALUABLE -- Luxar fits in 55-94x: 0, JPEG-XL: 1
**Verdict: NOT EVALUABLE**

## 1b. Beyond 120x, Luxar keeps more nuclei than JPEG2000 (embryo 2, both detectors)
- s02 t150 LoG: Luxar 269x 1.03 vs JPEG2000 221x 0.53 
- s02 t150 LoG: Luxar 227x 1.03 vs JPEG2000 221x 0.53 
- s02 t150 LoG: Luxar 136x 1.03 vs JPEG2000 110x 1.01 
- s02 t150 Cellpose: Luxar 269x 0.87 vs JPEG2000 221x 0.39 
- s02 t150 Cellpose: Luxar 227x 0.88 vs JPEG2000 221x 0.39 
- s02 t150 Cellpose: Luxar 136x 0.87 vs JPEG2000 110x 1.02   <-- Luxar not better
- s02 t180 LoG: Luxar 260x 0.99 vs JPEG2000 221x 0.48 
- s02 t180 LoG: Luxar 218x 1.01 vs JPEG2000 221x 0.48 
- s02 t180 LoG: Luxar 131x 1.02 vs JPEG2000 110x 0.95 
- s02 t180 Cellpose: Luxar 260x 0.82 vs JPEG2000 221x 0.32 
- s02 t180 Cellpose: Luxar 218x 0.84 vs JPEG2000 221x 0.32 
- s02 t180 Cellpose: Luxar 131x 0.87 vs JPEG2000 110x 0.96   <-- Luxar not better
**Verdict: DOES NOT REPLICATE** (12 size-matched pairs within 1.25x)

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
- s02 t150: 2.3/nuc 0.85, 4.1/nuc 0.98, 6.0/nuc 1.01, 9.6/nuc 1.03, 13.3/nuc 1.03, 30.3/nuc 1.03
- s02 t180: 1.3/nuc 0.66, 2.3/nuc 0.87, 3.4/nuc 0.93, 5.4/nuc 0.99, 7.5/nuc 1.01, 16.9/nuc 1.02
**Verdict (frame 100): NOT CONTRADICTED -- never below 90% (lowest budget 3.1/nuc)**

## 3. Luxar's own budget K*
- s01 t150: K* = 64000 (signal_limited, still climbing), ~30x: LoG 1.005, Cellpose 0.965
- s01 t194: K* = 64000 (signal_limited, still climbing), ~28x: LoG 1.006, Cellpose 0.983
**Verdict: SAFE for blob detection -- limit the PSNR-blind claim to hand-chosen budgets**
