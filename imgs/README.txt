Figures for the paper.

- Fig. 1 (pipeline): drawn inline as native TikZ in main.tex -> NO image file needed.
- maest_v2_training_curve.png: Fig. 2 (pre-training loss + LR schedule). Real
  asset; regenerated via scripts/plot_training_curve.py (axes cropped to the
  trained range, dashed line at the submitted epoch-130 checkpoint).
- Qualitative figure (Sec 4.6): export 4 axial-slice PNGs and uncomment the
  skeleton in main.tex:
    qual_success_abdomen.png, qual_failure_abdomen.png,
    qual_success_chest.png,   qual_failure_chest.png
