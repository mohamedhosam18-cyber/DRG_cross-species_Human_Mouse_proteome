# Cross-Species Proteomic Analysis of Human and Mouse Dorsal Root Ganglia

This repository contains the computational workflow used for a cross-species proteomic comparison of human and mouse dorsal root ganglia (DRG).

The analysis investigates conserved and divergent molecular signatures relevant to sensory neuroscience, pain research, and translational comparison between human and mouse DRG.

## Workflow

The numbered scripts should be run in order:

1. Prepare human proteomic data
2. Prepare mouse proteomic data
3. Build ortholog mapping table
4. Map proteins to orthologs
5. Build merged human–mouse matrix
6. Perform cross-species analysis
7. Generate volcano plots
8. Extract top differentially abundant proteins
9. Run enrichment analysis
10. Compare enrichment results
11. Generate enrichment visualizations
12. Generate final cross-species figures
13. Run PCA and distribution analysis
14. Run GSEA
15. Visualize GSEA results
16. Generate GSEA UpSet plots
17. Generate heatmaps
18. Generate concordance scatter plots
19. Generate thresholded UpSet plots
20. Validate transcriptomic marker panels
21. Generate radar plots for module activity convergence

## Notes

This workflow is intended as a reproducible analysis companion to the manuscript. Cross-species comparisons are based on normalized proteomic matrices and ortholog-mapped proteins.
