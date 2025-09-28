Berkeley tabular modeling project
Personal work that applies leakage safe modeling and honest validation

![Repository structure](assets/repo_structure.png)

I built a compact pipeline that classifies income on the Adult dataset while keeping data handling strictly leakage safe. 

1 I evaluate only out of fold predictions then pick a decision threshold using F1 on those same held out scores

2 I include a label shuffle control to show the pipeline does not learn spurious noise

3 I report permutation importances on raw columns for interpretability without peeking

#Quick start:

1 make setup

2 make run

3 open the results folder to review the artifacts

#Expected outputs

1 metrics.json with out of fold AUC and F1 at the chosen threshold

2 per_fold.csv with fold level AUC accuracy and F1

3 importances.csv with permutation importances for raw columns

4 top_importances.png with the bar chart shown below

![Example feature importances](assets/permutation_importances_example.png)

#Technical choices

1 Data loading comes from OpenML on first run and is cached to data or you can drop adult.csv into that folder
2 All preprocessing is inside a scikit learn pipeline to keep fitting scoped to folds
3 Cross validation uses stratified folds and the grid is deliberately small so runs are fast and reproducible
4 I keep plotting and reporting minimal so the artifacts are easy to skim in a code review

#Contact
If you want to discuss the modeling choices or how I would adapt this structure to alpha research send a note.
