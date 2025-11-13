**Findings in the project:**
* VehGas does not help with PCA.
* Region One-hot-encoding adds too much dimensionality and no variance explanation.
* Ordinal encoding of **Area** brings almost no explained variability --> better excluded.
* We tried encoding Region as the mean density of .groupby(by = 'Region'), adds no explainability --> better removed.
* 