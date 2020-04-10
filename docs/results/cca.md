---
# remote_theme: pmarsceill/just-the-docs
title: Canonical Correlation Analysis

---


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
pca_80 = pd.read_pickle('../data/cca/pca_80fve.pk')
```


```python
pca_80.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>component</th>
      <th>fve</th>
      <th>fve_ratio</th>
      <th>arch</th>
      <th>depth</th>
      <th>layer</th>
      <th>objective</th>
      <th>cum_fve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>69.628601</td>
      <td>0.059599</td>
      <td>conv</td>
      <td>4</td>
      <td>conv_4</td>
      <td>no-recon</td>
      <td>0.059599</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>57.645645</td>
      <td>0.049343</td>
      <td>conv</td>
      <td>4</td>
      <td>conv_4</td>
      <td>no-recon</td>
      <td>0.108942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>44.470741</td>
      <td>0.038065</td>
      <td>conv</td>
      <td>4</td>
      <td>conv_4</td>
      <td>no-recon</td>
      <td>0.147007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>37.003548</td>
      <td>0.031674</td>
      <td>conv</td>
      <td>4</td>
      <td>conv_4</td>
      <td>no-recon</td>
      <td>0.178681</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>30.727896</td>
      <td>0.026302</td>
      <td>conv</td>
      <td>4</td>
      <td>conv_4</td>
      <td>no-recon</td>
      <td>0.204983</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_context('talk')
g = sns.FacetGrid(data=pca_80,row='objective',col='arch',hue='depth',sharex='col',margin_titles=True,
                  ylim=(0,1),
                  height=4, palette='plasma',legend_out=True,
                 )
# plt.xscale('log')
g.map(sns.lineplot,'component','cum_fve').add_legend()
```




    <seaborn.axisgrid.FacetGrid at 0x7f23d9984898>




![png](CCA-figures_files/CCA-figures_3_1.png)



```python
count_pca_80 = pca_80.groupby(['arch','objective','layer'])['fve'].count().reset_index().rename(columns={'fve':'n_components'})
```


```python
g = sns.FacetGrid(data=count_pca_80,col='arch',row='objective',sharex='col',sharey=False,margin_titles=True,
#                   ylim=(0,1),
                  height=4, palette='plasma',legend_out=True,
                 )
# plt.xscale('log')
g.map(plt.bar,'layer','n_components')
g.fig.autofmt_xdate(rotation=45)
```


![png](CCA-figures_files/CCA-figures_5_0.png)

