class :
   ''' Iterator class '''
   def __init__(self, team):
       # Team object reference
       self._team = team
       # member variable to keep track of current index
       self._index = 0
 
   def __next__(self):
       ''''Returns the next value from team object's lists '''
       if self._index < (len(self._team._juniorMembers) + len(self._team._seniorMembers)) :
           if self._index < len(self._team._juniorMembers): # Check if junior members are fully iterated or not
               result = (self._team._juniorMembers[self._index] , 'junior')
           else:
               result = (self._team._seniorMembers[self._index - len(self._team._juniorMembers)]   , 'senior')
           self._index +=1
           return result
       # End of Iteration
       raise StopIteration

class Evaluator(object):
    def __init__(self, n_splits=None,cv=None,train_size=0.75, x_regions=None, y_regions=None, variation=[3],sortby='image_id',random_state=None):
        self.cv = 5
        self.train_size=0.75
        self.n_splits=10
        self.random_state = random_state
        
        self._iter_idx = 0
        
        if cv is not None:
            np.arange()
            for i in range(self.cv):
                self._split()
    
    def _split(self,X, random_state)
        tr,te = train_test_split(X, train_size=self.train_size, random_state=self.random_state)
        
        return tr,te
    
    def __next__(self):
        if 
        i = 0
        for rand_delta in np.arange(cv):
            tr_idx, te_idx, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),
                                                   train_size=train_size,
                                                   random_state=np.random.randint(0,50)+rand_delta)
        cv_tr.append(tr_idx)
        cv_te.append(te_idx)
        
        

def cca(x,neural_data,layers=None, brain_region=['IT','V4'], cv=5, n_components=5, variation=[3],sortby='image_id',train_size=0.75):
    var_lookup = stimulus_set[stimulus_set.variation.isin(variation)].image_id.values
    x = x.where(x.image_id.isin(var_lookup),drop=True)
    nd = neural_data.where(neural_data.image_id.isin(var_lookup),drop=True)
    
    x = x.sortby(sortby)
    nd = nd.sortby(sortby)
    
    assert list(getattr(x,sortby).values) == list(getattr(nd,sortby).values)
    num_images = x.shape[0]
    out_recs = []
    
    cv_tr = []
    cv_te = []
    
    for rand_delta in np.arange(cv):
        tr_idx, te_idx, _,_ = train_test_split(np.arange(num_images),np.arange(num_images),train_size=train_size,random_state=np.random.randint(0,50)+rand_delta)
        cv_tr.append(tr_idx)
        cv_te.append(te_idx)
    
    for br in brain_region:
        nd_reg = nd.sel(region=br)
        
        if layers is None:
            layers = np.unique(x.region.values)
            
        for reg in layers:
            if reg == 'pixel':
                continue
            x_reg = x.sel(region=reg)
            
            depth = np.unique(x_reg.layer.values)[0]
            with tqdm(zip(np.arange(cv),cv_tr,cv_te), total=cv) as t:
                t.set_description('{}{} x {}{}'.format(reg,x_reg.shape,br,nd_reg.shape))
                
                r_mean = []
                fve_mean = []
                cca_mean = []
                for n,tr,te in t:
                    cca = CCA(n_components=n_components)
                    cca.fit(x_reg.values[tr],nd_reg.values[tr])

                    u,v = cca.transform(x_reg.values[te],nd_reg.values[te])
                    
                    y_pred = cca.predict(x_reg.values[te])
                    y_true = nd_reg.values[te]
                    
                    fve = explained_variance_score(y_true,y_pred,multioutput='uniform_average')
                    r_vals = [pearsonr(y_pred[:,i],y_true[:,i]) for i in range(y_pred.shape[-1])]
                    
                    cca_r = np.mean([pearsonr(u[:,i],v[:,i]) for i in np.arange(n_components)])

#                     r_vals = [pearsonr(ab_vec[0][:,i],ab_vec[1][:,i]) for i in range(ab_vec[0].shape[-1])]
                    
                    r_mean.append(np.mean([r for r,v in r_vals]))
                    cca_mean.append(cca_r)
                    fve_mean.append(fve)
                
                    out_recs.append({
                        'region':br,
                        'layer':reg,
                        'pearsonr': np.mean([r for r,v in r_vals]),
                        'cca_r':cca_r,
                        'fve':fve,
                        'iter':n,
                        'depth':depth,
                    })
                    
                    t.set_postfix(pearson=np.mean(r_mean), cca=np.mean(cca_mean), fve=np.mean(fve_mean))
                    
    return pd.DataFrame.from_records(out_recs)