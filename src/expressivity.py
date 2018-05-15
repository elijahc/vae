import numpy as np
# from ray.dataframe import pd

# def pairwise_correlations( g_t)

def euclidean_metric( g_t,delta):
    n_d = g_t.shape[0]
    v = np.gradient( g_t, delta, axis=0 )
    vv = np.empty( n_d, dtype=np.float32 )
    for t in np.arange( n_d ):
        vv[t] = np.dot( v[t], v[t].T )
    
    return vv

def curvature( g_t, delta ):
    n_d = g_t.shape[0]
    v = np.gradient( g_t, delta, axis=0 )
    vv = np.empty( n_d, dtype=np.float32 )
    vhat = np.empty_like( v )

    for t in np.arange( n_d ):
        vv[t] = np.dot( v[t], v[t].T )
        vhat[t] = v[t] / np.sqrt( vv[t] )
            
    k = np.empty(n_d, dtype=np.float32)
    a = np.gradient(v, delta, axis=0)

    for i in np.arange( n_d ):
        aa = np.dot( a[i], a[i].T )
        va = np.dot( v[i], a[i].T )
        k[i] = ( vv[i]**-(3/2))*np.sqrt(( vv[i]*aa)-va**2)

    return k

def grassmanian_metric(g_t,delta):
    k = curvature(g_t,delta)
    g_E = euclidean_metric(g_t,delta)
    
    return (k**2)*g_E

def curvature_length(g_t,delta,N=None):
    # Number of examples, e.g. number of theta's
    n_d = g_t.shape[0]
    g_dt = np.ediff1d(delta)
    g_E = euclidean_metric(g_t,delta)[:-1]
    dL_E= np.sqrt(g_E)*g_dt
    L_E = dL_E.sum()
    if N is not None:
        L_E = L_E/np.sqrt(N)
    return L_E
    
    
def grassmanian_length( g_t, delta ):
    n_d = g_t.shape[0]
    v = np.gradient( g_t, delta, axis=0 )
    vv = np.empty( n_d, dtype=np.float32 )
    vhat = np.empty_like( v )

    for t in np.arange( n_d ):
        vv[t] = np.dot( v[t], v[t].T )
        vhat[t] = v[t] / np.sqrt( vv[t] )
            
    
    a_hat = np.gradient( vhat, delta, axis=0)
    gauss_metric = np.array([np.dot(a_hat[i],a_hat[i].T) for i in np.arange( n_d )])
    if isinstance(delta,float):
        
        dG = np.sqrt(gauss_metric)[:-1]*delta
    elif isinstance(delta,np.ndarray):
        dG = np.sqrt(gauss_metric)[:-1]*np.ediff1d(delta)
        
    return dG.sum()

class Expressivity():
    def __init__(self,model,trajectory,delta,index=None):

        # evaluate expressivity on a specific layer if index is provided
        self.trajectory = trajectory
        self.delta = delta
        self.n_d = trajectory.shape[0]
        self.model = model

        if index is not None:
            activation_functors = gen_activation_functors(model)
            func = activation_functors[index]
            self.g_t = np.squeeze(func([self.trajectory])[0])
        else:
            self.g_t = self.model.predict(self.trajectory,batch_size=32)

        self.v = np.gradient(self.g_t,self.delta,axis=0)
        self.vv = np.empty(self.n_d,dtype=np.float32)
        self.vhat = np.empty_like(self.v)

        for t in np.arange(self.n_d):
            self.vv[t] = np.dot(self.v[t],self.v[t].T)
            self.vhat[t] = self.v[t]/np.sqrt(self.vv[t])

    def curvature(self):
        return curvature(self.g_t,self.n_d,self.delta)
    
    def curve_length(self):
        self.dL = np.sqrt(self.vv)*self.delta
        return self.dL.sum()

    def grassmanian_length(self):
        a_hat = np.gradient(self.vhat,self.delta,axis=0)
        gauss_metric = np.array([np.dot(a_hat[i],a_hat[i].T) for i in np.arange(self.n_d)])
        self.dG = np.sqrt(gauss_metric)*self.delta
        return self.dG.sum()
    
def salience(model,x_test,masks,x_iso=None):
    funcs = gen_activation_functors(model)
    outs = []
    input_G = []
    for i,mask in enumerate(masks):
        x_traj = x_test[mask]
        
        if x_iso is None:
            # Calc embedding
            print('Calculating Isomap embeddings...')
            x_traj,x_iso = gen_sorted_isomap(x_traj,n_neighbors=20,n_components=1,n_jobs=1)
        
        x_G = grassmanian_length(x_traj,delta=x_iso)
        input_G.append(x_G)
        g_t = [np.squeeze(f([x_traj])[0]) for f in funcs]
        y_G = [grassmanian_length(g,1.0/len(g)) for g in g_t]
        for l_idx,Y in enumerate(y_G):
            rec = {
                'Grassmanian Length':Y,
                'x_G':x_G,
                'Layer': l_idx+1,
                'Digit':i+1,
                'G_delta':Y-x_G
            }
            outs.append(rec)
    return pd.DataFrame.from_records(outs)

def manifold_overlap():
    pass