class MLPPartial(object):
    def __init__(self,input_shape,
                 layer_units=[3000,2000],
                 activations = 'relu',
                 output_shapes=[10,2],
                 output_names=None,):
        
        if isinstance(activations,str):
            activations = [activations]*len(layer_units)
        
        if output_names is None:
            output_names = ['output_{}'.format(i+1) for i in range(len(output_shapes))]

        self.input_shape = input_shape
        self.layer_units = layer_units
        self.output_shapes = output_shapes
        self.output_names = output_names
        self.activations = activations
        self.input = Input(shape=self.input_shape,name='encoder_input')

    def build(self,):
        x = self.input
        net = Dense(self.layer_units[0], activation=self.activations[0])(x)

        i = 1
        for units,act in zip(self.layer_units[1:],self.activations[1:]):
            net = Dense( units, activation=act, name='dense_{}'.format(i) )(net)
            i+=1
        
        self.outputs = [Dense(units,activation='linear',name=name)(net) for units,name in zip(self.output_shapes,self.output_names)]
        mod = Model(inputs=self.input,outputs=self.outputs)
        return mod