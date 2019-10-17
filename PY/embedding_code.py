df = pd.read_csv("Data/occurrences_train.csv",sep=';', error_bad_lines=False)
df = df[["class","order","family","genus","species"]]
df.reset_index(drop=True,inplace=True)

ll = list(df['class'].unique()) + list(df['order'].unique()) + list(df['family'].unique()) + list(df['genus'].unique()) + list(df['species'].unique())
ll

mapper = {}
counter = 0
for i in (ll):
    mapper[i] = counter
    counter += 1

df['class'] = df['class'].map(mapper)
df['order'] = df.order.map(mapper)
df['family'] = df.family.map(mapper)
df['genus'] = df.genus.map(mapper)
df['species'] = df.species.map(mapper)

for inp_col in range(df.shape[1]-1):

    #FIND THE MAXIMUM VOCABULARY SIZE OF THE INPUT.
    idim = df.max(axis=0)[inp_col] + 1

    #MODEL DEFINITION
    model = Sequential()
    model.add(Embedding(input_dim = idim ,output_dim = int (np.ceil((np.sum(df.max(axis=0))**(1/4.0))) ),input_length=1,name="embedding"))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    model.compile('rmsprop','mae')
    
    model.fit(df.values[:,inp_col],df.values[:,inp_col+1] ,epochs=100)
    t = model.get_layer("embedding").get_weights()[0]
    pickle.dump(t, open(df.columns[inp_col]+"-"+df.columns[inp_col+1]+".pkl","wb"), protocol=2)
