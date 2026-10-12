import automl

datapath = ""
pipeline = Automl()
pipeline.fit(datapath)
pipeline.eval()
predictions = pipeline.predict(datapath)