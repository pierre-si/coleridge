# Creating a dataset
1. In input/ create a folder containing the files you want to upload
2. Run kaggle datasets init -p /path/to/dataset to generate a metadata file
3. Add your dataset’s metadata to the generated file, datapackage.json (title, id)
4. Run kaggle datasets create --dir-mode zip -p /path/to/dataset to create the dataset
→ il semblerait que les fichiers zip soient automatiquement décompressés lors de leur utilisation par un notebook, contrairement aux tar.  

