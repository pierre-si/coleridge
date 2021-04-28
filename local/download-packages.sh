# to run in an environment with python 3.7
pip download -d datasets/ datasets

pip download -d seqeval/ seqeval
cd seqeval
pip wheel seqeval-1.2.2.tar.gz
rm seqeval-1.2.2.tar.gz

cd ..
pip download -d spacy spacy
cd spacy
pip wheel smart_open-3.0.0.tar.gz
rm smart_open-3.0.0.tar.gz
# other language models at https://github.com/explosion/spacy-models/releases
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl
