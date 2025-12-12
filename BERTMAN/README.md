# How to run the app

## Install dependencies

### Root folder

Run _npm install_ in the root folder.

_Optional_: run _npm audit fix_ if there are some vulnerabilities due to updated packages.

### Backend folder

Navigate to _backend_ and run _pip install -r requirements.txt_

### Frontend folder

Navigate to _frontend_ and run _npm install_

# Run the app

## Run the backend

Go to [this link](https://drive.google.com/file/d/1LQ7FOtOT2Zuyx41NEkfyw9-rox4cl-Mu/view?usp=sharing) and download the fine-tuned BERT model, then put it in _/backend/data/bert_fine_tuned_model_.

Navigate to _backend_ and run _flask --app app run_.

Note that it will run in port 5000 by default. Make sure that this is respected or change the port in the frontend API calls.

## Run the frontend

Make sure that you have at least **NodeJS 22** installed.

Navigate to _frontend_ and run _npm run dev_.

Note that it will run in port 5173 by default.

### Optional

To build the app, run _npm run build_. Note that this was not yet tested.

# Test the model

To test the BERT model, you can use the _/backend/data/example.json_ file. Select a random neighborhood and run the model. Note that, if it works correctly, it will save the results in the database, which has a UNIQUE clause on the _link_ field.
