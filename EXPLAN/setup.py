import os
import requests
import zipfile

# Downloading YaDT C4.5 decision tree library
url = 'http://pages.di.unipi.it/ruggieri/YaDT/YaDT2.2.0.zip'
r = requests.get(url, allow_redirects=True)
open('YaDT2.2.0.zip', 'wb').write(r.content)

# Extracting YaDT library in the root directory
with zipfile.ZipFile('YaDT2.2.0.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
os.remove('YaDT2.2.0.zip')

# Making dTcmd file executable
os.system("chmod +x ./yadt/dTcmd")

# Removing any .pyc file
os.system("find . -name '*.pyc' -exec rm -f {} \;")

# Creating a directory for the results
os.makedirs("experiments")

print("Setup is complete!")

